// Copyright 2026 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "model_proto_manager.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "ret_check.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_lite_generated.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"
#include "util.h"

namespace sentencepiece {
namespace {

constexpr absl::string_view kNullPiece("\0", 1);

}  // namespace

void ModelProtoManager::Reset() {
  lite_processor_.reset();
  trainer_spec_.Clear();
  normalizer_spec_.Clear();
  flatbuffer_view_ = absl::string_view();
  precompiled_charsmap_ = absl::string_view();
  if constexpr (std::endian::native == std::endian::big) {
    big_endian_charsmap_.clear();
  }
  min_score_ = 0.0f;
  has_trainer_spec_ = false;
  has_normalizer_spec_ = false;
  has_precompiled_charsmap_ = false;
  absl::MutexLock lock(&mutex_);
  cached_model_proto_.reset();
}

absl::Status ModelProtoManager::ValidateModel(ModelProto* model_proto,
                                              int* unk_id) {
  RET_CHECK(model_proto != nullptr);
  RET_CHECK(unk_id != nullptr);

  // Workaround for https://github.com/google/sentencepiece/issues/1308
  // Third-party conversion scripts may erroneously insert a raw null-byte
  // piece ("\0") even when byte_fallback is enabled (where "<0x00>" is
  // already defined). Mark it as UNUSED so it is excluded from trie building
  // and encoding.
  const bool is_char_model =
      model_proto->trainer_spec().model_type() == TrainerSpec::CHAR;
  std::vector<bool> byte_found(256, false);
  *unk_id = -1;
  for (int i = 0; i < model_proto->pieces_size(); ++i) {
    auto* sp = model_proto->mutable_pieces(i);
    static constexpr size_t kMaxPieceSize = 4096;
    if (sp->piece().empty()) {
      return absl::InternalError("piece must not be empty.");
    }
    if (sp->piece().size() > kMaxPieceSize) {
      return absl::InternalError("piece is too long.");
    }
    if (std::isnan(sp->score()) || std::isinf(sp->score())) {
      return absl::InternalError(
          absl::StrCat("piece [", i, "] has an invalid (NaN/Inf) score."));
    }
    if (is_char_model && sp->type() == ModelProto::SentencePiece::NORMAL &&
        (!string_util::IsStructurallyValid(sp->piece()) ||
         string_util::OneCharLen(sp->piece().data()) != sp->piece().size())) {
      return absl::InternalError(
          absl::StrCat("piece [", i, "] ('", sp->piece(),
                       "') in CHAR model must be a single Unicode character."));
    }
    if (model_proto->trainer_spec().byte_fallback() &&
        sp->piece() == kNullPiece) {
      LOG(WARNING)
          << "Piece at index " << i
          << " is a null character (\\0) and byte_fallback is enabled. "
          << "Treating it as an UNUSED piece.";
      sp->set_type(ModelProto::SentencePiece::UNUSED);
    }
    if (sp->type() == ModelProto::SentencePiece::UNKNOWN) {
      if (*unk_id >= 0) {
        return absl::InternalError("unk is already defined.");
      }
      if (!model_proto->trainer_spec().unk_piece().empty() &&
          sp->piece() != model_proto->trainer_spec().unk_piece()) {
        return absl::InternalError(
            absl::StrCat("unk_piece mismatch: ", sp->piece(), " vs ",
                         model_proto->trainer_spec().unk_piece()));
      }
      *unk_id = i;
    }
    if (sp->type() == ModelProto::SentencePiece::BYTE) {
      if (!model_proto->trainer_spec().byte_fallback()) {
        return absl::InternalError(
            absl::StrCat("byte piece ", sp->piece(),
                         " is found although `byte_fallback` is false."));
      }
      const int byte = PieceToByte(sp->piece());
      if (0 <= byte && byte < 256) {
        byte_found[byte] = true;
      } else {
        return absl::InternalError(
            absl::StrCat("byte piece ", sp->piece(), " is invalid."));
      }
    }
  }

  if (*unk_id == -1 && !model_proto->trainer_spec().byte_fallback()) {
    return absl::InternalError("unk is not defined.");
  }
  if (model_proto->trainer_spec().byte_fallback()) {
    if (std::find(byte_found.begin(), byte_found.end(), false) !=
        byte_found.end()) {
      return absl::InternalError(
          "there are not 256 byte pieces although `byte_fallback` is true.");
    }
  }

  if (model_proto->has_denormalizer_spec() &&
      !model_proto->denormalizer_spec().precompiled_charsmap().empty()) {
    LOG(WARNING) << "ModelProto contains a non-empty denormalizer_spec, which "
                    "is deprecated and ignored during Decode in SentencePiece "
                    "0.3.0.";
  }

  return absl::OkStatus();
}

void ModelProtoManager::MaybeRewriteModelProto(ModelProto* model_proto,
                                               int unk_id) {
  if (model_proto == nullptr) return;

  if (!model_proto->has_normalizer_spec()) {
    (void)model_proto->mutable_normalizer_spec();
  }

  if (model_proto->has_trainer_spec() &&
      model_proto->trainer_spec().unk_id() != unk_id && unk_id >= 0) {
    model_proto->mutable_trainer_spec()->set_unk_id(unk_id);
  }
}

void ModelProtoManager::PopulateMinScore() {
  min_score_ = 0.0f;
  if (!lite_processor_ || trainer_spec_.model_type() == TrainerSpec::BPE) {
    return;
  }
  const int vocab_size = static_cast<int>(lite_processor_->vocab_size());
  bool score_initialized = false;
  for (int i = 0; i < vocab_size; ++i) {
    if (lite_processor_->piece_type(i) == ModelProto::SentencePiece::NORMAL) {
      const float s = lite_processor_->GetScore(i);
      if (!score_initialized || s < min_score_) {
        min_score_ = s;
        score_initialized = true;
      }
    }
  }
}

void ModelProtoManager::PopulatePrecompiledCharsmapFromFlatbuffer(
    absl::string_view fb_bytes) {
  const auto* fb_model = ::sentencepiece::lite::GetModelProto(fb_bytes.data());
  if (fb_model != nullptr && fb_model->normalizer_spec() != nullptr &&
      fb_model->normalizer_spec()->precompiled_charsmap() != nullptr) {
    const auto* charsmap = fb_model->normalizer_spec()->precompiled_charsmap();
    const absl::string_view fb_view(
        reinterpret_cast<const char*>(charsmap->data()), charsmap->size());
    if constexpr (std::endian::native == std::endian::big) {
      big_endian_charsmap_.assign(fb_view.data(), fb_view.size());
      lite::SwapPrecompiledCharsmapEndian(&big_endian_charsmap_,
                                          /*from_little_endian=*/false);
      precompiled_charsmap_ = big_endian_charsmap_;
    } else {
      precompiled_charsmap_ = fb_view;
    }
  }
}

void ModelProtoManager::PopulateMetadataFromFlatbuffer(
    absl::string_view fb_bytes) {
  const auto* fb_model = ::sentencepiece::lite::GetModelProto(fb_bytes.data());
  if (fb_model == nullptr || !lite_processor_) return;

  has_trainer_spec_ = true;
  const auto fb_model_type = fb_model->model_type();
  switch (fb_model_type) {
    case ::sentencepiece::lite::ModelType_BPE:
      trainer_spec_.set_model_type(TrainerSpec::BPE);
      break;
    case ::sentencepiece::lite::ModelType_WORD:
      trainer_spec_.set_model_type(TrainerSpec::WORD);
      break;
    case ::sentencepiece::lite::ModelType_CHAR:
      trainer_spec_.set_model_type(TrainerSpec::CHAR);
      break;
    case ::sentencepiece::lite::ModelType_UNIGRAM:
    default:
      trainer_spec_.set_model_type(TrainerSpec::UNIGRAM);
      break;
  }

  const int vocab_size = static_cast<int>(lite_processor_->vocab_size());
  trainer_spec_.set_vocab_size(vocab_size);

  const int unk_id = lite_processor_->unk_id();
  trainer_spec_.set_unk_id(unk_id);
  if (unk_id >= 0 && unk_id < vocab_size) {
    const absl::string_view unk_piece = lite_processor_->IdToPiece(unk_id);
    if (unk_piece != trainer_spec_.unk_piece()) {
      trainer_spec_.set_unk_piece(std::string(unk_piece));
    }
  }

  const int bos_id = lite_processor_->bos_id();
  trainer_spec_.set_bos_id(bos_id);
  if (bos_id >= 0 && bos_id < vocab_size) {
    const absl::string_view bos_piece = lite_processor_->IdToPiece(bos_id);
    if (bos_piece != trainer_spec_.bos_piece()) {
      trainer_spec_.set_bos_piece(std::string(bos_piece));
    }
  }

  const int eos_id = lite_processor_->eos_id();
  trainer_spec_.set_eos_id(eos_id);
  if (eos_id >= 0 && eos_id < vocab_size) {
    const absl::string_view eos_piece = lite_processor_->IdToPiece(eos_id);
    if (eos_piece != trainer_spec_.eos_piece()) {
      trainer_spec_.set_eos_piece(std::string(eos_piece));
    }
  }

  const int pad_id = lite_processor_->pad_id();
  trainer_spec_.set_pad_id(pad_id);
  if (pad_id >= 0 && pad_id < vocab_size) {
    const absl::string_view pad_piece = lite_processor_->IdToPiece(pad_id);
    if (pad_piece != trainer_spec_.pad_piece()) {
      trainer_spec_.set_pad_piece(std::string(pad_piece));
    }
  }

  if (fb_model->unk_surface() != nullptr &&
      fb_model->unk_surface()->string_view() != trainer_spec_.unk_surface()) {
    trainer_spec_.set_unk_surface(fb_model->unk_surface()->str());
  }

  const int byte0_id = lite_processor_->PieceToId("<0x00>");
  const bool byte_fallback =
      byte0_id != -1 &&
      lite_processor_->piece_type(byte0_id) == ModelProto::SentencePiece::BYTE;
  if (byte_fallback) {
    trainer_spec_.set_byte_fallback(true);
  }

  if (fb_model->normalizer_spec() != nullptr) {
    const auto* fb_ns = fb_model->normalizer_spec();
    has_normalizer_spec_ = true;
    if (fb_ns->name() != nullptr) {
      normalizer_spec_.set_name(fb_ns->name()->str());
    }
    normalizer_spec_.set_add_dummy_prefix(fb_ns->add_dummy_prefix());
    normalizer_spec_.set_remove_extra_whitespaces(
        fb_ns->remove_extra_whitespaces());
    normalizer_spec_.set_escape_whitespaces(fb_ns->escape_whitespaces());
    if (fb_ns->treat_whitespace_as_suffix()) {
      trainer_spec_.set_treat_whitespace_as_suffix(true);
    }
    if (fb_ns->precompiled_charsmap() != nullptr) {
      has_precompiled_charsmap_ = true;
    }
  }

  PopulatePrecompiledCharsmapFromFlatbuffer(fb_bytes);
  PopulateMinScore();
}

absl::Status ModelProtoManager::Load(std::unique_ptr<ModelProto> model_proto) {
  Reset();

  if (!model_proto) {
    return absl::InvalidArgumentError("model_proto is null.");
  }

  int unk_id = -1;
  ABSL_RETURN_IF_ERROR(ValidateModel(model_proto.get(), &unk_id));

  // Always retain trainer_spec and normalizer_spec, but strip
  // `precompiled_charsmap` from `normalizer_spec_` so it is referenced
  // directly from the FlatBuffer via `precompiled_charsmap_`.
  has_normalizer_spec_ = model_proto->has_normalizer_spec();
  if (has_normalizer_spec_) {
    normalizer_spec_ = model_proto->normalizer_spec();
    has_precompiled_charsmap_ = normalizer_spec_.has_precompiled_charsmap();
    normalizer_spec_.clear_precompiled_charsmap();
  }

  // Rewrite model_proto in-place for FlatBuffer compilation; `model_proto` and
  // its vocabulary pieces are freed immediately upon returning from Load().
  MaybeRewriteModelProto(model_proto.get(), unk_id);

  has_trainer_spec_ = model_proto->has_trainer_spec();
  if (has_trainer_spec_) {
    trainer_spec_ = model_proto->trainer_spec();
  }

  auto fb_or =
      lite::ToFlatbuffer(*model_proto, {.skip_char_bigrams = false,
                                        .treat_null_byte_as_unused = true,
                                        .allow_unsupported_model_type = true});
  if (!fb_or.ok()) {
    Reset();
    return fb_or.status();
  }

  // Lifetime invariant: `fb_buffer` is a heap-allocated `std::string` managed
  // by a `std::shared_ptr`. Moving `fb_buffer` into `lite_processor_` below
  // transfers shared ownership without reallocating or moving the underlying
  // string bytes, so `flatbuffer_view_` and `precompiled_charsmap_` remain
  // valid for the entire lifetime of `lite_processor_` (until `Reset()`).
  auto fb_buffer = std::make_shared<std::string>(std::move(*fb_or));
  flatbuffer_view_ = *fb_buffer;
  PopulatePrecompiledCharsmapFromFlatbuffer(flatbuffer_view_);

  lite_processor_ =
      std::make_unique<lite::SentencePieceLiteProcessor>(std::move(fb_buffer));
  if (lite_processor_->status() != lite::StatusCode::kOk) {
    absl::Status status =
        ToAbslStatus(lite_processor_->status(),
                     "Failed to initialize SentencePieceLiteProcessor.");
    Reset();
    return status;
  }

  PopulateMinScore();
  return absl::OkStatus();
}

absl::Status ModelProtoManager::FinalizeFlatbufferLoad(
    absl::string_view fb_view, absl::string_view error_msg) {
  if (lite_processor_->status() != lite::StatusCode::kOk) {
    absl::Status status = ToAbslStatus(lite_processor_->status(), error_msg);
    Reset();
    return status;
  }

  const auto* fb_model = ::sentencepiece::lite::GetModelProto(fb_view.data());
  if (fb_model == nullptr || fb_model->char_bigram_trie_blob() == nullptr ||
      fb_model->char_bigram_trie_blob()->size() == 0) {
    Reset();
    return absl::InvalidArgumentError(
        "FlatBuffer model is missing char_bigram_trie_blob (must be built "
        "with .skip_char_bigrams = false).");
  }

  if (fb_model->model_type() == ::sentencepiece::lite::ModelType_CHAR) {
    const int vocab_size = static_cast<int>(lite_processor_->vocab_size());
    for (int i = 0; i < vocab_size; ++i) {
      if (lite_processor_->piece_type(i) == ModelProto::SentencePiece::NORMAL) {
        const absl::string_view piece = lite_processor_->IdToPiece(i);
        if (!string_util::IsStructurallyValid(piece) ||
            string_util::OneCharLen(piece.data()) != piece.size()) {
          Reset();
          return absl::InternalError(absl::StrCat(
              "piece [", i, "] ('", piece,
              "') in CHAR model must be a single Unicode character."));
        }
      }
    }
  }

  flatbuffer_view_ = fb_view;
  PopulateMetadataFromFlatbuffer(flatbuffer_view_);
  return absl::OkStatus();
}

absl::Status ModelProtoManager::LoadFromSerializedFlatbuffer(
    absl::string_view serialized) {
  Reset();
  if (serialized.empty()) {
    return absl::InvalidArgumentError("Serialized FlatBuffer is empty.");
  }

  auto fb_buffer = std::make_shared<std::string>(serialized);
  const absl::string_view fb_view(*fb_buffer);
  lite_processor_ =
      std::make_unique<lite::SentencePieceLiteProcessor>(std::move(fb_buffer));
  return FinalizeFlatbufferLoad(fb_view,
                                "Failed to load serialized FlatBuffer model.");
}

absl::Status ModelProtoManager::LoadFromFlatbufferView(
    absl::string_view flatbuffer_view) {
  Reset();
  if (flatbuffer_view.empty()) {
    return absl::InvalidArgumentError("FlatBuffer view is empty.");
  }

  if (reinterpret_cast<uintptr_t>(flatbuffer_view.data()) % 8 != 0) {
    return LoadFromSerializedFlatbuffer(flatbuffer_view);
  }

  lite_processor_ = std::make_unique<lite::SentencePieceLiteProcessor>(
      std::string_view(flatbuffer_view.data(), flatbuffer_view.size()));
  return FinalizeFlatbufferLoad(flatbuffer_view,
                                "Failed to load FlatBuffer model from view.");
}

const ModelProto& ModelProtoManager::model_proto() const {
  if (!lite_processor_) {
    return ModelProto::default_instance();
  }

  absl::MutexLock lock(&mutex_);
  if (!cached_model_proto_) {
    auto proto = std::make_unique<ModelProto>();
    if (has_trainer_spec_) {
      *proto->mutable_trainer_spec() = trainer_spec_;
    }
    if (has_normalizer_spec_) {
      *proto->mutable_normalizer_spec() = normalizer_spec_;
      if (has_precompiled_charsmap_) {
        proto->mutable_normalizer_spec()->set_precompiled_charsmap(
            precompiled_charsmap_.data(), precompiled_charsmap_.size());
      }
    }

    const int vocab_size = static_cast<int>(lite_processor_->vocab_size());
    proto->mutable_pieces()->Reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      auto* sp = proto->add_pieces();
      const auto piece_view = lite_processor_->IdToPiece(i);
      sp->set_piece(piece_view.data(), piece_view.size());

      // In proto2 (`sentencepiece_model.proto`), `set_type(NORMAL)` and
      // `set_score(0.0f)` set explicit field-presence bits (`has_type` /
      // `has_score`) in `SerializeAsString()`. To preserve byte-exact wire
      // compatibility with `ModelProto::SerializeAsString()` (where NORMAL
      // pieces omit `type` and always set `score`, while non-NORMAL pieces set
      // `type` and omit `score` when 0.0f), we set only the non-default fields.
      const auto type = static_cast<ModelProto::SentencePiece::Type>(
          lite_processor_->piece_type(i));
      const float score = lite_processor_->GetScore(i);
      if (type == ModelProto::SentencePiece::NORMAL) {
        sp->set_score(score);
      } else {
        sp->set_type(type);
        if (score != 0.0f) {
          sp->set_score(score);
        }
      }
    }
    cached_model_proto_ = std::move(proto);
  }
  return *cached_model_proto_;
}

std::string ModelProtoManager::serialized_model_proto() const {
  if (!lite_processor_) {
    return "";
  }
  return model_proto().SerializeAsString();
}

}  // namespace sentencepiece
