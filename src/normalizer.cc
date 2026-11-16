// Copyright 2016 Google Inc.
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
// limitations under the License.!

#include "normalizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "ret_check.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model_converters.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {

Normalizer::Normalizer(const NormalizerSpec& spec,
                       const TrainerSpec& trainer_spec)
    : status_(absl::OkStatus()) {
  Init(spec, trainer_spec.treat_whitespace_as_suffix());
}

Normalizer::Normalizer(const NormalizerSpec& spec) : status_(absl::OkStatus()) {
  Init(spec, /*treat_whitespace_as_suffix=*/false);
}

Normalizer::~Normalizer() = default;

void Normalizer::Init(const NormalizerSpec& spec,
                      bool treat_whitespace_as_suffix) {
  ModelProto dummy_proto;
  *dummy_proto.mutable_normalizer_spec() = spec;

  auto* piece = dummy_proto.add_pieces();
  piece->set_piece("<unk>");
  piece->set_type(ModelProto::SentencePiece::UNKNOWN);
  piece->set_score(0.0f);

  auto* trainer_spec = dummy_proto.mutable_trainer_spec();
  trainer_spec->set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);
  trainer_spec->set_bos_id(-1);
  trainer_spec->set_eos_id(-1);
  trainer_spec->set_pad_id(-1);
  trainer_spec->set_treat_whitespace_as_suffix(treat_whitespace_as_suffix);

  auto fb_or = lite::ToFlatbuffer(dummy_proto);
  if (!fb_or.ok()) {
    status_ = fb_or.status();
    return;
  }

  lite_processor_ = std::make_unique<lite::SentencePieceLiteProcessor>(
      std::make_shared<std::string>(std::move(*fb_or)));
  if (lite_processor_->status() != lite::StatusCode::kOk) {
    status_ = absl::Status(
        static_cast<absl::StatusCode>(lite_processor_->status()),
        "Failed to initialize SentencePieceLiteProcessor for Normalizer.");
    lite_processor_.reset();
  }
}

absl::Status Normalizer::Normalize(absl::string_view input,
                                   std::string* normalized,
                                   std::vector<size_t>* norm_to_orig) const {
  if (norm_to_orig) norm_to_orig->clear();
  normalized->clear();

  if (input.empty()) {
    return absl::OkStatus();
  }

  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK(lite_processor_);

  const auto code = lite_processor_->Normalize(input, normalized, norm_to_orig);
  if (code != lite::StatusCode::kOk) {
    return absl::Status(static_cast<absl::StatusCode>(code),
                        "Normalization failed.");
  }
  return absl::OkStatus();
}

std::string Normalizer::Normalize(absl::string_view input) const {
  std::string normalized;
  Normalize(input, &normalized, nullptr).IgnoreError();
  return normalized;
}

// static
std::string Normalizer::EncodePrecompiledCharsMap(
    absl::string_view trie_blob, absl::string_view normalized) {
  // <trie size(4byte)><double array trie><normalized string>
  std::string blob;
  blob.append(string_util::EncodePOD<uint32_t>(trie_blob.size()));
  blob.append(trie_blob.data(), trie_blob.size());

  if constexpr (absl::endian::native == absl::endian::big) {
    uint32_t* data = reinterpret_cast<uint32_t*>(blob.data());
    for (int i = 0; i < blob.size() / 4; ++i)
      data[i] = absl::gbswap_32(data[i]);
  }

  blob.append(normalized.data(), normalized.size());

  return blob;
}

// static
absl::Status Normalizer::DecodePrecompiledCharsMap(
    absl::string_view blob, absl::string_view* trie_blob,
    absl::string_view* normalized, std::string* buffer) {
  uint32_t trie_blob_size = 0;
  if (blob.size() <= sizeof(trie_blob_size) ||
      !string_util::DecodePOD<uint32_t>(
          absl::string_view(blob.data(), sizeof(trie_blob_size)),
          &trie_blob_size)) {
    return absl::InternalError("Blob for normalization rule is broken.");
  }

  if constexpr (absl::endian::native == absl::endian::big) {
    trie_blob_size = absl::gbswap_32(trie_blob_size);
  }

  if (trie_blob_size >= blob.size() - sizeof(trie_blob_size)) {
    return absl::InternalError("Trie data size exceeds the input blob size.");
  }

  // Dart unit_size is 4 and blob size in units must be a multiple of 256.
  if (trie_blob_size < 1024 || (trie_blob_size & 0x3FF) != 0) {
    return absl::InternalError("Trie data size is not divisible by 1024.");
  }

  blob.remove_prefix(sizeof(trie_blob_size));

  if constexpr (absl::endian::native == absl::endian::big) {
    RET_CHECK(buffer);
    buffer->assign(blob.data(), trie_blob_size);
    uint32_t* data =
        reinterpret_cast<uint32_t*>(const_cast<char*>(buffer->data()));
    for (int i = 0; i < buffer->size() / 4; ++i)
      data[i] = absl::gbswap_32(data[i]);
    *trie_blob = absl::string_view(buffer->data(), trie_blob_size);
  } else {
    *trie_blob = absl::string_view(blob.data(), trie_blob_size);
  }

  blob.remove_prefix(trie_blob_size);
  *normalized = absl::string_view(blob.data(), blob.size());

  if (normalized->empty() || normalized->back() != '\0') {
    return absl::InternalError("normalized block must be null terminated.");
  }

  return absl::OkStatus();
}

}  // namespace normalizer
}  // namespace sentencepiece
