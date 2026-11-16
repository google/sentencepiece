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
// limitations under the License.

#include "sentencepiece_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "filesystem.h"
#include "google/protobuf/arena.h"
#include "model_proto_manager.h"
#include "ret_check.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "util.h"

using ::google::protobuf::Arena;

namespace sentencepiece {
namespace {

// Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK).
constexpr absl::string_view kSpaceSymbol = "\xe2\x96\x81";

// REPLACEMENT CHARACTER (U+FFFD) in UTF-8.
constexpr absl::string_view kReplacementCharacter = "\xef\xbf\xbd";

// maximum nbest or sampling size.
constexpr int kMaxNBestSize = 512;

// Piece-type checks for hot encode/decode loops. Unlike the public
// SentencePieceProcessor::Is*() accessors, these skip the per-call status()
// validation, so callers must have validated status() beforehand.
inline bool IsControlId(const lite::SentencePieceLiteProcessor& processor,
                        int id) {
  return processor.piece_type(id) == ModelProto::SentencePiece::CONTROL;
}

inline bool IsByteId(const lite::SentencePieceLiteProcessor& processor,
                     int id) {
  return processor.piece_type(id) == ModelProto::SentencePiece::BYTE;
}

inline bool IsUnknownId(const lite::SentencePieceLiteProcessor& processor,
                        int id) {
  return id == processor.unk_id() ||
         processor.piece_type(id) == ModelProto::SentencePiece::UNKNOWN;
}

inline bool IsUnusedId(const lite::SentencePieceLiteProcessor& processor,
                       int id) {
  return processor.piece_type(id) == ModelProto::SentencePiece::UNUSED;
}

}  // namespace

SentencePieceProcessor::SentencePieceProcessor() = default;
SentencePieceProcessor::~SentencePieceProcessor() = default;

absl::Status SentencePieceProcessor::Load(absl::string_view filename) {
  lite_processor_ = nullptr;
  model_proto_manager_.reset();
  auto model_proto = std::make_unique<ModelProto>();
  status_ = io::LoadModelProto(filename, model_proto.get());
  if (!status_.ok()) return status_;
  return Load(std::move(model_proto));
}

void SentencePieceProcessor::LoadOrDie(absl::string_view filename) {
  CHECK_OK(Load(filename));
}

absl::Status SentencePieceProcessor::Load(const ModelProto& model_proto) {
  auto model_proto_copy = std::make_unique<ModelProto>();
  *model_proto_copy = model_proto;
  return Load(std::move(model_proto_copy));
}

absl::Status SentencePieceProcessor::LoadFromSerializedProto(
    absl::string_view serialized) {
  lite_processor_ = nullptr;
  model_proto_manager_.reset();
  auto model_proto = std::make_unique<ModelProto>();
  if (!model_proto->ParseFromArray(serialized.data(), serialized.size())) {
    status_ = absl::InternalError(
        "model_proto->ParseFromArray(serialized.data(), serialized.size()) "
        "failed.");
    return status_;
  }
  return Load(std::move(model_proto));
}

absl::Status SentencePieceProcessor::LoadFromSerializedFlatbuffer(
    absl::string_view serialized) {
  lite_processor_ = nullptr;
  model_proto_manager_ = std::make_unique<ModelProtoManager>();
  status_ = model_proto_manager_->LoadFromSerializedFlatbuffer(serialized);
  if (!status_.ok()) {
    model_proto_manager_.reset();
    return status_;
  }
  return FinishLoad();
}

absl::Status SentencePieceProcessor::LoadFromFlatbufferView(
    absl::string_view flatbuffer_view) {
  lite_processor_ = nullptr;
  model_proto_manager_ = std::make_unique<ModelProtoManager>();
  status_ = model_proto_manager_->LoadFromFlatbufferView(flatbuffer_view);
  if (!status_.ok()) {
    model_proto_manager_.reset();
    return status_;
  }
  return FinishLoad();
}

absl::Status SentencePieceProcessor::Load(
    std::unique_ptr<ModelProto> model_proto) {
  RET_CHECK(model_proto != nullptr);
  lite_processor_ = nullptr;
  model_proto_manager_ = std::make_unique<ModelProtoManager>();
  status_ = model_proto_manager_->Load(std::move(model_proto));
  if (!status_.ok()) {
    model_proto_manager_.reset();
    return status_;
  }
  return FinishLoad();
}

absl::Status SentencePieceProcessor::FinishLoad() {
  lite_processor_ = model_proto_manager_->lite_processor();
  const auto& ts = model_proto_manager_->trainer_spec();
  model_type_ = ts.model_type();
  unk_piece_ = ts.unk_piece().empty() ? kDefaultUnkPiece : ts.unk_piece();
  bos_piece_ = ts.bos_piece().empty() ? kDefaultBosPiece : ts.bos_piece();
  eos_piece_ = ts.eos_piece().empty() ? kDefaultEosPiece : ts.eos_piece();
  unk_surface_ = ts.has_unk_surface() ? ts.unk_surface() : kDefaultUnkSurface;
  treat_whitespace_as_suffix_ = ts.treat_whitespace_as_suffix();
  if (model_proto_manager_->has_normalizer_spec()) {
    const auto& ns = model_proto_manager_->normalizer_spec();
    add_dummy_prefix_ = ns.add_dummy_prefix();
    remove_extra_whitespaces_ = ns.remove_extra_whitespaces();
  } else {
    add_dummy_prefix_ = true;
    remove_extra_whitespaces_ = true;
  }

  byte_fallback_ =
      ts.byte_fallback() && IsByte(lite_processor_->PieceToId("<0x00>"));

  status_ = absl::OkStatus();
  return status_;
}

absl::Status SentencePieceProcessor::SetEncodeExtraOptions(
    absl::string_view extra_options) {
  return ParseExtraOptions(extra_options, &encode_extra_options_);
}

absl::Status SentencePieceProcessor::SetDecodeExtraOptions(
    absl::string_view extra_options) {
  return ParseExtraOptions(extra_options, &decode_extra_options_);
}

absl::Status SentencePieceProcessor::status() const {
  if (!lite_processor_ || lite_processor_->status() != lite::StatusCode::kOk) {
    return status_.ok()
               ? absl::FailedPreconditionError("Model is not initialized.")
               : status_;
  }
  return status_;
}

#define RET_CHECK_STATUS_STL(container)               \
  ABSL_RETURN_IF_ERROR(status());                     \
  RET_CHECK(container) << "output container is null"; \
  container->clear();

#define RET_CHECK_STATUS_PROTO(proto)         \
  ABSL_RETURN_IF_ERROR(status());             \
  RET_CHECK(proto) << "output proto is null"; \
  proto->Clear();

absl::Status SentencePieceProcessor::EncodeNormalizedInternal(
    absl::string_view normalized, std::vector<int>* ids,
    std::vector<std::string_view>* pieces) const {
  RET_CHECK(ids != nullptr);
  ids->clear();
  if (pieces != nullptr) pieces->clear();
  if (normalized.empty()) return absl::OkStatus();

  if (model_type_ == TrainerSpec::WORD) {
    return EncodeWordNormalized(normalized, ids, pieces);
  }
  return ToAbslStatus(
      lite_processor_->EncodeNormalized(normalized, ids, pieces),
      "EncodeNormalized failed.");
}

absl::Status SentencePieceProcessor::EncodeWordNormalized(
    absl::string_view normalized, std::vector<int>* ids,
    std::vector<std::string_view>* pieces) const {
  const int unk = lite_processor_->unk_id();
  for (const auto& w :
       SplitIntoWords(normalized, treat_whitespace_as_suffix_)) {
    int id = lite_processor_->PieceToId(w);
    if (id == -1 || IsControlId(*lite_processor_, id) ||
        IsUnusedId(*lite_processor_, id) || IsByteId(*lite_processor_, id) ||
        IsUnknownId(*lite_processor_, id)) {
      id = unk;
    }
    if (id == unk && byte_fallback_) {
      // Byte fallback convention (shared with `lite_processor_->EmitPiece`):
      // For an unknown span `w` decomposed into `w.size()` `<0xXX>` byte
      // tokens, leading byte tokens store an empty view `w.substr(0, 0)` and
      // only the final byte token stores the full span `w` so
      // `PopulateSentencePieceText` can locate the normalized span offset.
      for (size_t i = 0; i < w.size(); ++i) {
        const uint8_t b = static_cast<uint8_t>(w[i]);
        ids->push_back(lite_processor_->PieceToId(ByteToPiece(b)));
        if (pieces != nullptr) {
          pieces->push_back((i == w.size() - 1)
                                ? std::string_view(w.data(), w.size())
                                : std::string_view(w.data(), 0));
        }
      }
      continue;
    }
    if (id != unk || ids->empty() || ids->back() != unk) {
      ids->push_back(id);
      if (pieces != nullptr) {
        pieces->emplace_back(w.data(), w.size());
      }
    } else if (pieces != nullptr && !pieces->empty()) {
      auto& back = pieces->back();
      back = std::string_view(back.data(), back.size() + w.size());
    }
  }
  return absl::OkStatus();
}

void SentencePieceProcessor::PopulatePieceStrings(
    absl::Span<const int> ids, absl::Span<const std::string_view> piece_views,
    std::vector<std::string>* pieces) const {
  const bool has_unk_piece = HasUnkPieceOption();
  pieces->reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    if (IsByteId(*lite_processor_, id)) {
      pieces->emplace_back(lite_processor_->IdToPiece(id));
    } else if (IsUnknownId(*lite_processor_, id) && has_unk_piece) {
      pieces->emplace_back(unk_piece_);
    } else {
      pieces->emplace_back(piece_views[i]);
    }
  }
}

//////////////////////////////////////////////////////////////
// Simple API.
absl::Status SentencePieceProcessor::Encode(
    absl::string_view input, std::vector<std::string>* pieces) const {
  RET_CHECK_STATUS_STL(pieces);
  if (!input.empty()) {
    std::string normalized;
    ABSL_RETURN_IF_ERROR(
        Normalize(input, &normalized, /*norm_to_orig=*/nullptr));
    std::vector<int> ids;
    std::vector<std::string_view> piece_views;
    ABSL_RETURN_IF_ERROR(
        EncodeNormalizedInternal(normalized, &ids, &piece_views));
    PopulatePieceStrings(ids, piece_views, pieces);
  }
  return ApplyExtraOptions(encode_extra_options_, pieces);
}

absl::Status SentencePieceProcessor::Encode(absl::string_view input,
                                            std::vector<int>* ids) const {
  RET_CHECK_STATUS_STL(ids);
  if (!input.empty()) {
    if (model_type_ != TrainerSpec::WORD) {
      const auto code = lite_processor_->Encode(input, ids);
      ABSL_RETURN_IF_ERROR(ToAbslStatus(code, "Encode failed."));
    } else {
      std::string normalized;
      ABSL_RETURN_IF_ERROR(
          Normalize(input, &normalized, /*norm_to_orig=*/nullptr));
      ABSL_RETURN_IF_ERROR(
          EncodeWordNormalized(normalized, ids, /*pieces=*/nullptr));
    }
  }
  return ApplyExtraOptions(encode_extra_options_, ids);
}

absl::Status SentencePieceProcessor::Decode(
    absl::Span<const std::string> pieces, std::string* detokenized) const {
  absl::FixedArray<absl::string_view, 128> views(pieces.begin(), pieces.end());
  return DecodePiecesInternal(views, detokenized, /*spt=*/nullptr);
}

absl::Status SentencePieceProcessor::Decode(
    absl::Span<const absl::string_view> pieces,
    std::string* detokenized) const {
  return DecodePiecesInternal(pieces, detokenized, /*spt=*/nullptr);
}

absl::Status SentencePieceProcessor::Decode(absl::Span<const int> ids,
                                            std::string* detokenized) const {
  RET_CHECK_STATUS_STL(detokenized);
  if (ids.empty()) {
    return absl::OkStatus();
  }

  absl::Span<const int> active_ids = ids;
  std::vector<int> work_ids;
  if (!decode_extra_options_.empty()) {
    work_ids.assign(ids.begin(), ids.end());
    ABSL_RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, &work_ids));
    active_ids = work_ids;
  }

  const auto code = lite_processor_->Decode(
      lite::Span<const int>(active_ids.data(), active_ids.size()), detokenized);
  return ToAbslStatus(code, "Decode failed.");
}

absl::Status SentencePieceProcessor::NBestEncode(
    absl::string_view input, int nbest_size,
    std::vector<std::vector<std::string>>* pieces) const {
  RET_CHECK_STATUS_STL(pieces);

  Arena arena;
  auto* spt = Arena::Create<NBestSentencePieceText>(&arena);
  ABSL_RETURN_IF_ERROR(NBestEncode(input, nbest_size, spt));
  pieces->reserve(spt->nbests().size());
  for (const auto& nbest : spt->nbests()) {
    std::vector<std::string>& result = pieces->emplace_back();
    result.reserve(nbest.pieces().size());
    for (const auto& sp : nbest.pieces()) {
      result.emplace_back(sp.piece());
    }
  }

  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::NBestEncode(
    absl::string_view input, int nbest_size,
    std::vector<std::vector<int>>* ids) const {
  RET_CHECK_STATUS_STL(ids);

  Arena arena;
  auto* spt = Arena::Create<NBestSentencePieceText>(&arena);
  ABSL_RETURN_IF_ERROR(NBestEncode(input, nbest_size, spt));
  ids->reserve(spt->nbests().size());
  for (const auto& nbest : spt->nbests()) {
    std::vector<int>& result = ids->emplace_back();
    result.reserve(nbest.pieces().size());
    for (const auto& sp : nbest.pieces()) {
      result.emplace_back(sp.id());
    }
  }

  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::SampleEncode(
    absl::string_view input, int nbest_size, float alpha,
    std::vector<std::string>* pieces) const {
  RET_CHECK_STATUS_STL(pieces);

  Arena arena;
  auto* spt = Arena::Create<SentencePieceText>(&arena);
  ABSL_RETURN_IF_ERROR(SampleEncode(input, nbest_size, alpha, spt));
  pieces->reserve(spt->pieces().size());
  for (const auto& sp : spt->pieces()) {
    pieces->emplace_back(sp.piece());
  }

  return absl::OkStatus();
}

// Note: `nbest_size` is retained for API compatibility with upstream
// SentencePieceProcessor. `SentencePieceLiteProcessor::SampleNormalized`
// samples directly from the full lattice (Gumbel-Max DP for Unigram,
// BPE-dropout for BPE), which is equivalent to exact full-lattice sampling
// (`nbest_size = -1`).
absl::Status SentencePieceProcessor::SampleEncode(absl::string_view input,
                                                  int /*nbest_size*/,
                                                  float alpha,
                                                  std::vector<int>* ids) const {
  RET_CHECK_STATUS_STL(ids);
  RET_CHECK(std::isfinite(alpha));
  RET_CHECK_GE(alpha, 0.0f);
  RET_CHECK(model_type_ == TrainerSpec::UNIGRAM ||
            model_type_ == TrainerSpec::BPE)
      << "SampleEncode is not available for the current model.";

  if (input.empty()) {
    return ApplyExtraOptions(encode_extra_options_, ids);
  }

  std::string normalized;
  ABSL_RETURN_IF_ERROR(Normalize(input, &normalized, /*norm_to_orig=*/nullptr));

  auto* mt = random::GetRandomGenerator();
  auto uniform_sampler = [mt]() -> float {
    return absl::Uniform<float>(*mt, 0.0f, 1.0f);
  };

  const auto code = lite_processor_->SampleNormalized(
      normalized, alpha, uniform_sampler, ids, /*pieces=*/nullptr);
  ABSL_RETURN_IF_ERROR(ToAbslStatus(code, "SampleNormalized failed."));
  return ApplyExtraOptions(encode_extra_options_, ids);
}

absl::Status SentencePieceProcessor::PopulateSentencePieceText(
    absl::string_view input, absl::string_view normalized,
    absl::Span<const size_t> norm_to_orig, absl::Span<const int> ids,
    absl::Span<const std::string_view> pieces, SentencePieceText* spt) const {
  RET_CHECK_EQ(ids.size(), pieces.size());

  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    const std::string_view w = pieces[i];

    if (IsByteId(*lite_processor_, id)) {
      // `SentencePieceLiteProcessor` emits `<0xXX>` byte tokens for an unknown
      // span with the following `pieces` convention:
      // - One element of the byte run (the last token of the unknown character
      //   or span) holds a non-empty `string_view` pointing to the start of
      //   that unknown span inside `normalized`.
      // - The remaining byte tokens in the same unknown span hold empty
      //   `string_view(unk_span.data(), 0)` views.
      // We locate `norm_pos` from the first non-empty view (or `w.data()`) and
      // advance through `normalized` character-by-character so each UTF-8
      // character's original surface and `[begin, end)` byte offsets are
      // attached to the final `<0xXX>` token of that character.
      size_t run_end = i + 1;
      while (run_end < ids.size() && IsByteId(*lite_processor_, ids[run_end])) {
        ++run_end;
      }
      size_t norm_pos = 0;
      bool found_non_empty = false;
      for (size_t k = i; k < run_end; ++k) {
        if (!pieces[k].empty()) {
          RET_CHECK_GE(pieces[k].data(), normalized.data());
          norm_pos = static_cast<size_t>(pieces[k].data() - normalized.data());
          found_non_empty = true;
          break;
        }
      }
      if (!found_non_empty) {
        RET_CHECK_GE(w.data(), normalized.data());
        norm_pos = static_cast<size_t>(w.data() - normalized.data());
      }

      size_t k = i;
      while (k < run_end) {
        const size_t remaining = run_end - k;
        const size_t char_len = std::min<size_t>(
            remaining, std::max<size_t>(1, string_util::OneCharLen(
                                               normalized.data() + norm_pos)));
        RET_CHECK_LT(norm_pos, norm_to_orig.size());
        RET_CHECK_LT(norm_pos + char_len, norm_to_orig.size());
        const size_t char_orig_begin = norm_to_orig[norm_pos];
        const size_t char_orig_end = norm_to_orig[norm_pos + char_len];
        RET_CHECK_LE(char_orig_begin, input.size());
        RET_CHECK_LE(char_orig_end, input.size());
        RET_CHECK_LE(char_orig_begin, char_orig_end);
        const auto surface = absl::ClippedSubstr(
            input, char_orig_begin, char_orig_end - char_orig_begin);

        for (size_t b = 0; b < char_len; ++b) {
          auto* sp = spt->add_pieces();
          const int byte_id = ids[k + b];
          sp->set_id(byte_id);
          const absl::string_view byte_piece =
              lite_processor_->IdToPiece(byte_id);
          sp->set_piece(byte_piece.data(), byte_piece.size());
          if (b == char_len - 1) {
            sp->set_surface(surface.data(), surface.size());
            sp->set_begin(char_orig_begin);
            sp->set_end(char_orig_end);
          } else {
            sp->set_begin(char_orig_begin);
            sp->set_end(char_orig_begin);
          }
        }
        norm_pos += char_len;
        k += char_len;
      }
      i = run_end - 1;
      continue;
    }

    RET_CHECK_GE(w.data(), normalized.data());
    const size_t begin = static_cast<size_t>(w.data() - normalized.data());
    const size_t end = begin + w.size();
    RET_CHECK_LT(begin, norm_to_orig.size());
    RET_CHECK_LT(end, norm_to_orig.size());
    const size_t orig_begin = norm_to_orig[begin];
    const size_t orig_end = norm_to_orig[end];
    RET_CHECK_LE(orig_begin, input.size());
    RET_CHECK_LE(orig_end, input.size());
    RET_CHECK_LE(orig_begin, orig_end);

    auto* sp = spt->add_pieces();
    sp->set_id(id);

    if (IsControlId(*lite_processor_, id)) {
      sp->set_piece(w.data(), w.size());
      sp->set_begin(orig_begin);
      sp->set_end(orig_begin);
    } else {
      sp->set_piece(w.data(), w.size());
      const auto surface =
          absl::ClippedSubstr(input, orig_begin, orig_end - orig_begin);
      sp->set_surface(surface.data(), surface.size());
      sp->set_begin(orig_begin);
      sp->set_end(orig_end);
    }
  }

  spt->set_text(input.data(), input.size());
  ABSL_RETURN_IF_ERROR(ApplyExtraOptions(encode_extra_options_, spt));
  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::Encode(absl::string_view input,
                                            SentencePieceText* spt) const {
  RET_CHECK_STATUS_PROTO(spt);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  ABSL_RETURN_IF_ERROR(Normalize(input, &normalized, &norm_to_orig));

  std::vector<int> ids;
  std::vector<std::string_view> pieces;
  ABSL_RETURN_IF_ERROR(EncodeNormalizedInternal(normalized, &ids, &pieces));
  return PopulateSentencePieceText(input, normalized, norm_to_orig, ids, pieces,
                                   spt);
}

// =============================================================================
// NBestEncode Algorithm Overview (Unigram Only)
// =============================================================================
// `SentencePieceLiteProcessor` is a lattice-free runtime and does not
// materialize an explicit word lattice for A* search. Instead, `NBestEncode`
// (supported only for Unigram models) reconstructs the top-K segmentations by
// deterministically steering
// `SentencePieceLiteProcessor::SampleNormalized(normalized, alpha, sampler)`
// via custom `uniform_sampler` callbacks:
//
// 1. Deterministic Edge Steering via `uniform_sampler`:
//    In Unigram mode, `SampleNormalized` visits lattice transitions in a fixed,
//    input-determined order and perturbs each transition score by
//    `alpha * Gumbel(u) = -alpha * log(-log(u))`.
//    * Passing `kNeutralU = exp(-1.0) (~0.36788)` yields `Gumbel(u) == 0.0`
//      (exact zero noise, equivalent to deterministic Viterbi DP).
//    * Passing `kBlockU = 1e-6` with `kBlockAlpha = 10.0` for a specific call
//      index `idx` injects a heavy penalty (~-26.3) on that single
//      transition, effectively blocking edge `idx` while keeping all other
//      edges exact.
//
// 2. Multi-Phase Candidate Search Pipeline:
//    - Step 0 (1-Best & Call Counting):
//      Runs `SampleNormalized` with `alpha = 0.0` and `kNeutralU` to obtain the
//      exact 1-best segmentation (`candidates[0]`) and count `total_calls`
//      (total number of lattice edge evaluations).
//    - Phase 1 (Yen-Style Single-Edge Blocking Sweep):
//      Sweeps `target_edge` across `[0, total_calls)` and blocks one edge at a
//      time. Blocking an edge that does NOT lie on the 1-best path leaves the
//      1-best segmentation unchanged. Whenever blocking `target_edge` produces
//      a different segmentation, `target_edge` is identified as an active edge
//      on the 1-best path (`active_edges`). By Yen's k-shortest path theorem,
//      the highest-scoring deviation among all single-edge blocks of the 1-best
//      path is the exact 2nd-best segmentation. The sweep examines at most 160
//      edges, so this holds only when `total_calls <= 160`; on longer inputs
//      the edges are subsampled and the 2nd candidate is approximate.
//    - Phase 2 (Pairwise Blocking of Top Active Edges):
//      Sorts `active_edges` by descending score (smallest score drop = most
//      ambiguous subword boundaries) and blocks pairs `(e1, e2)` simultaneously
//      to recover exact 2-boundary deviations.
//    - Phase 3 (Localized Window Perturbation):
//      Blocks each top active edge `ei` and applies Gumbel noise only within a
//      local neighborhood (`|idx - ei| <= 6`) while holding the rest of the
//      sentence at `kNeutralU`, uncovering 3rd/4th-best splits at the same
//      boundary without disturbing unrelated spans.
//    - Phase 4 (Global Low-Temperature Sweep & Exact Re-ranking):
//      Performs a temperature-scheduled global Gumbel sweep to fill any
//      remaining slots up to `nbest_size`, then evaluates true model
//      log-likelihoods (`compute_score`) and sorts `candidates[1..]` in
//      descending score order.
absl::Status SentencePieceProcessor::NBestEncode(
    absl::string_view input, int nbest_size,
    NBestSentencePieceText* nbest_spt) const {
  RET_CHECK_STATUS_PROTO(nbest_spt);
  RET_CHECK_GT(nbest_size, 0) << "nbest_size must be > 0";
  RET_CHECK_LE(nbest_size, kMaxNBestSize)
      << "nbest_size must be nbest_size <= " << kMaxNBestSize;

  RET_CHECK(model_type_ == TrainerSpec::UNIGRAM)
      << "NBestEncode is not available for the current model.";

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  ABSL_RETURN_IF_ERROR(Normalize(input, &normalized, &norm_to_orig));

  struct Candidate {
    std::vector<int> ids;
    std::vector<std::string_view> pieces;
    float score = 0.0f;
  };

  // Retrieve cached min NORMAL piece score from ModelProtoManager (O(1)) so
  // UNKNOWN tokens receive the Viterbi penalty (`min_score - 10.0f`) instead
  // of their default proto score (`0.0f`).
  const float unk_score =
      model_proto_manager_ ? (model_proto_manager_->min_score() - 10.0f) : 0.0f;

  auto compute_score = [this, unk_score](absl::Span<const int> ids) -> float {
    float total = 0.0f;
    for (const int id : ids) {
      total += IsUnknownId(*lite_processor_, id)
                   ? unk_score
                   : lite_processor_->GetScore(id);
    }
    return total;
  };

  auto compute_hash = [](absl::Span<const int> ids) -> uint64_t {
    return absl::HashOf(absl::string_view(
        reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int)));
  };

  std::vector<Candidate> candidates;
  candidates.reserve(nbest_size + 16);
  absl::flat_hash_set<uint64_t> seen_ids;
  seen_ids.reserve(nbest_size * 2 + 16);

  // Deterministic `uniform_sampler` steering constants for SampleNormalized
  // (Gumbel-Max DP: perturbed_score = score - alpha * log(-log(u))):
  //   * kNeutralU = exp(-1.0) (~0.36787944): precomputed constant where
  //     -log(-log(exp(-1))) == -log(1) == 0.0 (exact zero Gumbel noise).
  //   * kBlockU   = 1e-6, kBlockAlpha = 10.0 -> ~-26.3 penalty on target edge.
  static constexpr float kNeutralU =
      0.36787944f;  // Precomputed std::exp(-1.0f)
  static constexpr float kBlockU = 1e-6f;
  static constexpr float kBlockAlpha = 10.0f;

  auto sample_into = [&](float alpha, auto&& sampler,
                         Candidate* out) -> absl::Status {
    const auto code = lite_processor_->SampleNormalized(
        normalized, alpha, sampler, &out->ids, &out->pieces);
    return ToAbslStatus(code, "SampleNormalized failed in NBestEncode.");
  };

  // Step 0: Obtain the deterministic 1-best segmentation and count
  // `total_calls` (lattice edge evaluations).
  int total_calls = 0;
  {
    Candidate best;
    if (nbest_size > 1 && !normalized.empty()) {
      auto count_sampler = [&total_calls]() -> float {
        ++total_calls;
        return kNeutralU;
      };
      ABSL_RETURN_IF_ERROR(sample_into(0.0f, count_sampler, &best));
    } else {
      ABSL_RETURN_IF_ERROR(
          EncodeNormalizedInternal(normalized, &best.ids, &best.pieces));
    }
    best.score = compute_score(best.ids);
    seen_ids.insert(compute_hash(best.ids));
    candidates.push_back(std::move(best));
  }

  if (nbest_size > 1 && !normalized.empty() && total_calls > 0) {
    std::vector<std::pair<float, int>> active_edges;
    Candidate scratch;

    // Runs `SampleNormalized` into `scratch` and appends it to `candidates` if
    // its token ID sequence has not been seen yet.
    auto sample_and_add_candidate = [&](float alpha,
                                        auto&& sampler) -> absl::Status {
      ABSL_RETURN_IF_ERROR(sample_into(alpha, sampler, &scratch));
      if (seen_ids.insert(compute_hash(scratch.ids)).second) {
        scratch.score = compute_score(scratch.ids);
        candidates.push_back(std::move(scratch));
      }
      return absl::OkStatus();
    };

    // Phase 1: Yen-style single-edge blocking.
    // Blocking one transition on the 1-best path while keeping all other
    // transitions at `kNeutralU` identifies which call indices belong to the
    // 1-best trajectory (`active_edges`). It finds the exact 2nd-best path
    // when every transition is tried (`total_calls <= 160`); otherwise the
    // transitions are subsampled and the 2nd-best path can be missed.
    const int max_block_trials = std::min(total_calls, 160);
    for (int b = 0; b < max_block_trials; ++b) {
      const int target_edge = static_cast<int>(
          (static_cast<int64_t>(b) * total_calls) / max_block_trials);
      int call_idx = 0;
      auto block_one_sampler = [&call_idx, target_edge]() -> float {
        const int idx = call_idx++;
        return (idx == target_edge) ? kBlockU : kNeutralU;
      };
      ABSL_RETURN_IF_ERROR(
          sample_into(kBlockAlpha, block_one_sampler, &scratch));
      if (scratch.ids != candidates[0].ids) {
        const float sc = compute_score(scratch.ids);
        active_edges.emplace_back(sc, target_edge);
        if (seen_ids.insert(compute_hash(scratch.ids)).second) {
          scratch.score = sc;
          candidates.push_back(std::move(scratch));
        }
      }
    }

    // When nbest_size == 2 and Phase 1 found at least one alternative path,
    // the best path found by Phase 1 is returned as the 2nd candidate without
    // further search.
    if (nbest_size > 2 || candidates.size() < 2) {
      std::sort(active_edges.begin(), active_edges.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

      // Phase 2: Pairwise blocking of top active 1-best edges.
      // Recovers 2-boundary deviations formed by combining the smallest-gap
      // subword boundaries.
      const int top_m = std::min(static_cast<int>(active_edges.size()),
                                 std::max(6, std::min(nbest_size, 10)));
      for (int i = 0; i < top_m; ++i) {
        for (int j = i + 1; j < top_m; ++j) {
          const int e1 = active_edges[i].second;
          const int e2 = active_edges[j].second;
          int call_idx = 0;
          auto block_pair_sampler = [&call_idx, e1, e2]() -> float {
            const int idx = call_idx++;
            return (idx == e1 || idx == e2) ? kBlockU : kNeutralU;
          };
          ABSL_RETURN_IF_ERROR(
              sample_and_add_candidate(kBlockAlpha, block_pair_sampler));
        }
      }

      absl::BitGen rng(std::seed_seq{42});

      // Phase 3: Localized Gumbel temperature perturbation around top active
      // edges with the 1-best edge blocked (`kBlockU`), recovering 3rd/4th-best
      // alternatives at the same subword boundary while keeping the rest of the
      // sentence unperturbed (`kNeutralU`).
      const int local_m = std::min(static_cast<int>(active_edges.size()), 5);
      for (int i = 0; i < local_m; ++i) {
        const int ei = active_edges[i].second;
        for (int sub = 0; sub < 6; ++sub) {
          const float alpha = 0.15f + 0.25f * static_cast<float>(sub);
          int call_idx = 0;
          auto local_sampler = [&call_idx, &rng, ei]() -> float {
            const int idx = call_idx++;
            if (idx == ei) return kBlockU;
            if (std::abs(idx - ei) <= 6) {
              return absl::Uniform<float>(rng, 0.0f, 1.0f);
            }
            return kNeutralU;
          };
          ABSL_RETURN_IF_ERROR(sample_and_add_candidate(alpha, local_sampler));
        }
      }

      // Phase 4: Global sweep using low-temperature Gumbel noise (alpha in
      // [0.02, 0.47]) to populate any remaining multi-boundary candidates up
      // to `nbest_size`.
      auto uniform_sampler = [&rng]() -> float {
        return absl::Uniform<float>(rng, 0.0f, 1.0f);
      };
      const int num_global =
          std::min(std::max(nbest_size * 6, 36), kMaxNBestSize * 4);
      for (int trial = 0; trial < num_global ||
                          (static_cast<int>(candidates.size()) < nbest_size &&
                           trial < num_global * 3);
           ++trial) {
        const float frac =
            static_cast<float>(trial) / static_cast<float>(num_global);
        const float sample_alpha = 0.02f + 0.45f * frac * std::sqrt(frac);
        ABSL_RETURN_IF_ERROR(
            sample_and_add_candidate(sample_alpha, uniform_sampler));
      }
    }

    if (candidates.size() > 1) {
      std::stable_sort(candidates.begin() + 1, candidates.end(),
                       [](const Candidate& a, const Candidate& b) {
                         return a.score > b.score;
                       });
      if (static_cast<int>(candidates.size()) > nbest_size) {
        candidates.resize(nbest_size);
      }
    }
  }

  for (const auto& cand : candidates) {
    auto* spt = nbest_spt->add_nbests();
    spt->set_score(cand.score);
    ABSL_RETURN_IF_ERROR(PopulateSentencePieceText(
        input, normalized, norm_to_orig, cand.ids, cand.pieces, spt));
  }

  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::SampleEncode(
    absl::string_view input, int /*nbest_size*/, float alpha,
    SentencePieceText* spt) const {
  RET_CHECK_STATUS_PROTO(spt);

  RET_CHECK(std::isfinite(alpha));
  RET_CHECK_GE(alpha, 0.0);

  RET_CHECK(model_type_ == TrainerSpec::UNIGRAM ||
            model_type_ == TrainerSpec::BPE)
      << "SampleEncode is not available for the current model.";

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  ABSL_RETURN_IF_ERROR(Normalize(input, &normalized, &norm_to_orig));

  auto* mt = random::GetRandomGenerator();
  auto uniform_sampler = [mt]() -> float {
    return absl::Uniform<float>(*mt, 0.0f, 1.0f);
  };

  std::vector<int> ids;
  std::vector<std::string_view> pieces;
  const auto code = lite_processor_->SampleNormalized(
      normalized, alpha, uniform_sampler, &ids, &pieces);
  ABSL_RETURN_IF_ERROR(ToAbslStatus(code, "SampleNormalized failed."));

  return PopulateSentencePieceText(input, normalized, norm_to_orig, ids, pieces,
                                   spt);
}

absl::Status SentencePieceProcessor::Decode(
    absl::Span<const absl::string_view> pieces, SentencePieceText* spt) const {
  return DecodePiecesInternal(pieces, /*detokenized=*/nullptr, spt);
}

// `DecodePiecesInternal` implements full SentencePiece detokenization rather
// than delegating to `SentencePieceLiteProcessor::DecodeIds`, because the full
// `SentencePieceProcessor` API requires:
// 1. Applying `decode_extra_options_` (e.g. `REVERSE`, `BOS`, `EOS`).
// 2. Reconstructing per-piece `surface` strings and `[begin, end)` byte
//    offsets in `SentencePieceText`.
// 3. Preserving arbitrary out-of-vocabulary string pieces passed to
//    `Decode(Span<string_view>)` as literal text (whereas `<unk>` maps to
//    `unk_surface_`), and grouping consecutive `<0xXX>` byte pieces into valid
//    UTF-8 characters (or `U+FFFD` replacement characters).
absl::Status SentencePieceProcessor::DecodePiecesInternal(
    absl::Span<const absl::string_view> pieces, std::string* detokenized,
    SentencePieceText* spt) const {
  if (spt != nullptr) {
    RET_CHECK_STATUS_PROTO(spt);
    detokenized = spt->mutable_text();
  } else {
    RET_CHECK_STATUS_STL(detokenized);
    if (pieces.empty()) {
      return absl::OkStatus();
    }
  }

  // Appends decoded piece to `text` and returns a boolean indicating if the
  // function has consumed a bos whitespace token (a piece starting with a
  // kSpaceSymbol). This is used to strip only the first whitespace token from
  // the decoded sequence for add_dummy_prefix.
  auto DecodeSentencePiece = [&](absl::string_view piece, int id,
                                 bool is_bos_ws, std::string* text) -> bool {
    if (IsControlId(*lite_processor_, id)) {  // <s>, </s>
      return false;                           // invisible symbol.
    } else if (id == -1 || IsUnknownId(*lite_processor_, id)) {
      if (id >= 0 && lite_processor_->IdToPiece(id) == piece) {  // <unk>
        text->append(unk_surface_.data(), unk_surface_.size());
      } else {  // return piece when piece is not <unk>.
        text->append(piece.data(), piece.size());
      }
      return false;
    }

    bool has_bos_ws = false;  // whether the token starts with a kSpaceSymbol
    if (is_bos_ws && (add_dummy_prefix_ || remove_extra_whitespaces_)) {
      // Consume if the current position is bos and
      // piece starts with kSpaceSymbol.
      has_bos_ws = absl::ConsumePrefix(&piece, kSpaceSymbol);

      if (remove_extra_whitespaces_) {
        // if we are removing extra whitespace, we remove all leading whitespace
        has_bos_ws = false;
      }
    }

    if (absl::ConsumePrefix(&piece, kSpaceSymbol)) {
      text->push_back(' ');
    }
    if (!absl::StrContains(piece, kSpaceSymbol)) {
      text->append(piece.data(), piece.size());
    } else {
      absl::StrAppend(text, absl::StrReplaceAll(piece, {{kSpaceSymbol, " "}}));
    }
    return has_bos_ws;
  };

  absl::Span<const absl::string_view> active_pieces = pieces;
  std::vector<absl::string_view> work_pieces;
  if (spt != nullptr) {
    spt->mutable_pieces()->Reserve(pieces.size());
    for (absl::string_view w : pieces) {
      auto* sp = spt->add_pieces();
      sp->mutable_piece()->assign(w.data(), w.size());
      sp->set_id(lite_processor_->PieceToId(w));
    }
    ABSL_RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, spt));
  } else if (!decode_extra_options_.empty()) {
    work_pieces.assign(pieces.begin(), pieces.end());
    ABSL_RETURN_IF_ERROR(
        ApplyExtraOptions(decode_extra_options_, &work_pieces));
    active_pieces = work_pieces;
  }

  const int num_tokens = spt != nullptr
                             ? spt->pieces_size()
                             : static_cast<int>(active_pieces.size());
  auto get_piece = [&](int index) -> absl::string_view {
    return spt != nullptr ? absl::string_view(spt->pieces(index).piece())
                          : active_pieces[index];
  };
  auto get_id = [&](int index) -> int {
    return spt != nullptr ? spt->pieces(index).id()
                          : lite_processor_->PieceToId(active_pieces[index]);
  };

  detokenized->reserve(detokenized->size() + num_tokens);
  auto SetSurface = [&](int index, absl::string_view surface) {
    if (spt != nullptr) {
      auto* sp = spt->mutable_pieces(index);
      sp->set_surface(surface.data(), surface.size());
      sp->set_begin(detokenized->size());
      sp->set_end(detokenized->size() + surface.size());
    }
    absl::StrAppend(detokenized, surface);
  };

  auto ProcessBytePieces = [&](int token_index_begin,
                               int token_index_end) -> absl::Status {
    if (token_index_begin >= token_index_end) {
      return absl::OkStatus();
    }

    // Constructs byte sequence.
    std::string bytes;
    bytes.reserve(token_index_end - token_index_begin);
    for (int i = token_index_begin; i < token_index_end; ++i) {
      const int byte = PieceToByte(get_piece(i));
      RET_CHECK_LE(0, byte);
      bytes.append(1, byte);
    }

    // Set surfaces of `bytes` for each Unicode character.
    int offset = 0;
    const int bytes_len = bytes.size();
    while (offset < bytes_len) {
      // Consume `bytes` by one Unicode character.
      size_t consumed;  // Number of bytes consumed in this iteration.
      const bool is_valid = string_util::IsValidDecodeUTF8(
          absl::string_view(bytes).substr(offset), &consumed);

      // Set surfaces of the consumed byte pieces.
      const int token_index = token_index_begin + offset;

      if (!is_valid) {
        // The byte piece at `token_index` is structurally invalid. Map it to
        // REPLACEMENT CHARACTER (U+FFFD).
        RET_CHECK_EQ(consumed, 1);
        SetSurface(token_index, kReplacementCharacter);
      } else {
        const absl::string_view utf8 =
            absl::string_view(bytes).substr(offset, consumed);
        for (size_t j = 0; j < consumed; j++) {
          // The last byte piece holds the surface of the original unknown
          // character. The other byte pieces hold an empty string as
          // surface.
          if (j == consumed - 1) {
            SetSurface(token_index + j, utf8);
          } else {
            SetSurface(token_index + j, "");
          }
        }
      }
      offset += consumed;
    }
    RET_CHECK_EQ(token_index_begin + offset, token_index_end);

    return absl::OkStatus();
  };

  int byte_start = 0;
  bool is_bos_ws = true;  // whether we expect a bos ws token to consume.
  bool bos_ws_seen = false;

  for (int i = 0; i < num_tokens; ++i) {
    const int id = get_id(i);
    if (!IsByteId(*lite_processor_, id)) {
      ABSL_RETURN_IF_ERROR(ProcessBytePieces(byte_start, i));

      // if we have seen a bos_ws token or any non-empty token
      if (bos_ws_seen || !detokenized->empty()) is_bos_ws = false;

      byte_start = i + 1;
      const size_t begin_pos = detokenized->size();
      bos_ws_seen =
          DecodeSentencePiece(get_piece(i), id, is_bos_ws, detokenized);
      if (spt != nullptr) {
        auto* mut_sp = spt->mutable_pieces(i);
        mut_sp->set_begin(begin_pos);
        mut_sp->set_end(detokenized->size());
        mut_sp->set_surface(detokenized->data() + begin_pos,
                            detokenized->size() - begin_pos);
      }
    }
  }
  ABSL_RETURN_IF_ERROR(ProcessBytePieces(byte_start, num_tokens));

  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::Decode(absl::Span<const int> ids,
                                            SentencePieceText* spt) const {
  RET_CHECK_STATUS_PROTO(spt);
  const int num_pieces = static_cast<int>(lite_processor_->vocab_size());
  for (const int id : ids) {
    if (id < 0 || id >= num_pieces) {
      return absl::Status(absl::StatusCode::kOutOfRange,
                          absl::StrCat("Invalid id: ", id));
    }
  }

  absl::Span<const int> active_ids = ids;
  std::vector<int> work_ids;
  if (!decode_extra_options_.empty()) {
    work_ids.assign(ids.begin(), ids.end());
    ABSL_RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, &work_ids));
    active_ids = work_ids;
  }

  std::vector<std::string_view> surfaces;
  const auto code = lite_processor_->Decode(
      lite::Span<const int>(active_ids.data(), active_ids.size()),
      spt->mutable_text(), &surfaces);
  ABSL_RETURN_IF_ERROR(ToAbslStatus(code, "Decode failed."));

  const char* text_base = spt->text().data();
  spt->mutable_pieces()->Reserve(active_ids.size());
  for (size_t i = 0; i < active_ids.size(); ++i) {
    auto* sp = spt->add_pieces();
    const int id = active_ids[i];
    const absl::string_view piece = lite_processor_->IdToPiece(id);
    const std::string_view surface = surfaces[i];
    const size_t begin = static_cast<size_t>(surface.data() - text_base);
    sp->set_piece(piece.data(), piece.size());
    sp->set_id(id);
    sp->set_surface(surface.data(), surface.size());
    sp->set_begin(begin);
    sp->set_end(begin + surface.size());
  }
  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::ParallelEncodeInternal(
    absl::string_view input, size_t chunk_len, ThreadPool& thread_pool,
    std::vector<std::string>* pieces, std::vector<int>* ids,
    SentencePieceText* spt) const {
  ABSL_RETURN_IF_ERROR(status());
  if (pieces != nullptr) pieces->clear();
  if (ids != nullptr) ids->clear();
  if (spt != nullptr) spt->Clear();

  chunk_len = std::max<size_t>(1, chunk_len);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  ABSL_RETURN_IF_ERROR(
      Normalize(input, &normalized, spt != nullptr ? &norm_to_orig : nullptr));

  std::vector<int> merged_ids;
  std::vector<std::string_view> merged_pieces;
  if (!normalized.empty()) {
    // Group pre-tokenized safe sub-chunks into chunks of UTF-8 byte length >=
    // chunk_len.
    std::vector<std::string_view> chunks;
    std::string_view current_chunk;
    auto append_sub_chunk = [&](std::string_view sub_chunk) {
      if (sub_chunk.empty()) return;
      if (current_chunk.empty()) {
        current_chunk = sub_chunk;
      } else {
        current_chunk = std::string_view(
            current_chunk.data(),
            static_cast<size_t>((sub_chunk.data() + sub_chunk.size()) -
                                current_chunk.data()));
      }
      if (current_chunk.size() >= chunk_len) {
        chunks.push_back(current_chunk);
        current_chunk = std::string_view();
      }
    };
    if (model_type_ == TrainerSpec::WORD) {
      for (const auto& w :
           SplitIntoWords(normalized, treat_whitespace_as_suffix_)) {
        append_sub_chunk(w);
      }
    } else {
      ABSL_RETURN_IF_ERROR(
          ToAbslStatus(lite_processor_->PretokenizeAtSafeBoundaries(
                           normalized, append_sub_chunk),
                       "PretokenizeAtSafeBoundaries failed."));
    }
    if (!current_chunk.empty()) {
      chunks.push_back(current_chunk);
    }

    const bool need_piece_views = (pieces != nullptr || spt != nullptr);
    std::vector<std::vector<int>> chunk_ids(chunks.size());
    std::vector<std::vector<std::string_view>> chunk_pieces(
        need_piece_views ? chunks.size() : 0);

    ABSL_RETURN_IF_ERROR(RunBatch(
        chunks.size(),
        [&](size_t i) -> absl::Status {
          return EncodeNormalizedInternal(
              chunks[i], &chunk_ids[i],
              need_piece_views ? &chunk_pieces[i] : nullptr);
        },
        thread_pool));

    // Merge encoded chunks, coalescing consecutive UNK tokens across chunk
    // boundaries when byte fallback is disabled.
    size_t total_tokens = 0;
    for (const auto& c_ids : chunk_ids) {
      total_tokens += c_ids.size();
    }
    merged_ids.reserve(total_tokens);
    if (need_piece_views) {
      merged_pieces.reserve(total_tokens);
    }

    for (size_t c = 0; c < chunks.size(); ++c) {
      const auto& c_ids = chunk_ids[c];
      for (size_t j = 0; j < c_ids.size(); ++j) {
        const int id = c_ids[j];
        const bool is_unk = IsUnknownId(*lite_processor_, id);
        if (!byte_fallback_ && is_unk && !merged_ids.empty() &&
            IsUnknownId(*lite_processor_, merged_ids.back())) {
          if (need_piece_views) {
            auto& back = merged_pieces.back();
            const std::string_view w = chunk_pieces[c][j];
            back = std::string_view(back.data(), back.size() + w.size());
          }
        } else {
          merged_ids.push_back(id);
          if (need_piece_views) {
            merged_pieces.push_back(chunk_pieces[c][j]);
          }
        }
      }
    }
  }

  if (spt != nullptr) {
    ABSL_RETURN_IF_ERROR(PopulateSentencePieceText(
        input, normalized, norm_to_orig, merged_ids, merged_pieces, spt));
  }

  if (pieces != nullptr) {
    PopulatePieceStrings(merged_ids, merged_pieces, pieces);
    ABSL_RETURN_IF_ERROR(ApplyExtraOptions(encode_extra_options_, pieces));
  }

  if (ids != nullptr) {
    *ids = std::move(merged_ids);
    ABSL_RETURN_IF_ERROR(ApplyExtraOptions(encode_extra_options_, ids));
  }

  return absl::OkStatus();
}

absl::Status SentencePieceProcessor::ParallelEncode(
    absl::string_view input, int chunk_len, ThreadPool& thread_pool,
    std::vector<std::string>* pieces) const {
  RET_CHECK_STATUS_STL(pieces);
  RET_CHECK_GT(chunk_len, 0);
  return ParallelEncodeInternal(input, chunk_len, thread_pool, pieces, nullptr,
                                nullptr);
}

absl::Status SentencePieceProcessor::ParallelEncode(
    absl::string_view input, int chunk_len, ThreadPool& thread_pool,
    std::vector<int>* ids) const {
  RET_CHECK_STATUS_STL(ids);
  RET_CHECK_GT(chunk_len, 0);
  return ParallelEncodeInternal(input, chunk_len, thread_pool, nullptr, ids,
                                nullptr);
}

absl::Status SentencePieceProcessor::ParallelEncode(
    absl::string_view input, int chunk_len, ThreadPool& thread_pool,
    SentencePieceText* spt) const {
  RET_CHECK_STATUS_PROTO(spt);
  RET_CHECK_GT(chunk_len, 0);
  return ParallelEncodeInternal(input, chunk_len, thread_pool, nullptr, nullptr,
                                spt);
}

#define RET_CHECK_OR_RETURN_DEFAULT(value)                                   \
  if (!status().ok()) {                                                      \
    LOG(ERROR) << status().message() << "\nReturns default value " << value; \
    return value;                                                            \
  }

absl::Status SentencePieceProcessor::Normalize(absl::string_view input,
                                               std::string* normalized) const {
  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK(normalized != nullptr);
  return ToAbslStatus(lite_processor_->Normalize(input, normalized, nullptr),
                      "Normalize failed.");
}

absl::Status SentencePieceProcessor::Normalize(
    absl::string_view input, std::string* normalized,
    std::vector<size_t>* norm_to_orig) const {
  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK(normalized != nullptr);
  return ToAbslStatus(
      lite_processor_->Normalize(input, normalized, norm_to_orig),
      "Normalize failed.");
}

std::string SentencePieceProcessor::Normalize(absl::string_view input) const {
  std::string normalized;
  Normalize(input, &normalized).IgnoreError();
  return normalized;
}

int SentencePieceProcessor::GetPieceSize() const {
  RET_CHECK_OR_RETURN_DEFAULT(0);
  return static_cast<int>(lite_processor_->vocab_size());
}

int SentencePieceProcessor::PieceToId(absl::string_view piece) const {
  RET_CHECK_OR_RETURN_DEFAULT(-1);
  return lite_processor_->PieceToId(piece);
}

absl::string_view SentencePieceProcessor::IdToPiece(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT("<unk>");
  return lite_processor_->IdToPiece(id);
}

bool SentencePieceProcessor::SafeIdToPiece(int id, std::string* piece) const {
  RET_CHECK_OR_RETURN_DEFAULT(false);
  if (piece == nullptr || id < 0 ||
      id >= static_cast<int>(lite_processor_->vocab_size())) {
    return false;
  }
  const absl::string_view view = lite_processor_->IdToPiece(id);
  piece->assign(view.data(), view.size());
  return true;
}

float SentencePieceProcessor::GetScore(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT(0.0f);
  return lite_processor_->GetScore(id);
}

bool SentencePieceProcessor::IsControl(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT(false);
  return lite_processor_->piece_type(id) == ModelProto::SentencePiece::CONTROL;
}

bool SentencePieceProcessor::IsUnknown(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT(false);
  return id == lite_processor_->unk_id() ||
         lite_processor_->piece_type(id) == ModelProto::SentencePiece::UNKNOWN;
}

bool SentencePieceProcessor::IsUnused(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT(false);
  return lite_processor_->piece_type(id) == ModelProto::SentencePiece::UNUSED;
}

bool SentencePieceProcessor::IsByte(int id) const {
  RET_CHECK_OR_RETURN_DEFAULT(false);
  return lite_processor_->piece_type(id) == ModelProto::SentencePiece::BYTE;
}

int SentencePieceProcessor::unk_id() const {
  return lite_processor_ ? lite_processor_->unk_id() : -1;
}

int SentencePieceProcessor::bos_id() const {
  return lite_processor_ ? lite_processor_->bos_id() : -1;
}

int SentencePieceProcessor::eos_id() const {
  return lite_processor_ ? lite_processor_->eos_id() : -1;
}

int SentencePieceProcessor::pad_id() const {
  return lite_processor_ ? lite_processor_->pad_id() : -1;
}

template <typename T>
absl::Status SentencePieceProcessor::ApplyExtraOptions(
    absl::Span<const ExtraOption> extra_options, T* output) const {
  for (const auto& extra_option : extra_options) {
    switch (extra_option) {
      case REVERSE:
        if constexpr (std::is_same_v<T, SentencePieceText>) {
          std::reverse(output->mutable_pieces()->begin(),
                       output->mutable_pieces()->end());
        } else {
          std::reverse(output->begin(), output->end());
        }
        break;
      case EOS:
        if (const int id = eos_id(); id != -1) {
          if constexpr (std::is_same_v<T, SentencePieceText>) {
            auto* piece = output->add_pieces();
            piece->set_id(id);
            piece->set_piece(eos_piece_.data(), eos_piece_.size());
            piece->set_begin(output->text().size());
            piece->set_end(output->text().size());
          } else {
            using V = typename T::value_type;
            if constexpr (std::is_same_v<V, int>) {
              output->emplace_back(id);
            } else {
              output->emplace_back(eos_piece_);
            }
          }
        }
        break;
      case BOS:
        if (const int id = bos_id(); id != -1) {
          if constexpr (std::is_same_v<T, SentencePieceText>) {
            auto* array = output->mutable_pieces();
            array->Add();
            for (int i = array->size() - 1; i > 0; --i) {
              array->SwapElements(i - 1, i);
            }
            auto* piece = array->Mutable(0);
            piece->set_id(id);
            piece->set_piece(bos_piece_.data(), bos_piece_.size());
            piece->set_begin(0);
            piece->set_end(0);
          } else {
            using V = typename T::value_type;
            if constexpr (std::is_same_v<V, int>) {
              output->emplace(output->begin(), id);
            } else {
              output->emplace(output->begin(), bos_piece_);
            }
          }
        }
        break;
      case UNK_PIECE:
        if constexpr (std::is_same_v<T, SentencePieceText>) {
          for (int i = 0; i < output->pieces_size(); ++i) {
            auto* piece = output->mutable_pieces(i);
            if (IsUnknownId(*lite_processor_, piece->id())) {
              piece->set_piece(unk_piece_.data(), unk_piece_.size());
            }
          }
        }
        break;
      default:
        if constexpr (std::is_same_v<T, SentencePieceText>) {
          output->Clear();
        } else {
          output->clear();
        }
        return absl::InternalError("unknown extra_option type.");
    }
  }
  return absl::OkStatus();
}

bool SentencePieceProcessor::HasUnkPieceOption() const {
  for (const auto& option : encode_extra_options_) {
    if (option == UNK_PIECE) return true;
  }
  return false;
}

absl::Status SentencePieceProcessor::ParseExtraOptions(
    absl::string_view extra_option,
    std::vector<SentencePieceProcessor::ExtraOption>* extra_options) const {
  extra_options->clear();
  if (extra_option.empty()) return absl::OkStatus();

  ABSL_RETURN_IF_ERROR(status());

  for (const auto& s : absl::StrSplit(extra_option, ':')) {
    ExtraOption opt;
    if (s == "bos") {
      RET_CHECK(bos_id() != -1)
          << "id for `" << bos_piece_ << "` is not defined.";
      opt = BOS;
    } else if (s == "eos") {
      RET_CHECK(eos_id() != -1)
          << "id for `" << eos_piece_ << "` is not defined.";
      opt = EOS;
    } else if (s == "reverse") {
      opt = REVERSE;
    } else if (s == "unk" || s == "unk_piece") {
      opt = UNK_PIECE;
    } else {
      return absl::InternalError(
          absl::StrCat("option \"", s, "\" is not available."));
    }
    extra_options->push_back(opt);
  }
  return absl::OkStatus();
}

const ModelProto& SentencePieceProcessor::model_proto() const {
  return model_proto_manager_ ? model_proto_manager_->model_proto()
                              : ModelProto::default_instance();
}

std::string SentencePieceProcessor::serialized_model_proto() const {
  return model_proto_manager_ ? model_proto_manager_->serialized_model_proto()
                              : "";
}

absl::string_view SentencePieceProcessor::serialized_flatbuffer() const {
  return model_proto_manager_ ? model_proto_manager_->serialized_flatbuffer()
                              : absl::string_view();
}

namespace io {
absl::Status LoadModelProto(absl::string_view filename,
                            ModelProto* model_proto) {
  if (filename.empty()) {
    return absl::NotFoundError("model file path should not be empty.");
  }

  auto input = filesystem::NewReadableFile(filename, true);
  ABSL_RETURN_IF_ERROR(input->status());
  std::string serialized;
  if (!input->ReadAll(&serialized)) {
    return absl::InternalError(absl::StrCat("could not read ", filename));
  }
  if (!model_proto->ParseFromArray(serialized.data(), serialized.size())) {
    return absl::InternalError(
        absl::StrCat("could not parse ModelProto from ", filename));
  }

  return absl::OkStatus();
}

absl::Status SaveModelProto(absl::string_view filename,
                            const ModelProto& model_proto) {
  if (filename.empty()) {
    return absl::NotFoundError("model file path should not be empty.");
  }
  auto output = filesystem::NewWritableFile(filename, true);
  ABSL_RETURN_IF_ERROR(output->status());
  RET_CHECK(output->Write(model_proto.SerializeAsString()));

  return absl::OkStatus();
}
}  // namespace io
}  // namespace sentencepiece
