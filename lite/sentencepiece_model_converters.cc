// Copyright 2026 Google LLC
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

#include "sentencepiece_model_converters.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "darts.h"
#include "flatbuffers/flatbuffers.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_lite_generated.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::lite {
namespace {

bool IsInvisiblePiece(const ::sentencepiece::ModelProto::SentencePiece& sp) {
  return sp.type() == ::sentencepiece::ModelProto::SentencePiece::UNKNOWN ||
         sp.type() == ::sentencepiece::ModelProto::SentencePiece::CONTROL ||
         sp.type() == ::sentencepiece::ModelProto::SentencePiece::UNUSED ||
         sp.type() == ::sentencepiece::ModelProto::SentencePiece::BYTE;
}

absl::StatusOr<std::vector<uint8_t>> BuildDoubleArrayBlob(
    std::vector<std::pair<std::string_view, int>> elements) {
  if (elements.empty()) {
    return std::vector<uint8_t>();
  }
  std::sort(elements.begin(), elements.end());
  for (size_t i = 0; i < elements.size(); ++i) {
    if (elements[i].first.find('\0') != std::string_view::npos) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Key contains invalid embedded null byte: ", elements[i].first));
    }
    if (i > 0 && elements[i].first == elements[i - 1].first) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate key for trie building: ", elements[i].first));
    }
  }

  std::vector<const char*> keys(elements.size());
  std::vector<size_t> lengths(elements.size());
  std::vector<int> values(elements.size());
  for (size_t i = 0; i < elements.size(); ++i) {
    keys[i] = elements[i].first.data();
    lengths[i] = elements[i].first.size();
    values[i] = elements[i].second;
  }

  Darts::DoubleArray trie;
  if (trie.build(keys.size(), const_cast<char**>(keys.data()), lengths.data(),
                 values.data()) != 0) {
    return absl::InternalError("Failed to build double-array trie");
  }

  const uint8_t* trie_ptr = reinterpret_cast<const uint8_t*>(trie.array());
  size_t trie_size_bytes = trie.total_size();
  return std::vector<uint8_t>(trie_ptr, trie_ptr + trie_size_bytes);
}

inline size_t OneCharLen(std::string_view s) {
  if (s.empty()) return 0;
  size_t len =
      "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(static_cast<unsigned char>(s[0])) >>
                                         4];
  return (len <= s.size()) ? len : s.size();
}

// Builds the bigram Double-Array Trie and compiles suffix links into its
// leaves.
//
// DESIGN NOTE: Why we use a build-and-patch approach instead of a 2-pass
// rebuild: Darts-clone minimizes the trie into a DAWG when values are passed.
// Because DAWG minimization merges states based on leaf values, changing values
// between passes (e.g. from dummy 0s to real suffix links) changes the DAWG
// structure and shifts node offsets. This makes it impossible to rebuild the
// trie with suffix link values while keeping the node positions calculated in
// the first pass.
//
// To resolve this, we build the trie ONCE without values (keyset mode), which
// generates a standard trie with a deterministic, invariant layout. We then
// traverse this trie to locate the leaf node of each bigram (prev, curr) and
// manually patch its value in memory with the node position of its second
// character (curr).
absl::StatusOr<std::vector<uint8_t>> BuildBigramTrieBlob(
    std::vector<std::string_view> sorted_bigrams) {
  if (sorted_bigrams.empty()) {
    return std::vector<uint8_t>();
  }

  std::vector<const char*> bigram_keys(sorted_bigrams.size());
  std::vector<size_t> bigram_lengths(sorted_bigrams.size());
  for (size_t i = 0; i < sorted_bigrams.size(); ++i) {
    if (sorted_bigrams[i].find('\0') != std::string_view::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("Bigram key contains invalid embedded null byte: ",
                       sorted_bigrams[i]));
    }
    bigram_keys[i] = sorted_bigrams[i].data();
    bigram_lengths[i] = sorted_bigrams[i].size();
  }

  Darts::DoubleArray bigram_trie;
  if (bigram_trie.build(bigram_keys.size(),
                        const_cast<char**>(bigram_keys.data()),
                        bigram_lengths.data(), nullptr) != 0) {
    return absl::InternalError("Failed to build base bigram trie");
  }

  const size_t array_size_units =
      bigram_trie.total_size() / sizeof(Darts::Details::DoubleArrayUnit);
  if (array_size_units == 0) {
    return absl::InternalError("Bigram trie array size is zero.");
  }

  auto get_node_pos = [&bigram_trie,
                       array_size_units](std::string_view key) -> int64_t {
    const auto* array =
        reinterpret_cast<const Darts::Details::DoubleArrayUnit*>(
            bigram_trie.array());
    size_t node_pos = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      if (node_pos >= array_size_units) return -1;
      const auto& unit = array[node_pos];
      size_t next_pos =
          node_pos ^ unit.offset() ^ static_cast<unsigned char>(key[i]);
      if (next_pos >= array_size_units) return -1;
      const auto& next_unit = array[next_pos];
      if (next_unit.label() != static_cast<unsigned char>(key[i])) {
        return -1;
      }
      node_pos = next_pos;
    }
    return static_cast<int64_t>(node_pos);
  };

  auto get_leaf_pos = [&bigram_trie,
                       array_size_units](std::string_view key) -> int64_t {
    const auto* array =
        reinterpret_cast<const Darts::Details::DoubleArrayUnit*>(
            bigram_trie.array());
    size_t node_pos = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      if (node_pos >= array_size_units) return -1;
      const auto& unit = array[node_pos];
      node_pos ^= unit.offset() ^ static_cast<unsigned char>(key[i]);
    }
    if (node_pos >= array_size_units) return -1;
    const auto& unit = array[node_pos];
    size_t leaf_pos = node_pos ^ unit.offset();
    if (leaf_pos >= array_size_units) return -1;
    return static_cast<int64_t>(leaf_pos);
  };

  uint32_t* mutable_array = const_cast<uint32_t*>(
      reinterpret_cast<const uint32_t*>(bigram_trie.array()));

  for (size_t i = 0; i < sorted_bigrams.size(); ++i) {
    const std::string_view bigram = sorted_bigrams[i];
    const size_t prev_len = OneCharLen(bigram);
    std::string_view curr = bigram.substr(prev_len);

    int64_t curr_pos = get_node_pos(curr);
    if (curr_pos == -1) {
      curr_pos = 0;
    }

    int64_t leaf_pos = get_leaf_pos(bigram);
    if (leaf_pos < 0 || leaf_pos >= static_cast<int64_t>(array_size_units)) {
      return absl::InternalError(
          "Out-of-bounds leaf position in bigram trie patching.");
    }
    mutable_array[leaf_pos] = static_cast<uint32_t>(curr_pos) | (1U << 31);
  }

  const uint8_t* byte_ptr =
      reinterpret_cast<const uint8_t*>(bigram_trie.array());
  return std::vector<uint8_t>(byte_ptr, byte_ptr + bigram_trie.total_size());
}

}  // namespace

absl::StatusOr<std::string> ToFlatbuffer(
    const ::sentencepiece::ModelProto& proto, const ConverterOptions& options) {
  if (proto.pieces_size() <= 0 || proto.pieces_size() > 16000000) {
    return absl::InvalidArgumentError(
        absl::StrCat("Model vocabulary size (", proto.pieces_size(),
                     ") is out of safe bounds [1, 16,000,000]."));
  }

  ::sentencepiece::lite::ModelProtoT fbs;

  // 1. Extract and validate trainer_spec options first
  bool is_bpe = false;
  bool byte_fallback_enabled = false;
  std::string_view bos_piece = "<s>";
  std::string_view eos_piece = "</s>";
  std::string_view pad_piece = "<pad>";
  if (proto.has_trainer_spec()) {
    const auto& ts = proto.trainer_spec();
    auto model_type = ts.model_type();
    if (model_type != ::sentencepiece::TrainerSpec::UNIGRAM &&
        model_type != ::sentencepiece::TrainerSpec::BPE) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported model type: ",
          ::sentencepiece::TrainerSpec::ModelType_Name(model_type)));
    }
    is_bpe = (model_type == ::sentencepiece::TrainerSpec::BPE);
    byte_fallback_enabled = ts.byte_fallback();
    fbs.unk_id = ts.unk_id();
    fbs.unk_surface = ts.unk_surface();
    fbs.model_type =
        static_cast<::sentencepiece::lite::ModelType>(ts.model_type());
    if (!ts.bos_piece().empty()) {
      bos_piece = ts.bos_piece();
    }
    if (!ts.eos_piece().empty()) {
      eos_piece = ts.eos_piece();
    }
    if (!ts.pad_piece().empty()) {
      pad_piece = ts.pad_piece();
    }
  } else {
    // If trainer_spec is missing, default model type in SentencePiece is
    // UNIGRAM.
    fbs.model_type = ::sentencepiece::lite::ModelType_UNIGRAM;
  }

  fbs.bos_id = -1;
  fbs.eos_id = -1;
  fbs.pad_id = -1;

  // 2. Convert pieces, scores, types with validation and prepare Trie building
  fbs.pieces.reserve(proto.pieces_size());
  if (is_bpe) {
    fbs.int_scores.reserve(proto.pieces_size());
  } else {
    fbs.scores.reserve(proto.pieces_size());
  }
  fbs.types.reserve(proto.pieces_size());

  std::vector<std::string> sanitized_pieces;
  sanitized_pieces.reserve(proto.pieces_size());
  std::vector<std::pair<std::string_view, int>> pieces_to_sort;
  pieces_to_sort.reserve(proto.pieces_size());
  std::vector<std::pair<std::string_view, int>> user_defined_pieces;
  constexpr size_t kMaxPieceLength = 4096;
  for (int i = 0; i < proto.pieces_size(); ++i) {
    const auto& sp = proto.pieces(i);
    std::string_view piece_view = sp.piece();
    auto piece_type = sp.type();
    // Security / Crash Prevention Guard:
    // Darts-clone treats byte 0 ('\0') as an end-of-key / leaf terminator. If a
    // piece string contains an embedded null byte, DAWGs minimization in Darts
    // fails and aborts the entire process (SIGABRT). We strictly reject
    // embedded null bytes, empty pieces, and oversized pieces here to prevent
    // adversarial crash or DoS attacks during trie building.
    if (piece_view.empty() || piece_view.size() > kMaxPieceLength) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Piece at index ", i, " is invalid (empty or too long)."));
    }
    if (piece_view.find('\0') != std::string_view::npos) {  // NOLINT
      if (options.treat_null_byte_as_unused && byte_fallback_enabled) {
        sanitized_pieces.push_back(absl::StrCat("<unused_", i, ">"));
        piece_view = sanitized_pieces.back();
        piece_type = ::sentencepiece::ModelProto::SentencePiece::UNUSED;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Piece at index ", i, " is invalid (contains null byte)."));
      }
    }
    fbs.pieces.push_back(std::string(piece_view));
    pieces_to_sort.emplace_back(piece_view, i);
    if (piece_type ==
        ::sentencepiece::ModelProto::SentencePiece::USER_DEFINED) {
      user_defined_pieces.emplace_back(piece_view, i);
    }
    if (piece_type == ::sentencepiece::ModelProto::SentencePiece::CONTROL) {
      if (piece_view == bos_piece) {
        fbs.bos_id = i;
      }
      if (piece_view == eos_piece) {
        fbs.eos_id = i;
      }
      if (piece_view == pad_piece) {
        fbs.pad_id = i;
      }
    }
    if (is_bpe) {
      fbs.int_scores.push_back(static_cast<int32_t>(sp.score()));
    } else {
      fbs.scores.push_back(sp.score());
    }
    // Convert Piece type
    fbs.types.push_back(
        static_cast<::sentencepiece::lite::PieceType>(piece_type));
  }

  // 3. Convert normalizer_spec options
  if (proto.has_normalizer_spec()) {
    const auto& ns = proto.normalizer_spec();
    fbs.normalizer_spec =
        std::make_unique<::sentencepiece::lite::NormalizerSpecT>();
    fbs.normalizer_spec->name = ns.name();
    std::vector<uint8_t> charsmap(ns.precompiled_charsmap().begin(),
                                  ns.precompiled_charsmap().end());
    // SentencePiece stores precompiled_charsmap in Little-Endian format.
    // On Big-Endian platforms, convert trie_blob_size and all DoubleArray units
    // to native Big-Endian so that the runtime can access the trie directly
    // with zero copy.
    if constexpr (std::endian::native == std::endian::big) {
      if (charsmap.size() >= sizeof(uint32_t)) {
        uint32_t* words = reinterpret_cast<uint32_t*>(charsmap.data());
        const uint32_t trie_blob_size = absl::byteswap(words[0]);
        words[0] = trie_blob_size;
        if (sizeof(uint32_t) + trie_blob_size <= charsmap.size() &&
            (trie_blob_size % sizeof(uint32_t)) == 0) {
          const size_t num_words =
              (sizeof(uint32_t) + trie_blob_size) / sizeof(uint32_t);
          for (size_t i = 1; i < num_words; ++i) {
            words[i] = absl::byteswap(words[i]);
          }
        }
      }
    }
    fbs.normalizer_spec->precompiled_charsmap = std::move(charsmap);
    fbs.normalizer_spec->add_dummy_prefix = ns.add_dummy_prefix();
    fbs.normalizer_spec->remove_extra_whitespaces =
        ns.remove_extra_whitespaces();
    fbs.normalizer_spec->escape_whitespaces = ns.escape_whitespaces();
    if (proto.has_trainer_spec()) {
      fbs.normalizer_spec->treat_whitespace_as_suffix =
          proto.trainer_spec().treat_whitespace_as_suffix();
    }
  }

  // 4. Build Trie offline
  auto pieces_trie_blob_or = BuildDoubleArrayBlob(std::move(pieces_to_sort));
  if (!pieces_trie_blob_or.ok()) {
    return pieces_trie_blob_or.status();
  }
  fbs.pieces_trie_blob = std::move(*pieces_trie_blob_or);

  // 5. Build Prefix Matcher Trie offline for USER_DEFINED pieces if present
  if (!user_defined_pieces.empty()) {
    auto prefix_matcher_trie_blob_or =
        BuildDoubleArrayBlob(std::move(user_defined_pieces));
    if (!prefix_matcher_trie_blob_or.ok()) {
      return prefix_matcher_trie_blob_or.status();
    }
    fbs.prefix_matcher_trie_blob = std::move(*prefix_matcher_trie_blob_or);
  }

  // 6. Extract and Build strict character bigrams offline.
  // When in BPE mode, skip_char_bigrams is ignored and bigrams are always
  // built.
  if (is_bpe || !options.skip_char_bigrams) {
    std::vector<std::string_view> bigrams_list;
    for (int i = 0; i < proto.pieces_size(); ++i) {
      std::string_view piece = proto.pieces(i).piece();
      size_t prev_len = 0;
      size_t curr_offset = 0;
      while (curr_offset < piece.size()) {
        const int mblen = OneCharLen(piece.substr(curr_offset));
        if (mblen == 0) break;
        if (prev_len > 0) {
          bigrams_list.emplace_back(
              piece.substr(curr_offset - prev_len, prev_len + mblen));
        }
        prev_len = mblen;
        curr_offset += mblen;
      }
    }
    std::sort(bigrams_list.begin(), bigrams_list.end());
    bigrams_list.erase(std::unique(bigrams_list.begin(), bigrams_list.end()),
                       bigrams_list.end());
    auto char_bigram_trie_blob_or =
        BuildBigramTrieBlob(std::move(bigrams_list));
    if (!char_bigram_trie_blob_or.ok()) {
      return char_bigram_trie_blob_or.status();
    }
    fbs.char_bigram_trie_blob = std::move(*char_bigram_trie_blob_or);
  }

  // 7. Precompute direct mappings (shortcuts) in-memory using double-pass
  // serialization (Only needed for BPE models)
  if (is_bpe) {
    flatbuffers::FlatBufferBuilder builder;
    auto offset = ::sentencepiece::lite::ModelProto::Pack(builder, &fbs);
    builder.Finish(offset);
    std::string_view preliminary_bytes(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());

    SentencePieceLiteProcessor processor(preliminary_bytes);
    if (processor.status() != StatusCode::kOk) {
      return absl::InternalError(
          "Failed to initialize preliminary processor for BPE direct mapping.");
    }

    std::vector<uint8_t> direct_mappings(proto.pieces_size(), 0);
    bool all_are_direct_mappings = true;
    bool has_any_direct_mapping = false;
    // Performance optimization:
    // Reusing the vector buffer outside the loop across all vocabulary
    // iterations (e.g., 250,000+ times) avoids hundreds of thousands of heap
    // allocations and vector constructions/destructions during shortcut
    // precomputation.
    std::vector<int> ids;
    ids.reserve(16);
    for (int i = 0; i < proto.pieces_size(); ++i) {
      const auto& sp = proto.pieces(i);
      // Invisible pieces (control, unk, unused, byte) are filtered out at
      // runtime by IsInvisible() anyway. We mark them as 1 so that BPE models
      // containing only safe normal tokens can serialize their vector as null.
      if (IsInvisiblePiece(sp)) {
        direct_mappings[i] = 1;
        continue;
      }
      // We use EncodeNormalized() here because the vocabulary pieces stored in
      // the model are already normalized strings. Standard Encode() would run
      // the normalizer again, which is redundant and can distort special
      // tokens. This also matches runtime EncodeChunk behavior on normalized
      // chunks.
      if (processor.EncodeNormalized(sp.piece(), &ids) == StatusCode::kOk &&
          ids.size() == 1 && ids[0] == i) {
        direct_mappings[i] = 1;
        has_any_direct_mapping = true;
      } else {
        all_are_direct_mappings = false;
      }
    }

    if (has_any_direct_mapping) {
      fbs.has_direct_mappings = true;
      if (!all_are_direct_mappings) {
        fbs.is_direct_mapping = std::move(direct_mappings);
      }
    }
  }

  flatbuffers::FlatBufferBuilder builder;
  auto offset = ::sentencepiece::lite::ModelProto::Pack(builder, &fbs);
  builder.Finish(offset);
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

}  // namespace sentencepiece::lite
