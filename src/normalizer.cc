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
#include <utility>
#include <vector>

#include "common.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/strings/strip.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {

constexpr int Normalizer::kMaxTrieResultsSize;

Normalizer::Normalizer(const NormalizerSpec& spec,
                       const TrainerSpec& trainer_spec)
    : spec_(&spec),
      treat_whitespace_as_suffix_(trainer_spec.treat_whitespace_as_suffix()),
      status_(absl::OkStatus()) {
  Init();
}

Normalizer::Normalizer(const NormalizerSpec& spec)
    : spec_(&spec), status_(absl::OkStatus()) {
  Init();
}

Normalizer::~Normalizer() {}

void Normalizer::Init() {
  absl::string_view index = spec_->precompiled_charsmap();

  if (!index.empty()) {
    absl::string_view trie_blob;
    status_ = DecodePrecompiledCharsMap(index, &trie_blob, &normalized_,
                                        &precompiled_charsmap_buffer_);

    if (!status_.ok()) return;

    // Reads the body of double array.
    trie_ = std::make_unique<Darts::DoubleArray>();

    // copy_array ensures correct alignment even if the trie_blob is
    // mis-aligned. Although standard proto parsing copies data to the heap
    // (guaranteeing alignment), zero-copy parsing or custom buffers might
    // pass misaligned data.
    if (reinterpret_cast<uintptr_t>(trie_blob.data()) % 4 == 0) {
      trie_->set_array(const_cast<char*>(trie_blob.data()),
                       trie_blob.size() / trie_->unit_size());
    } else {
      trie_->copy_array(reinterpret_cast<const char*>(trie_blob.data()),
                        trie_blob.size());
    }

    if (!trie_->validate(normalized_.size())) {
      status_ = absl::InternalError("precompiled_charsmap is invalid.");
      trie_.reset();
      return;
    }
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

  RETURN_IF_ERROR(status());

  size_t consumed = 0;

  // Ignores heading space.
  if (spec_->remove_extra_whitespaces()) {
    while (!input.empty()) {
      const auto p = NormalizePrefix(input);
      if (p.first != " ") {
        break;
      }
      input.remove_prefix(p.second);
      consumed += p.second;
    }
  }

  // all chars are whitespace.
  if (input.empty()) {
    return absl::OkStatus();
  }

  // Reserves the output buffer to avoid re-allocations.
  const size_t kReservedSize = input.size() * 1.5;
  normalized->reserve(kReservedSize);
  if (norm_to_orig) norm_to_orig->reserve(kReservedSize);

  // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
  // if escape_whitespaces() is set (default = true).
  const absl::string_view kSpaceSymbol =
      spec_->escape_whitespaces() ? "\xe2\x96\x81" : " ";

  // adds kSpaceSymbol to the current context.
  auto add_ws = [&consumed, &normalized, &norm_to_orig, &kSpaceSymbol]() {
    normalized->append(kSpaceSymbol.data(), kSpaceSymbol.size());
    if (norm_to_orig) {
      norm_to_orig->insert(norm_to_orig->end(), kSpaceSymbol.size(), consumed);
    }
  };

  // Adds a space symbol as a prefix (default is true)
  // With this prefix, "world" and "hello world" are converted into
  // "_world" and "_hello_world", which help the trainer to extract
  // "_world" as one symbol.
  if (!treat_whitespace_as_suffix_ && spec_->add_dummy_prefix()) add_ws();

  bool is_prev_space = spec_->remove_extra_whitespaces();
  while (!input.empty()) {
    auto p = NormalizePrefix(input);
    absl::string_view sp = p.first;

    // Removes heading spaces in sentence piece,
    // if the previous sentence piece ends with whitespace.
    while (is_prev_space && absl::ConsumePrefix(&sp, " ")) {
    }

    if (!sp.empty()) {
      const char* data = sp.data();
      for (size_t n = 0; n < sp.size(); ++n) {
        if (data[n] == ' ') {
          add_ws();
        } else {
          *normalized += data[n];
          if (norm_to_orig) norm_to_orig->push_back(consumed);
        }
      }
      // Checks whether the last character of sp is whitespace.
      is_prev_space = absl::EndsWith(sp, " ");
    }

    consumed += p.second;
    input.remove_prefix(p.second);
    if (!spec_->remove_extra_whitespaces()) {
      is_prev_space = false;
    }
  }

  // Ignores trailing space.
  if (spec_->remove_extra_whitespaces()) {
    while (absl::EndsWith(*normalized, kSpaceSymbol)) {
      const int length = normalized->size() - kSpaceSymbol.size();
      RET_CHECK_GE(length, 0);
      normalized->resize(length);
      if (norm_to_orig) {
        consumed = (*norm_to_orig)[length];
        norm_to_orig->resize(length);
      }
    }
  }

  // Adds a space symbol as a suffix (default is false)
  if (treat_whitespace_as_suffix_ && spec_->add_dummy_prefix()) add_ws();

  if (norm_to_orig) {
    norm_to_orig->push_back(consumed);
    RET_CHECK_EQ(norm_to_orig->size(), normalized->size() + 1);
  }

  return absl::OkStatus();
}

std::string Normalizer::Normalize(absl::string_view input) const {
  std::string normalized;
  Normalize(input, &normalized, nullptr).IgnoreError();
  return normalized;
}

std::pair<absl::string_view, int> Normalizer::NormalizePrefix(
    absl::string_view input) const {
  if (input.empty()) return {};

  if (matcher_ != nullptr) {
    bool found = false;
    const int mblen = matcher_->PrefixMatch(input, &found);
    if (found) return {input.substr(0, mblen), mblen};
  }

  size_t longest_length = 0;
  size_t longest_value = 0;

  if (trie_ != nullptr) {
    // Allocates trie_results in stack, which makes the encoding speed 36%
    // faster. (38k sentences/sec => 60k sentences/sec). Builder checks that the
    // result size never exceeds kMaxTrieResultsSize. This array consumes
    // 0.5kByte in stack, which is less than default stack frames (16kByte).
    Darts::DoubleArray::result_pair_type
        trie_results[Normalizer::kMaxTrieResultsSize];

    // commonPrefixSearch returns the total number of matches, which can be
    // larger than the buffer size (see darts.h). A precompiled charsmap loaded
    // from an untrusted model is not subject to the build-time
    // kMaxTrieResultsSize check, so cap the count to the buffer capacity before
    // indexing trie_results.
    const size_t num_nodes =
        std::min<size_t>(trie_->commonPrefixSearch(
                             input.data(), trie_results,
                             Normalizer::kMaxTrieResultsSize, input.size()),
                         Normalizer::kMaxTrieResultsSize);

    // Finds the longest rule.
    for (size_t k = 0; k < num_nodes; ++k) {
      if (longest_length == 0 || trie_results[k].length > longest_length) {
        longest_length = trie_results[k].length;  // length of prefix
        longest_value = trie_results[k].value;    // pointer to |normalized_|.
      }
    }
  }

  std::pair<absl::string_view, int> result;
  if (longest_length == 0 || longest_length > input.size() ||
      longest_value >= normalized_.size()) {
    size_t length = 0;
    if (!string_util::IsValidDecodeUTF8(input, &length)) {
      // Found a malformed utf8.
      // The rune is set to be 0xFFFD (REPLACEMENT CHARACTER),
      // which is a valid Unicode of three bytes in utf8,
      // but here we only consume one byte.
      result.second = 1;
      static constexpr absl::string_view kReplacementChar = "\xEF\xBF\xBD";
      result.first = kReplacementChar;
    } else {
      result.second = length;
      result.first = absl::string_view(input.data(), result.second);
    }
  } else {
    result.second = longest_length;
    // No need to pass the size of normalized sentence,
    // since |normalized| is delimitered by "\0".
    result.first = normalized_.data() + longest_value;
  }

  return result;
}

// static
std::string Normalizer::EncodePrecompiledCharsMap(
    absl::string_view trie_blob, absl::string_view normalized) {
  // <trie size(4byte)><double array trie><normalized string>
  std::string blob;
  blob.append(string_util::EncodePOD<uint32_t>(trie_blob.size()));
  blob.append(trie_blob.data(), trie_blob.size());

  if constexpr (util::is_bigendian()) {
    uint32_t* data = reinterpret_cast<uint32_t*>(blob.data());
    for (int i = 0; i < blob.size() / 4; ++i) data[i] = util::Swap32(data[i]);
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

  if constexpr (util::is_bigendian()) {
    trie_blob_size = util::Swap32(trie_blob_size);
  }

  if (trie_blob_size >= blob.size() - sizeof(trie_blob_size)) {
    return absl::InternalError("Trie data size exceeds the input blob size.");
  }

  // Dart unit_size is 4 and blob size in units must be a multiple of 256.
  if (trie_blob_size < 1024 || (trie_blob_size & 0x3FF) != 0) {
    return absl::InternalError("Trie data size is not divisible by 1024.");
  }

  blob.remove_prefix(sizeof(trie_blob_size));

  if constexpr (util::is_bigendian()) {
    RET_CHECK(buffer);
    buffer->assign(blob.data(), trie_blob_size);
    uint32_t* data =
        reinterpret_cast<uint32_t*>(const_cast<char*>(buffer->data()));
    for (int i = 0; i < buffer->size() / 4; ++i)
      data[i] = util::Swap32(data[i]);
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

PrefixMatcher::PrefixMatcher(const std::set<absl::string_view>& dic) {
  if (dic.empty()) return;
  std::vector<const char*> key;
  std::vector<size_t> lengths;
  key.reserve(dic.size());
  lengths.reserve(dic.size());
  for (const auto& it : dic) {
    key.push_back(it.data());
    lengths.push_back(it.size());
  }
  trie_ = std::make_unique<Darts::DoubleArray>();
  if (trie_->build(key.size(), const_cast<char**>(key.data()),
                   const_cast<size_t*>(lengths.data()), nullptr) != 0) {
    LOG(ERROR) << "Failed to build the TRIE for PrefixMatcher";
    trie_.reset();
  }
}

int PrefixMatcher::PrefixMatch(absl::string_view w, bool* found) const {
  if (trie_ == nullptr) {
    if (found) *found = false;
    return std::min<int>(w.size(), string_util::OneCharLen(w.data()));
  }

  constexpr int kResultSize = 64;
  Darts::DoubleArray::result_pair_type trie_results[kResultSize];
  const int num_nodes = std::min<int>(
      trie_->commonPrefixSearch(w.data(), trie_results, kResultSize, w.size()),
      kResultSize);

  if (found) *found = (num_nodes > 0);
  if (num_nodes == 0) {
    return std::min<int>(w.size(), string_util::OneCharLen(w.data()));
  }

  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }

  return mblen;
}

std::string PrefixMatcher::GlobalReplace(absl::string_view w,
                                         absl::string_view out) const {
  std::string result;
  while (!w.empty()) {
    bool found = false;
    const int mblen = PrefixMatch(w, &found);
    if (found) {
      result.append(out.data(), out.size());
    } else {
      result.append(w.data(), mblen);
    }
    w.remove_prefix(mblen);
  }
  return result;
}

}  // namespace normalizer
}  // namespace sentencepiece
