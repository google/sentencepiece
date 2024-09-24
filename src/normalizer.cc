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

#include <utility>
#include <vector>

#include "common.h"
#include "filesystem.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {

constexpr int Normalizer::kMaxTrieResultsSize;

Normalizer::Normalizer(const NormalizerSpec &spec, const TrainerSpec &trainer_spec)
    : spec_(&spec),
      treat_whitespace_as_suffix_(trainer_spec.treat_whitespace_as_suffix()),
      verbatim_control_char_(trainer_spec.verbatim_control_char()),
      status_(util::OkStatus()) {
  Init();
}

Normalizer::Normalizer(const NormalizerSpec &spec)
    : spec_(&spec), status_(util::OkStatus()), fixed_stats_{0} {
  Init();
}

Normalizer::~Normalizer() {
  if (!spec_->normalization_report_file().empty()) {
    LOG(INFO) << "Writing normalization report to " << spec_->normalization_report_file();
    auto report = filesystem::NewWritableFile(spec_->normalization_report_file());
    report->WriteLine(
        std::string("copied_as_is\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsCopiedAsIs].load()));
    report->WriteLine(
        std::string("heading_space_removed\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsHeadingSpaceRemoved].load()));
	report->WriteLine(
	    std::string("trailing_space_removed\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsTrailingSpaceRemoved].load()));
	report->WriteLine(
	    std::string("extra_space_removed\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsExtraSpaceRemoved].load()));
    report->WriteLine(
        std::string("empty_after_removing_space\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsEmptyAfterRemovingSpaces].load()));
    report->WriteLine(
        std::string("space_replaced_with_u2581\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsSpaceReplacedWith2581].load()));
    report->WriteLine(
        std::string("malformed_utf8\t") + string_util::SimpleItoa(fixed_stats_[kFixedNormalizationStatisticsMalformedUTF8].load()));

	std::vector<std::pair<int64, absl::string_view>> stats;
    for (auto& p : stats_) {
      stats.emplace_back(-p.second, p.first);
    }
    std::sort(stats.begin(), stats.end());
    for (auto& p : stats) {
      report->WriteLine(absl::StrCat(p.second, "\t", string_util::SimpleItoa(-p.first)));
    }
  }
}

void Normalizer::Init() {
  absl::string_view index = spec_->precompiled_charsmap();
  if (!index.empty()) {
    absl::string_view trie_blob, normalized;
#ifdef IS_BIG_ENDIAN
    status_ = DecodePrecompiledCharsMap(index, &trie_blob, &normalized,
                                        &precompiled_charsmap_buffer_);
#else
    status_ = DecodePrecompiledCharsMap(index, &trie_blob, &normalized);
#endif
    if (!status_.ok()) return;

    // Reads the body of double array.
    trie_ = absl::make_unique<Darts::DoubleArray>();

    // The second arg of set_array is not the size of blob,
    // but the number of double array units.
    trie_->set_array(const_cast<char *>(trie_blob.data()),
                     trie_blob.size() / trie_->unit_size());

    normalized_ = normalized.data();
  }
}

util::Status Normalizer::Normalize(absl::string_view input,
                                   std::string *normalized,
                                   std::vector<size_t> *norm_to_orig) const {
  norm_to_orig->clear();
  normalized->clear();

  if (input.empty()) {
    return util::OkStatus();
  }

  RETURN_IF_ERROR(status());

  int consumed = 0;
  bool verbatim = static_cast<uint8_t>(input[0]) == verbatim_control_char_;
  if (verbatim) {
    input.remove_prefix(1);
  }

  // Ignores heading space.
  if (spec_->remove_extra_whitespaces()) {
    while (!input.empty()) {
      const auto p = NormalizePrefix(input);
      if (p.first != " ") {
        break;
      }
      fixed_stats_[kFixedNormalizationStatisticsHeadingSpaceRemoved]++;
      input.remove_prefix(p.second);
      consumed += p.second;
    }
  }

  // all chars are whitespace.
  if (input.empty()) {
    fixed_stats_[kFixedNormalizationStatisticsEmptyAfterRemovingSpaces]++;
    return util::OkStatus();
  }

  // Reserves the output buffer to avoid re-allocations.
  const size_t kReservedSize = input.size() * 3 + verbatim;
  normalized->reserve(kReservedSize);
  norm_to_orig->reserve(kReservedSize);
  if (verbatim) {
    normalized->push_back(verbatim_control_char_);
  }

  // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
  // if escape_whitespaces() is set (default = true).
  const absl::string_view kSpaceSymbol = "\xe2\x96\x81";

  // adds kSpaceSymbol to the current context.
  auto add_ws = [this, &consumed, &normalized, &norm_to_orig, &kSpaceSymbol]() {
    static auto SP_ADD_DUMMY_PREFIX = std::getenv("SP_ADD_DUMMY_PREFIX");
    if (SP_ADD_DUMMY_PREFIX != nullptr && strcmp(SP_ADD_DUMMY_PREFIX, "no") == 0) {
      return;
    }
    if (spec_->escape_whitespaces()) {
      normalized->append(kSpaceSymbol.data(), kSpaceSymbol.size());
      for (size_t n = 0; n < kSpaceSymbol.size(); ++n) {
        norm_to_orig->push_back(consumed);
      }
      fixed_stats_[kFixedNormalizationStatisticsSpaceReplacedWith2581]++;
    } else {
      normalized->append(" ");
      norm_to_orig->push_back(consumed);
    }
  };

  // Adds a space symbol as a prefix (default is true)
  // With this prefix, "world" and "hello world" are converted into
  // "_world" and "_hello_world", which help the trainer to extract
  // "_world" as one symbol.
  if (!treat_whitespace_as_suffix_ && spec_->add_dummy_prefix()) add_ws();

  bool is_prev_space = spec_->remove_extra_whitespaces() && !verbatim;
  while (!input.empty()) {
    auto p = NormalizePrefix(input);
    absl::string_view sp = p.first;

    // Removes heading spaces in sentence piece,
    // if the previous sentence piece ends with whitespace.
    while (is_prev_space && absl::ConsumePrefix(&sp, " ")) {
      fixed_stats_[kFixedNormalizationStatisticsExtraSpaceRemoved]++;
    }

    if (!sp.empty()) {
      const char *data = sp.data();
      for (size_t n = 0; n < sp.size(); ++n) {
        if (spec_->escape_whitespaces() && data[n] == ' ') {
          // replace ' ' with kSpaceSymbol.
          normalized->append(kSpaceSymbol.data(), kSpaceSymbol.size());
          for (size_t m = 0; m < kSpaceSymbol.size(); ++m) {
            norm_to_orig->push_back(consumed);
          }
          fixed_stats_[kFixedNormalizationStatisticsSpaceReplacedWith2581]++;
        } else {
          *normalized += data[n];
          norm_to_orig->push_back(consumed);
        }
      }
      // Checks whether the last character of sp is whitespace.
      is_prev_space = absl::EndsWith(sp, " ");
    }

    consumed += p.second;
    input.remove_prefix(p.second);
    if (!spec_->remove_extra_whitespaces() || verbatim) {
      is_prev_space = false;
    }
  }

  // Ignores trailing space.
  if (spec_->remove_extra_whitespaces() && !verbatim) {
    const absl::string_view space =
        spec_->escape_whitespaces() ? kSpaceSymbol : " ";
    while (absl::EndsWith(*normalized, space)) {
      const int length = normalized->size() - space.size();
      CHECK_GE_OR_RETURN(length, 0);
      consumed = (*norm_to_orig)[length];
      normalized->resize(length);
      norm_to_orig->resize(length);
      fixed_stats_[kFixedNormalizationStatisticsTrailingSpaceRemoved]++;
    }
  }

  // Adds a space symbol as a suffix (default is false)
  if (treat_whitespace_as_suffix_ && spec_->add_dummy_prefix()) add_ws();

  norm_to_orig->push_back(consumed);

  CHECK_EQ_OR_RETURN(norm_to_orig->size(), normalized->size() + 1 - verbatim);

  return util::OkStatus();
}

std::string Normalizer::Normalize(absl::string_view input) const {
  std::vector<size_t> norm_to_orig;
  std::string normalized;
  Normalize(input, &normalized, &norm_to_orig).IgnoreError();
  return normalized;
}

std::pair<absl::string_view, int> Normalizer::NormalizePrefix(
    absl::string_view input) const {
  std::pair<absl::string_view, int> result;

  if (input.empty()) return result;

  if (matcher_ != nullptr) {
    bool found = false;
    const int mblen = matcher_->PrefixMatch(input, &found);
    if (found) return std::make_pair(input.substr(0, mblen), mblen);
  }

  size_t longest_length = 0;
  int longest_value = 0;

  if (trie_ != nullptr) {
    // Allocates trie_results in stack, which makes the encoding speed 36%
    // faster. (38k sentences/sec => 60k sentences/sec). Builder checks that the
    // result size never exceeds kMaxTrieResultsSize. This array consumes
    // 0.5kByte in stack, which is less than default stack frames (16kByte).
    Darts::DoubleArray::result_pair_type
        trie_results[Normalizer::kMaxTrieResultsSize];

    const size_t num_nodes = trie_->commonPrefixSearch(
        input.data(), trie_results, Normalizer::kMaxTrieResultsSize,
        input.size());

    // Finds the longest rule.
    for (size_t k = 0; k < num_nodes; ++k) {
      if (longest_length == 0 || trie_results[k].length > longest_length) {
        longest_length = trie_results[k].length;  // length of prefix
        longest_value = trie_results[k].value;    // pointer to |normalized_|.
      }
    }
  }

  if (longest_length == 0) {
    size_t length = 0;
    if (!string_util::IsValidDecodeUTF8(input, &length)) {
      // Found a malformed utf8.
      // The rune is set to be 0xFFFD (REPLACEMENT CHARACTER),
      // which is a valid Unicode of three bytes in utf8,
      // but here we only consume one byte.
      result.second = 1;
      static const char kReplacementChar[] = "\xEF\xBF\xBD";
      result.first = absl::string_view(kReplacementChar);
      fixed_stats_[kFixedNormalizationStatisticsMalformedUTF8]++;
    } else {
      result.second = length;
      result.first = absl::string_view(input.data(), result.second);
      fixed_stats_[kFixedNormalizationStatisticsCopiedAsIs]++;
    }
  } else {
    result.second = longest_length;
    // No need to pass the size of normalized sentence,
    // since |normalized| is delimitered by "\0".
    result.first = absl::string_view(&normalized_[longest_value]);
    {
      const std::lock_guard lock(stats_lock_);
      stats_[absl::StrCat(
          "replaced '",
          absl::string_view(input.data(), result.second),
          "' -> '",
          result.first,
          "'"
      )]++;
    }
  }

  return result;
}

// static
std::string Normalizer::EncodePrecompiledCharsMap(
    absl::string_view trie_blob, absl::string_view normalized) {
  // <trie size(4byte)><double array trie><normalized string>
  std::string blob;
  blob.append(string_util::EncodePOD<uint32>(trie_blob.size()));
  blob.append(trie_blob.data(), trie_blob.size());

#ifdef IS_BIG_ENDIAN
  uint32 *data = reinterpret_cast<uint32 *>(const_cast<char *>(blob.data()));
  for (int i = 0; i < blob.size() / 4; ++i) data[i] = util::Swap32(data[i]);
#endif

  blob.append(normalized.data(), normalized.size());

  return blob;
}

// static
util::Status Normalizer::DecodePrecompiledCharsMap(
    absl::string_view blob, absl::string_view *trie_blob,
    absl::string_view *normalized, std::string *buffer) {
  uint32 trie_blob_size = 0;
  if (blob.size() <= sizeof(trie_blob_size) ||
      !string_util::DecodePOD<uint32>(
          absl::string_view(blob.data(), sizeof(trie_blob_size)),
          &trie_blob_size)) {
    return util::InternalError("Blob for normalization rule is broken.");
  }

#ifdef IS_BIG_ENDIAN
  trie_blob_size = util::Swap32(trie_blob_size);
#endif

  if (trie_blob_size >= blob.size()) {
    return util::InternalError("Trie data size exceeds the input blob size.");
  }

  blob.remove_prefix(sizeof(trie_blob_size));

#ifdef IS_BIG_ENDIAN
  CHECK_OR_RETURN(buffer);
  buffer->assign(blob.data(), trie_blob_size);
  uint32 *data = reinterpret_cast<uint32 *>(const_cast<char *>(buffer->data()));
  for (int i = 0; i < buffer->size() / 4; ++i) data[i] = util::Swap32(data[i]);
  *trie_blob = absl::string_view(buffer->data(), trie_blob_size);
#else
  *trie_blob = absl::string_view(blob.data(), trie_blob_size);
#endif

  blob.remove_prefix(trie_blob_size);
  *normalized = absl::string_view(blob.data(), blob.size());

  return util::OkStatus();
}

PrefixMatcher::PrefixMatcher(const std::set<absl::string_view> &dic) {
  if (dic.empty()) return;
  std::vector<const char *> key;
  key.reserve(dic.size());
  for (const auto &it : dic) key.push_back(it.data());
  trie_ = absl::make_unique<Darts::DoubleArray>();
  CHECK_EQ(0, trie_->build(key.size(), const_cast<char **>(&key[0]), nullptr,
                           nullptr));
}

int PrefixMatcher::PrefixMatch(absl::string_view w, bool *found) const {
  if (trie_ == nullptr) {
    if (found) *found = false;
    return std::min<int>(w.size(), string_util::OneCharLen(w.data()));
  }

  constexpr int kResultSize = 64;
  Darts::DoubleArray::result_pair_type trie_results[kResultSize];
  const int num_nodes =
      trie_->commonPrefixSearch(w.data(), trie_results, kResultSize, w.size());

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
