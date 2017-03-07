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

#include "builder.h"

#ifdef ENABLE_NFKC_COMPILE
#include <unicode/errorcode.h>
#include <unicode/locid.h>
#include <unicode/normlzr.h>
#include <unicode/numfmt.h>
#include <unicode/rbnf.h>
#include <unicode/utypes.h>
#endif

#include <set>

#include "normalization_rule.h"
#include "normalizer.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {
namespace {

#ifdef ENABLE_NFKC_COMPILE
// Normalize |input| with ICU's normalizer with |mode|.
Builder::Chars UnicodeNormalize(UNormalizationMode mode,
                                const Builder::Chars &input) {
  const std::string utf8 = string_util::UnicodeTextToUTF8(input);
  CHECK(!utf8.empty());

  icu::UnicodeString ustr;
  const size_t utf8_length = utf8.size();
  UChar *utf16 = ustr.getBuffer(utf8.size() + 1);
  int32 utf16_length = 0;
  icu::ErrorCode icuerrorcode;
  u_strFromUTF8Lenient(utf16, ustr.getCapacity(), &utf16_length, utf8.data(),
                       utf8_length, icuerrorcode);
  ustr.releaseBuffer(utf16_length);

  UErrorCode status = U_ZERO_ERROR;
  icu::UnicodeString dst;
  icu::Normalizer::normalize(ustr, mode, 0, dst, status);
  CHECK(U_SUCCESS(status));
  std::string normalized;
  normalized.reserve(dst.length() * 3);
  dst.toUTF8String(normalized);
  return string_util::UTF8ToUnicodeText(normalized);
}

Builder::Chars ToNFKD(const Builder::Chars &input) {
  return UnicodeNormalize(UNORM_NFKD, input);
}

Builder::Chars ToNFKC(const Builder::Chars &input) {
  return UnicodeNormalize(UNORM_NFKC, input);
}

Builder::Chars ToNFC(const Builder::Chars &input) {
  return UnicodeNormalize(UNORM_NFC, input);
}

Builder::Chars ToNFD(const Builder::Chars &input) {
  return UnicodeNormalize(UNORM_NFD, input);
}

// Given an NFKD-normalized string, returns a set of all strings which are
// normalized into the same |nfkd|. |norm2orig| is the normalized to
// un-normalized character mapping.
std::vector<Builder::Chars> ExpandUnnormalized(
    const Builder::Chars &nfkd,
    const std::map<char32, std::set<char32>> &norm2orig) {
  CHECK(!nfkd.empty());
  std::vector<Builder::Chars> results;
  for (const auto c : port::FindOrDie(norm2orig, nfkd[0])) {
    results.push_back({c});
  }
  for (size_t i = 1; i < nfkd.size(); ++i) {
    const auto &orig = port::FindOrDie(norm2orig, nfkd[i]);
    std::vector<Builder::Chars> new_results;
    for (const auto &r : results) {
      for (const auto c : orig) {
        new_results.emplace_back(r);
        new_results.back().push_back(c);
      }
    }
    results = std::move(new_results);
  }
  CHECK_EQ(nfkd.size(), results[0].size());
  return results;
}
#endif

// Normalizes |src| with |chars_map| and returns normalized Chars.
// |max_len| specifies the maximum length of the key in |chars_map|.
Builder::Chars Normalize(const Builder::CharsMap &chars_map,
                         const Builder::Chars &src, int max_len) {
  CHECK_GE(max_len, 1);
  Builder::Chars normalized;

  for (size_t i = 0; i < src.size();) {
    Builder::CharsMap::const_iterator it = chars_map.end();
    const size_t slice = std::min<size_t>(i + max_len, src.size());
    // starts with the longest prefix.
    Builder::Chars key(src.begin() + i, src.begin() + slice);
    while (!key.empty()) {
      it = chars_map.find(key);
      if (it != chars_map.end()) {
        break;
      }
      key.pop_back();  // remove the last character.
    }

    // Consumes one character when no rule is found.
    if (it == chars_map.end()) {
      normalized.push_back(src[i]);
      ++i;
    } else {
      CHECK(!it->second.empty());
      std::copy(it->second.begin(), it->second.end(),
                std::back_inserter(normalized));
      i += it->first.size();
    }
  }

  return normalized;
}
}  // namespace

// static
std::string Builder::CompileCharsMap(const CharsMap &chars_map) {
  CHECK(!chars_map.empty());

  LOG(INFO) << "Loading CharsMap of size " << chars_map.size();

  // Aggregates the same target strings to save footprint.
  std::map<Chars, int> normalized2pos;
  for (const auto &p : chars_map) {
    normalized2pos[p.second] = 0;
  }

  std::string normalized;
  for (auto &p : normalized2pos) {
    p.second = normalized.size();  // stores the pointer (position).
    const std::string utf8_out = string_util::UnicodeTextToUTF8(p.first);
    normalized += utf8_out;
    normalized += '\0';
  }

  std::vector<std::pair<std::string, int>> kv;  // key-value of Trie.
  for (const auto &p : chars_map) {
    // The value of Trie stores the pointer to the normalized string.
    const std::string utf8_in = string_util::UnicodeTextToUTF8(p.first);
    kv.emplace_back(utf8_in, port::FindOrDie(normalized2pos, p.second));
  }

  std::sort(kv.begin(), kv.end());
  std::vector<const char *> key(kv.size());
  std::vector<int> value(kv.size());
  for (size_t i = 0; i < kv.size(); ++i) {
    key[i] = kv[i].first.c_str();
    value[i] = kv[i].second;
  }

  Darts::DoubleArray trie;
  CHECK_EQ(
      0,
      trie.build(key.size(), const_cast<char **>(&key[0]), nullptr, &value[0]))
      << "cannot build double-array";

  int max_nodes_size = 0;
  std::vector<Darts::DoubleArray::result_pair_type> results(
      2 * Normalizer::kMaxTrieResultsSize);
  for (const char *str : key) {
    const int num_nodes = trie.commonPrefixSearch(str, results.data(),
                                                  results.size(), strlen(str));
    max_nodes_size = std::max(num_nodes, max_nodes_size);
  }
  CHECK_LT(max_nodes_size, Normalizer::kMaxTrieResultsSize)
      << "This charmaps contain many shared prefix. "
      << "The number of shared prefix must be less than "
      << Normalizer::kMaxTrieResultsSize;

  StringPiece trie_blob(static_cast<const char *>(trie.array()),
                        trie.size() * trie.unit_size());
  const std::string blob =
      Normalizer::EncodePrecompiledCharsMap(trie_blob, normalized);

  LOG(INFO) << "Generated normalizer blob. size= " << blob.size();

  return blob;
}

// static
std::string Builder::GetPrecompiledCharsMap(const std::string &name) {
  std::string result;
  for (size_t i = 0; i < kNormalizationRules_size; ++i) {
    const auto *blob = &kNormalizationRules_blob[i];
    if (blob->name == name) {
      result.assign(blob->data, blob->size);
      return result;
    }
  }
  LOG(FATAL) << "No precompiled charsmap is found: " << name;
  return result;
}

// static
NormalizerSpec Builder::GetNormalizerSpec(const std::string &name) {
  NormalizerSpec spec;
  spec.set_name(name);
  spec.set_precompiled_charsmap(GetPrecompiledCharsMap(name));
  return spec;
}

// static
Builder::CharsMap Builder::BuildNFKCMap() {
#ifdef ENABLE_NFKC_COMPILE
  LOG(INFO) << "Running BuildNFKCMap";

  // Set of fully NFKD decomposed characters.
  std::set<Builder::Chars> nfkd_decomposed;

  // Fully normalized one character to unnormalized one character map.
  std::map<char32, std::set<char32>> norm2orig;

  Builder::CharsMap nfkc_map;  // The final NFKC mapping.

  constexpr int kMaxUnicode = 0x110000;
  for (char32 cp = 1; cp <= kMaxUnicode; ++cp) {
    if (!U_IS_UNICODE_CHAR(cp)) {
      continue;
    }
    // Aggregates single character to fully NFKC normalized characters.
    const auto nfkc = ToNFKC({cp});
    if (nfkc.size() >= 2 || (nfkc.size() == 1 && nfkc[0] != cp)) {
      nfkc_map[{cp}] = nfkc;
    }
    const auto nfkd = ToNFKD({cp});
    if (nfkd.size() == 1) {
      // Aggregates reverse mapping from normalized to unnormalized character.
      norm2orig[nfkd[0]].insert(cp);
    } else {
      // One character is decomposed into multiple characters.
      nfkd_decomposed.insert(nfkd);
    }
  }

  for (const auto &nfkd : nfkd_decomposed) {
    const auto nfkc = ToNFC(nfkd);
    // This case is already covered by single-character to NFKC mapping.
    if (nfkc == nfkd) {
      continue;
    }
    // Expand all possible sequences which are normalized into the same |nfkd|.
    for (const auto &nfkd_orig : ExpandUnnormalized(nfkd, norm2orig)) {
      if (nfkd_orig != nfkc) {
        nfkc_map[nfkd_orig] = nfkc;
      }
    }
  }

  return RemoveRedundantMap(nfkc_map);
#else
  LOG(FATAL) << "NFKC compile is not enabled."
             << " rebuild with ./configure --enable-nfkc-compile";
  return {};
#endif
}

// static
Builder::CharsMap Builder::BuildIdentityMap() {
  // Adds one dummy entry since empty rule is not allowed.
  const CharsMap result = {{{0x0020}, {0x0020}}};
  return result;
}

// static
Builder::CharsMap Builder::BuildMapFromFile(StringPiece filename) {
  LOG(INFO) << "Loading maping file: " << filename.data();
  io::InputBuffer input(filename);
  std::string line;
  CharsMap chars_map;
  while (input.ReadLine(&line)) {
    const auto fields = string_util::SplitPiece(line, "\t");
    CHECK_GE(fields.size(), 2);
    std::vector<char32> src, trg;
    for (const auto &s : string_util::SplitPiece(fields[0], " ")) {
      src.push_back(string_util::HexToInt<char32>(s));
    }
    for (const auto &s : string_util::SplitPiece(fields[1], " ")) {
      trg.push_back(string_util::HexToInt<char32>(s));
    }
    CHECK(!src.empty());
    CHECK(!trg.empty());
    chars_map[src] = trg;
  }
  return chars_map;
}

// static
Builder::CharsMap Builder::RemoveRedundantMap(const CharsMap &chars_map) {
  CharsMap new_chars_map;

  size_t max_len = 0;
  for (const auto &p : chars_map) {
    max_len = std::max(p.first.size(), max_len);
    if (p.first.size() == 1) {
      new_chars_map.insert(p);
    }
  }
  CHECK_GT(max_len, 0);

  // Checks whether the rules with size of |len| can be normalized by
  // the rules with size of [1 .. len - 1].
  for (size_t len = 2; len <= max_len; ++len) {
    for (const auto &p : chars_map) {
      if (p.first.size() == len &&
          p.second != Normalize(new_chars_map, p.first, len - 1)) {
        new_chars_map.insert(p);
      }
    }
  }

  // Verify all characters in |chars_map| are normalized by |new_chars_map|.
  for (const auto &p : chars_map) {
    CHECK_EQ(p.second, Normalize(new_chars_map, p.first, max_len));
  }

  return new_chars_map;
}
}  // namespace normalizer
}  // namespace sentencepiece
