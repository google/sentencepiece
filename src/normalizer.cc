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
#include "common.h"
#include "stringpiece.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {

Normalizer::Normalizer(const NormalizerSpec &spec) : spec_(&spec) {
  StringPiece index = spec.precompiled_charsmap();
  CHECK(!index.empty());

  StringPiece trie_blob, normalized;
  DecodePrecompiledCharsMap(index, &trie_blob, &normalized);

  // Reads the body of double array.
  trie_ = port::MakeUnique<Darts::DoubleArray>();

  // The second arg of set_array is not the size of blob,
  // but the number of double array units.
  trie_->set_array(const_cast<char *>(trie_blob.data()),
                   trie_blob.size() / trie_->unit_size());

  normalized_ = normalized.data();
}

Normalizer::~Normalizer() {}

void Normalizer::Normalize(StringPiece input, std::string *normalized,
                           std::vector<size_t> *norm_to_orig) const {
  CHECK_NOTNULL(norm_to_orig)->clear();
  CHECK_NOTNULL(normalized)->clear();

  if (input.empty()) {
    return;
  }

  int consumed = 0;

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
    return;
  }

  // Reserves the output buffer to avoid re-allocations.
  const size_t kReservedSize = input.size() * 3;
  normalized->reserve(kReservedSize);
  norm_to_orig->reserve(kReservedSize);

  // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
  // if escape_whitespaces() is set (default = true).
  const StringPiece kSpaceSymbol = "\xe2\x96\x81";

  // Adds a space symbol as a prefix (default is true)
  // With this prefix, "world" and "hello world" are converted into
  // "_world" and "_hello_world", which help the trainer to extract
  // "_world" as one symbol.
  if (spec_->add_dummy_prefix()) {
    if (spec_->escape_whitespaces()) {
      normalized->append(kSpaceSymbol.data(), kSpaceSymbol.size());
      for (size_t n = 0; n < kSpaceSymbol.size(); ++n) {
        norm_to_orig->push_back(consumed);
      }
    } else {
      normalized->append(" ");
      norm_to_orig->push_back(consumed);
    }
  }

  bool is_prev_space = spec_->remove_extra_whitespaces();
  while (!input.empty()) {
    auto p = NormalizePrefix(input);
    StringPiece sp = p.first;

    // Removes heading spaces in sentence piece,
    // if the previous sentence piece ends with whitespace.
    while (is_prev_space && sp.Consume(" ")) {
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
        } else {
          *normalized += data[n];
          norm_to_orig->push_back(consumed);
        }
      }
      // Checks whether the last character of sp is whitespace.
      is_prev_space = sp.ends_with(" ");
    }

    consumed += p.second;
    input.remove_prefix(p.second);
    if (!spec_->remove_extra_whitespaces()) {
      is_prev_space = false;
    }
  }

  // Ignores tailing space.
  if (spec_->remove_extra_whitespaces()) {
    const StringPiece space = spec_->escape_whitespaces() ? kSpaceSymbol : " ";
    while (string_util::EndsWith(*normalized, space)) {
      const int length = normalized->size() - space.size();
      CHECK_GE(length, 0);
      consumed = (*norm_to_orig)[length];
      normalized->resize(length);
      norm_to_orig->resize(length);
    }
  }

  norm_to_orig->push_back(consumed);
  CHECK_EQ(norm_to_orig->size(), normalized->size() + 1);
}

std::string Normalizer::Normalize(StringPiece input) const {
  std::vector<size_t> norm_to_orig;
  std::string normalized;
  Normalize(input, &normalized, &norm_to_orig);
  return normalized;
}

std::pair<StringPiece, int> Normalizer::NormalizePrefix(
    StringPiece input) const {
  CHECK(!input.empty());

  // Allocates trie_results in stack, which makes the encoding speed 36% faster.
  // (38k sentences/sec => 60k sentences/sec).
  // Builder checks that the result size never exceeds kMaxTrieResultsSize.
  // This array consumes 0.5kByte in stack, which is less than
  // default stack frames (16kByte).
  Darts::DoubleArray::result_pair_type
      trie_results[Normalizer::kMaxTrieResultsSize];

  const size_t num_nodes = CHECK_NOTNULL(trie_)->commonPrefixSearch(
      input.data(), trie_results, Normalizer::kMaxTrieResultsSize,
      input.size());

  // Finds the longest rule.
  size_t longest_length = 0;
  int longest_value = 0;
  for (size_t k = 0; k < num_nodes; ++k) {
    if (longest_length == 0 || trie_results[k].length > longest_length) {
      longest_length = trie_results[k].length;  // length of prefix
      longest_value = trie_results[k].value;    // pointer to |normalized_|.
    }
  }

  std::pair<StringPiece, int> result;
  if (longest_length == 0) {
    result.second = std::min<int>(
        input.size(), std::max<int>(1, string_util::OneCharLen(input.data())));
    result.first.set(input.data(), result.second);
  } else {
    result.second = longest_length;
    // No need to pass the size of normalized sentence,
    // since |normalized| is delimitered by "\0".
    result.first.set(&normalized_[longest_value]);
  }

  CHECK(!result.first.empty());
  CHECK_GT(result.second, 0);

  return result;
}

// static
std::string Normalizer::EncodePrecompiledCharsMap(StringPiece trie_blob,
                                                  StringPiece normalized) {
  // <trie size(4byte)><double array trie><normalized string>
  std::string blob;
  blob.append(string_util::EncodePOD<uint32>(trie_blob.size()));
  blob.append(trie_blob.data(), trie_blob.size());
  blob.append(normalized.data(), normalized.size());
  return blob;
}

// static
void Normalizer::DecodePrecompiledCharsMap(StringPiece blob,
                                           StringPiece *trie_blob,
                                           StringPiece *normalized) {
  uint32 trie_blob_size = 0;
  CHECK_GT(blob.size(), sizeof(trie_blob_size));

  CHECK(string_util::DecodePOD<uint32>(
      StringPiece(blob.data(), sizeof(trie_blob_size)), &trie_blob_size));
  CHECK_LT(trie_blob_size, blob.size());
  blob.remove_prefix(sizeof(trie_blob_size));

  CHECK_NOTNULL(trie_blob)->set(blob.data(), trie_blob_size);

  blob.remove_prefix(trie_blob_size);
  CHECK_NOTNULL(normalized)->set(blob.data(), blob.size());
}
}  // namespace normalizer
}  // namespace sentencepiece
