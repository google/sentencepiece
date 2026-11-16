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

#ifndef NORMALIZER_NORMALIZER_H_
#define NORMALIZER_NORMALIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest_prod.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::lite {
class SentencePieceLiteProcessor;
}

namespace sentencepiece::normalizer {

// Normalizer implements a simple text normalizer with
// user-defined string-to-string rules and leftmost longest
// matching. The rules of Normalizer are built with
// Builder::CompileCharsMap() method. Pre-compiled rules are
// also available via Builder::GetPrecompiledCharsMap(<name>) method.
//
// Normalizer wraps sentencepiece::lite::SentencePieceLiteProcessor
// to perform all text normalization.
class Normalizer {
 public:
  // Instantiates Normalizer with |spec|.
  // |spec| should not be deleted until Normalizer is destroyed.
  explicit Normalizer(const NormalizerSpec& spec);
  Normalizer(const NormalizerSpec& spec, const TrainerSpec& trainer_spec);
  virtual ~Normalizer();

  // Returns Status.
  // Normalizes function is valid only when status is OK.
  virtual absl::Status status() const { return status_; }

  // Normalizes a plain utf8 string into an internal representation for
  // Sentencepiece model. |norm_to_orig| stores the byte-alignment from
  // normalized string to the original input, if not null.
  // This function can do the following normalizations:
  // - Character normalization.
  //   (NFKC / full-width to half-width conversion etc).
  // - Adds a prefix space.
  // - Replaces a space with a meta symbol.
  // - Removing heading, tailing and other redundant spaces.
  virtual absl::Status Normalize(absl::string_view input,
                                 std::string* normalized,
                                 std::vector<size_t>* norm_to_orig) const;

  // Returns a normalized string without alignments.
  // This function is used in sentencepiece training.
  [[nodiscard]] virtual std::string Normalize(absl::string_view input) const;

  friend class Builder;

 private:
  FRIEND_TEST(NormalizerTest, EncodeDecodePrecompiledCharsMapTest);
  FRIEND_TEST(NormalizerTest, ManySharedPrefixesTest);

  void Init(const NormalizerSpec& spec, bool treat_whitespace_as_suffix);

  // Encodes trie_blob and normalized string and return compiled blob.
  static std::string EncodePrecompiledCharsMap(absl::string_view trie_blob,
                                               absl::string_view normalized);

  // Decodes blob into trie_blob and normalized string.
  static absl::Status DecodePrecompiledCharsMap(absl::string_view blob,
                                                absl::string_view* trie_blob,
                                                absl::string_view* normalized,
                                                std::string* buffer);

  // Maximum size of the return value of Trie, which corresponds
  // to the maximum size of shared common prefix in the chars map.
  static constexpr int kMaxTrieResultsSize = 32;

  // Normalizer's status.
  absl::Status status_;

  // Lite processor holding the model and its backing buffer.
  std::unique_ptr<lite::SentencePieceLiteProcessor> lite_processor_;
};
}  // namespace sentencepiece::normalizer

#endif  // NORMALIZER_NORMALIZER_H_
