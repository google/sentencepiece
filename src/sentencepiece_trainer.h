// Copyright 2018 Google Inc.
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

#ifndef SENTENCEPIECE_TRAINER_H_
#define SENTENCEPIECE_TRAINER_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sentencepiece_processor.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"

namespace sentencepiece {

class TrainerSpec;
class NormalizerSpec;
class ModelProto;

namespace normalizer {
class Normalizer;
}  // namespace normalizer

// Iterator over the training sentences.
// Training sentences are loaded sequentially as follows:
//
// for (; !it.done(); it.Next()) {
//    const std::string &s = it.value();
// }
// RETURN_IF_ERROR(it.status());
//
class SentenceIterator {
 public:
  virtual ~SentenceIterator() = default;
  // Returns true if iteration finishes (including error case).
  // Uses SentenceIterator::status() method to know whether
  // all sentences are loaded successfully.
  [[nodiscard]] virtual bool done() const = 0;
  virtual void Next() = 0;
  [[nodiscard]] virtual const std::string& value() const = 0;
  virtual absl::Status status() const = 0;
};

// Container for C++ objects passed to Trainer.
struct TrainerComponents {
  SentenceIterator* sentence_iterator = nullptr;
  using Pretokenizer =
      std::function<std::vector<std::string>(absl::string_view)>;
  Pretokenizer pretokenizer = nullptr;
  bool allow_inconsistent_pretokenization = false;

  TrainerComponents();
  ~TrainerComponents();
  TrainerComponents(const TrainerComponents& other);
  TrainerComponents(TrainerComponents&& other) noexcept;
  TrainerComponents& operator=(const TrainerComponents& other);
  TrainerComponents& operator=(TrainerComponents&& other) noexcept;

  // Accessors for spec Protobuf messages (Pimpl)
  TrainerSpec* mutable_trainer_spec();
  const TrainerSpec& trainer_spec() const;

  NormalizerSpec* mutable_normalizer_spec();
  const NormalizerSpec& normalizer_spec() const;

  NormalizerSpec* mutable_denormalizer_spec();
  const NormalizerSpec& denormalizer_spec() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class SentencePieceTrainer {
 public:
  // Trains SentencePiece model with `components`.
  static absl::Status Train(const TrainerComponents& components,
                            std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with command-line string in `args` and
  // `components`.
  static absl::Status Train(absl::string_view args,
                            const TrainerComponents& components,
                            std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with `kwargs` map and `components`.
  static absl::Status Train(
      const std::unordered_map<std::string, std::string>& kwargs,
      const TrainerComponents& components,
      std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with `trainer_spec`.
  // Default `normalizer_spec` is used.
  // When `sentence_iterator` is passed, load sentences from the iterator.
  [[deprecated("Use Train(const TrainerComponents&, ...) instead.")]]
  static absl::Status Train(const TrainerSpec& trainer_spec,
                            SentenceIterator* sentence_iterator = nullptr,
                            std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with `trainer_spec` and
  // `normalizer_spec`.
  // When `sentence_iterator` is passed, load sentences from the iterator.
  [[deprecated("Use Train(const TrainerComponents&, ...) instead.")]]
  static absl::Status Train(const TrainerSpec& trainer_spec,
                            const NormalizerSpec& normalizer_spec,
                            SentenceIterator* sentence_iterator = nullptr,
                            std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with `trainer_spec`, `normalizer_spec`
  // and `denormalizer_spec`.
  // When `sentence_iterator` is passed, load sentences from the iterator.
  [[deprecated("Use Train(const TrainerComponents&, ...) instead.")]]
  static absl::Status Train(const TrainerSpec& trainer_spec,
                            const NormalizerSpec& normalizer_spec,
                            const NormalizerSpec& denormalizer_spec,
                            SentenceIterator* sentence_iterator = nullptr,
                            std::string* serialized_model_proto = nullptr);
  // Trains SentencePiece model with command-line string in `args`,
  // e.g.,
  // '--input=data --model_prefix=m --vocab_size=8192 model_type=unigram'
  // When `sentence_iterator` is passed, load sentences from the iterator.
  [[deprecated(
      "Use Train(absl::string_view, const TrainerComponents&, ...) instead.")]]
  static absl::Status Train(absl::string_view args,
                            SentenceIterator* sentence_iterator = nullptr,
                            std::string* serialized_model_proto = nullptr);

  // Trains SentencePiece model with mapin `kwargs`.
  // e.g., {{"input", "data"}, {"model_prefix, "m"}, {"vocab_size", "8192"}...}
  [[deprecated(
      "Use Train(const std::unordered_map<std::string, std::string>&, const "
      "TrainerComponents&, ...) instead.")]]
  static absl::Status Train(
      const std::unordered_map<std::string, std::string>& kwargs,
      SentenceIterator* sentence_iterator = nullptr,
      std::string* serialized_model_proto = nullptr);

  // The same as above, but passes the list of sentences.
  [[deprecated(
      "Use Train(absl::string_view, const TrainerComponents&, ...) instead.")]]
  static absl::Status Train(absl::string_view args,
                            const std::vector<std::string>& sentences,
                            std::string* serialized_model_proto = nullptr);

  // The same as above, but passes the list of sentences.
  [[deprecated(
      "Use Train(const std::unordered_map<std::string, std::string>&, const "
      "TrainerComponents&, ...) instead.")]]
  static absl::Status Train(
      const std::unordered_map<std::string, std::string>& kwargs,
      const std::vector<std::string>& sentences,
      std::string* serialized_model_proto = nullptr);

  // Handy function to make a normalizer spec from the pre-compiled
  // normalization name. Do not use this method in production as it crashes
  // When `name` is invalid. Useful for unittesting.
  static NormalizerSpec GetNormalizerSpec(absl::string_view name);

  // Populates necessary fields (precompiled_charmap) from
  // `NormalizerSpec::name` or `NormalizerSpec::normalization_rule_tsv`.
  static absl::Status PopulateNormalizerSpec(NormalizerSpec* normalizer_spec,
                                             bool is_denormalizer = false);

  // Overrides `trainer_spec`, `normalizer_spec`, `denormalizer_spec` with the
  // std::unordered_map in `kwargs`.
  static absl::Status MergeSpecsFromArgs(
      const std::unordered_map<std::string, std::string>& kwargs,
      TrainerSpec* trainer_spec, NormalizerSpec* normalizer_spec,
      NormalizerSpec* denormalizer_spec);

  static absl::Status MergeSpecsFromArgs(
      const std::unordered_map<std::string, std::string>& kwargs,
      TrainerComponents* components);

  // Overrides `trainer_spec`, `normalizer_spec`, `denormalizer_spec` with the
  // command line flags in `args`.
  static absl::Status MergeSpecsFromArgs(absl::string_view args,
                                         TrainerSpec* trainer_spec,
                                         NormalizerSpec* normalizer_spec,
                                         NormalizerSpec* denormalizer_spec);

  static absl::Status MergeSpecsFromArgs(absl::string_view args,
                                         TrainerComponents* components);

  // Helper function to set `field_name=value` in `message`.
  // When `field_name` is repeated, multiple values can be passed
  // with comma-separated values. `field_name` must not be a nested message.
  // The body of these functions are automatically generated with
  // data/gen_spec_parser.pl
  static absl::Status SetProtoField(absl::string_view name,
                                    absl::string_view value,
                                    TrainerSpec* message);

  static absl::Status SetProtoField(absl::string_view name,
                                    absl::string_view value,
                                    NormalizerSpec* message);

  // Populates model type from string representation, e.g., "bpe".
  // Supported model: "unigram", "bpe", "word", "char".
  static absl::Status PopulateModelTypeFromString(absl::string_view type,
                                                  TrainerSpec* trainer_spec);

 private:
  SentencePieceTrainer() {}
  ~SentencePieceTrainer() = default;
};

class SentencePieceNormalizer {
 public:
  SentencePieceNormalizer();
  virtual ~SentencePieceNormalizer();

  virtual absl::Status Load(std::unique_ptr<ModelProto> model_proto);
  virtual absl::Status Load(std::unique_ptr<NormalizerSpec> normalizer_spec);

  virtual absl::Status Load(absl::string_view filename);

  virtual absl::Status LoadFromSerializedProto(absl::string_view serialized);
  virtual absl::Status LoadFromSerializedNormalizerSpec(
      absl::string_view serialized);

  virtual absl::Status LoadFromRuleTSV(absl::string_view filename);

  virtual absl::Status LoadFromRuleName(absl::string_view name);

  virtual absl::Status LoadFromMap(
      absl::Span<const std::pair<std::string, std::string>> norm_map);

  virtual absl::Status Decompile(
      std::vector<std::pair<std::string, std::string>>* norm_map) const;

  virtual absl::Status Normalize(absl::string_view input,
                                 std::string* normalized) const;

  virtual absl::Status Normalize(absl::string_view input,
                                 std::string* normalized,
                                 std::vector<size_t>* norm_to_orig) const;

  [[nodiscard]] virtual std::string Normalize(absl::string_view input) const;

  [[nodiscard]] virtual NormalizerSpec* mutable_normalizer_spec();

  [[nodiscard]] virtual std::string serialized_model_proto() const;

  [[nodiscard]] virtual std::string serialized_normalizer_spec() const;

 private:
  std::unique_ptr<normalizer::Normalizer> normalizer_;
  std::unique_ptr<NormalizerSpec> normalizer_spec_;
};

// Converts the utf8 byte spans into Unicode char span.
void ConvertToUnicodeAlignment(absl::string_view orig, absl::string_view norm,
                               std::vector<size_t>* norm_to_orig);

// Sets data dir including the pre-compiled normalization data.
// The implementation is found in util.cc
void SetDataDir(absl::string_view data_dir);

}  // namespace sentencepiece

#endif  // SENTENCEPIECE_TRAINER_H_
