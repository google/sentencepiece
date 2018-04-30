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

#include <string>
#include "sentencepiece_processor.h"

namespace google {
namespace protobuf {
class Message;
}  // namespace protobuf
}  // namespace google

namespace sentencepiece {

class TrainerSpec;
class NormalizerSpec;

class SentencePieceTrainer {
 public:
  // Trains SentencePiece model with `trainer_spec`.
  // Default `normalizer_spec` is used.
  static util::Status Train(const TrainerSpec &trainer_spec);

  // Trains SentencePiece model with `trainer_spec` and
  // `normalizer_spec`.
  static util::Status Train(const TrainerSpec &trainer_spec,
                            const NormalizerSpec &normalizer_spec);

  // Trains SentencePiece model with command-line string in `args`,
  // e.g.,
  // '--input=data --model_prefix=m --vocab_size=8192 model_type=unigram'
  static util::Status Train(const std::string &args);

  // Overrides `trainer_spec` and `normalizer_spec` with the
  // command-line string in `args`.
  static util::Status MergeSpecsFromArgs(const std::string &args,
                                         TrainerSpec *trainer_spec,
                                         NormalizerSpec *normalizer_spec);

  // Helper function to set `field_name=value` in `message`.
  // When `field_name` is repeated, multiple values can be passed
  // with comma-separated values. `field_name` must not be a nested message.
  static util::Status SetProtoField(const std::string &field_name,
                                    const std::string &value,
                                    google::protobuf::Message *message);

  SentencePieceTrainer() = delete;
  ~SentencePieceTrainer() = delete;
};
}  // namespace sentencepiece
#endif  // SENTENCEPIECE_TRAINER_H_
