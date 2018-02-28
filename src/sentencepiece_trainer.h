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

namespace sentencepiece {

class TrainerSpec;
class NormalizerSpec;

class SentencePieceTrainer {
 public:
  // Entry point for main function.
  static void Train(int argc, char **argv);

  // Train from params with a single line.
  // "--input=foo --model_prefix=m --vocab_size=1024"
  static void Train(const std::string &arg);

  // Trains SentencePiece model with `trainer_spec`.
  // Default `normalizer_spec` is used.
  static void Train(const TrainerSpec &trainer_spec);

  // Trains SentencePiece model with `trainer_spec` and
  // `normalizer_spec`.
  static void Train(const TrainerSpec &trainer_spec,
                    const NormalizerSpec &normalizer_spec);

  SentencePieceTrainer() = delete;
  ~SentencePieceTrainer() = delete;
};
}  // namespace sentencepiece
#endif  // SENTENCEPIECE_TRAINER_H_
