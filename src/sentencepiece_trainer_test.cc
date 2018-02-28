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

#include "sentencepiece_trainer.h"
#include "sentencepiece_model.pb.h"
#include "testharness.h"

namespace sentencepiece {
namespace {

TEST(SentencePieceTrainerTest, TrainFromArgsTest) {
  SentencePieceTrainer::Train(
      "--input=../data/botchan.txt --model_prefix=m --vocab_size=1000");
  SentencePieceTrainer::Train(
      "--input=../data/botchan.txt --model_prefix=m --vocab_size=1000 "
      "--model_type=bpe");
  SentencePieceTrainer::Train(
      "--input=../data/botchan.txt --model_prefix=m --vocab_size=1000 "
      "--model_type=char");
  SentencePieceTrainer::Train(
      "--input=../data/botchan.txt --model_prefix=m --vocab_size=1000 "
      "--model_type=word");
}

TEST(SentencePieceTrainerTest, TrainWithCustomNormalizationRule) {
  SentencePieceTrainer::Train(
      "--input=../data/botchan.txt --model_prefix=m --vocab_size=1000 "
      "--normalization_rule_tsv=../data/nfkc.tsv");
}

TEST(SentencePieceTrainerTest, TrainTest) {
  TrainerSpec trainer_spec;
  trainer_spec.add_input("../data/botchan.txt");
  trainer_spec.set_model_prefix("m");
  trainer_spec.set_vocab_size(1000);
  SentencePieceTrainer::Train(trainer_spec);
}
}  // namespace
}  // namespace sentencepiece
