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
#include "flags.h"
#include "sentencepiece_model.pb.h"
#include "testharness.h"
#include "util.h"

DECLARE_string(data_dir);

namespace sentencepiece {
namespace {

TEST(SentencePieceTrainerTest, TrainFromArgsTest) {
  std::string input = util::JoinPath(FLAGS_data_dir, "botchan.txt");
  SentencePieceTrainer::Train(string_util::StrCat(
      "--input=", input, " --model_prefix=m --vocab_size=1000"));
  SentencePieceTrainer::Train(string_util::StrCat(
      "--input=", input, " --model_prefix=m --vocab_size=1000 ",
      "--model_type=bpe"));
  SentencePieceTrainer::Train(string_util::StrCat(
      "--input=", input, " --model_prefix=m --vocab_size=1000 ",
      "--model_type=char"));
  SentencePieceTrainer::Train(string_util::StrCat(
      "--input=", input, " --model_prefix=m --vocab_size=1000 ",
      "--model_type=word"));
}

TEST(SentencePieceTrainerTest, TrainWithCustomNormalizationRule) {
  std::string input = util::JoinPath(FLAGS_data_dir, "botchan.txt");
  std::string rule = util::JoinPath(FLAGS_data_dir, "nfkc.tsv");
  SentencePieceTrainer::Train(string_util::StrCat(
      "--input=", input, " --model_prefix=m --vocab_size=1000 ",
      "--normalization_rule_tsv=", rule));
}

TEST(SentencePieceTrainerTest, TrainErrorTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  normalizer_spec.set_normalization_rule_tsv("foo.tsv");
  normalizer_spec.set_precompiled_charsmap("foo");
  EXPECT_NOT_OK(SentencePieceTrainer::Train(trainer_spec, normalizer_spec));
}

TEST(SentencePieceTrainerTest, TrainTest) {
  TrainerSpec trainer_spec;
  trainer_spec.add_input(util::JoinPath(FLAGS_data_dir, "botchan.txt"));
  trainer_spec.set_model_prefix("m");
  trainer_spec.set_vocab_size(1000);
  NormalizerSpec normalizer_spec;
  EXPECT_OK(SentencePieceTrainer::Train(trainer_spec, normalizer_spec));
  EXPECT_OK(SentencePieceTrainer::Train(trainer_spec));
}

TEST(SentencePieceTrainerTest, SetProtoFieldTest) {
  TrainerSpec spec;

  EXPECT_NOT_OK(SentencePieceTrainer::SetProtoField("dummy", "1000", &spec));

  EXPECT_OK(SentencePieceTrainer::SetProtoField("vocab_size", "1000", &spec));
  EXPECT_EQ(1000, spec.vocab_size());
  EXPECT_NOT_OK(
      SentencePieceTrainer::SetProtoField("vocab_size", "UNK", &spec));

  EXPECT_OK(SentencePieceTrainer::SetProtoField("input_format", "TSV", &spec));
  EXPECT_EQ("TSV", spec.input_format());
  EXPECT_OK(SentencePieceTrainer::SetProtoField("input_format", "123", &spec));
  EXPECT_EQ("123", spec.input_format());

  EXPECT_OK(SentencePieceTrainer::SetProtoField("split_by_whitespace", "false",
                                                &spec));
  EXPECT_FALSE(spec.split_by_whitespace());
  EXPECT_OK(
      SentencePieceTrainer::SetProtoField("split_by_whitespace", "", &spec));
  EXPECT_TRUE(spec.split_by_whitespace());

  EXPECT_OK(
      SentencePieceTrainer::SetProtoField("character_coverage", "0.5", &spec));
  EXPECT_NEAR(spec.character_coverage(), 0.5, 0.001);
  EXPECT_NOT_OK(
      SentencePieceTrainer::SetProtoField("character_coverage", "UNK", &spec));

  EXPECT_OK(SentencePieceTrainer::SetProtoField("input", "foo,bar,buz", &spec));
  EXPECT_EQ(3, spec.input_size());
  EXPECT_EQ("foo", spec.input(0));
  EXPECT_EQ("bar", spec.input(1));
  EXPECT_EQ("buz", spec.input(2));

  EXPECT_OK(SentencePieceTrainer::SetProtoField("model_type", "BPE", &spec));
  EXPECT_NOT_OK(
      SentencePieceTrainer::SetProtoField("model_type", "UNK", &spec));

  // Nested message is not supported.
  ModelProto proto;
  EXPECT_NOT_OK(
      SentencePieceTrainer::SetProtoField("trainer_spec", "UNK", &proto));
}

TEST(SentencePieceTrainerTest, MergeSpecsFromArgs) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  EXPECT_NOT_OK(SentencePieceTrainer::MergeSpecsFromArgs("", nullptr, nullptr));

  EXPECT_OK(SentencePieceTrainer::MergeSpecsFromArgs("", &trainer_spec,
                                                     &normalizer_spec));

  EXPECT_NOT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--unknown=BPE", &trainer_spec, &normalizer_spec));

  EXPECT_NOT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--vocab_size=UNK", &trainer_spec, &normalizer_spec));

  EXPECT_NOT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--model_type=UNK", &trainer_spec, &normalizer_spec));

  EXPECT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--model_type=bpe", &trainer_spec, &normalizer_spec));

  EXPECT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--split_by_whitespace", &trainer_spec, &normalizer_spec));

  EXPECT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--normalization_rule_name=foo", &trainer_spec, &normalizer_spec));
  EXPECT_EQ("foo", normalizer_spec.name());

  EXPECT_NOT_OK(SentencePieceTrainer::MergeSpecsFromArgs(
      "--vocab_size=UNK", &trainer_spec, &normalizer_spec));
}

}  // namespace
}  // namespace sentencepiece
