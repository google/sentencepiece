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

#include "unigram_model_trainer.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

// Space symbol
#define WS "\xe2\x96\x81"

std::string RunTrainer(
    const std::vector<std::string> &input, int size,
    const std::vector<std::string> &user_defined_symbols = {}) {
  test::ScopedTempFile input_scoped_file("input");
  test::ScopedTempFile model_scoped_file("model");
  const std::string input_file = input_scoped_file.filename();
  const std::string model_prefix = model_scoped_file.filename();
  {
    io::OutputBuffer output(input_file);
    for (const auto &line : input) {
      output.WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);
  trainer_spec.set_hard_vocab_limit(false);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  for (const auto &w : user_defined_symbols) {
    trainer_spec.add_user_defined_symbols(w);
  }

  Trainer trainer(trainer_spec, normalizer_spec);
  EXPECT_OK(trainer.Train());

  SentencePieceProcessor processor;
  EXPECT_OK(processor.Load(model_prefix + ".model"));

  const auto &model = processor.model_proto();
  std::vector<std::string> pieces;

  // remove <unk>, <s>, </s>
  for (int i = 3; i < model.pieces_size(); ++i) {
    pieces.emplace_back(model.pieces(i).piece());
  }

  return string_util::Join(pieces, " ");
}

TEST(BPETrainerTest, BasicTest) {
  EXPECT_EQ("abra r a c b d", RunTrainer({"abracadabra"}, 20));
  EXPECT_EQ("p e n a l i", RunTrainer({"pen", "pineapple", "apple"}, 20));
  EXPECT_EQ("l he h e o", RunTrainer({"hellohe"}, 20));
  EXPECT_EQ("app e p l n i " WS,
            RunTrainer({"pen", "pineapple", "apple"}, 20, {"app"}));
}

TEST(UnigramTrainerTest, TrainerModelTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  const TrainerModel model(trainer_spec, normalizer_spec);
  EXPECT_EQ(EncodeResult(), model.Encode("test"));
}

TEST(UnigramTrainerTest, EndToEndTest) {
  const test::ScopedTempFile sf("tmp_model");

  EXPECT_OK(SentencePieceTrainer::Train(
      std::string("--model_prefix=") + sf.filename() +
      " --input=../data/wagahaiwa_nekodearu.txt"
      " --vocab_size=8000"
      " --normalization_rule_name=identity"
      " --model_type=unigram"
      " --user_defined_symbols=<user>"  // Allows duplicated symbol
      " --control_symbols=<ctrl>"));

  SentencePieceProcessor sp;
  EXPECT_OK(sp.Load(std::string(sf.filename()) + ".model"));
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  const int uid = sp.PieceToId("<user>");
  EXPECT_TRUE(sp.IsControl(cid));
  EXPECT_FALSE(sp.IsUnknown(uid));

  std::vector<std::string> tok;

  EXPECT_OK(sp.Encode("", &tok));
  EXPECT_TRUE(tok.empty());

  EXPECT_OK(sp.Encode(
      "吾輩《わがはい》は猫である。名前はまだ無い。"
      "どこで生れたかとんと見当《けんとう》がつかぬ。"
      "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している"
      "。",
      &tok));
  EXPECT_EQ(WS
            " 吾輩 《 わが はい 》 は 猫 である 。 名前 はまだ 無い 。 "
            "どこ で 生 れた か とん と 見当 《 けん とう 》 が つか ぬ 。 "
            "何でも 薄 暗 い じめ じめ した 所で ニャーニャー "
            "泣 い ていた 事 だけは 記憶 している 。",
            string_util::Join(tok, " "));
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
