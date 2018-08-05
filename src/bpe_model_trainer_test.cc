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

#include "bpe_model_trainer.h"

#include <string>
#include <vector>
#include "flags.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "util.h"

DECLARE_string(data_dir);

namespace sentencepiece {
namespace bpe {
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
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);

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
  EXPECT_EQ("ab ra abra ad cad abracad abracadabra ac br a b r c d",
            RunTrainer({"abracadabra"}, 20));
  EXPECT_EQ("ap le app apple en in ine pen p e a l n i",
            RunTrainer({"pen", "pineapple", "apple"}, 20));
  EXPECT_EQ("he ll llo hello hellohe el lo oh hel ohe e h l o",
            RunTrainer({"hellohe"}, 20));
  EXPECT_EQ("app le en in ine pen " WS "le pine e l n p i " WS,
            RunTrainer({"pen", "pineapple", "apple"}, 20, {"app"}));
}

TEST(BPETrainerTest, EndToEndTest) {
  const test::ScopedTempFile sf("tmp_model");
  const std::string input =
      util::JoinPath(FLAGS_data_dir, "wagahaiwa_nekodearu.txt");

  EXPECT_OK(SentencePieceTrainer::Train(string_util::StrCat(
      "--model_prefix=", sf.filename(), " --input=", input,
      " --vocab_size=8000 --normalization_rule_name=identity"
      " --model_type=bpe --control_symbols=<ctrl>")));

  SentencePieceProcessor sp;
  EXPECT_OK(sp.Load(std::string(sf.filename()) + ".model"));
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  EXPECT_TRUE(sp.IsControl(cid));

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
            " 吾輩 《 わが はい 》 は猫 である 。 名前 はまだ 無い 。 "
            "どこで 生 れた か とん と見 当 《 けんとう 》 が つかぬ 。 "
            "何でも 薄 暗 いじ め じ め した 所で ニャー ニャー 泣 いていた "
            "事 だけは 記憶 している 。",
            string_util::Join(tok, " "));
}

}  // namespace
}  // namespace bpe
}  // namespace sentencepiece
