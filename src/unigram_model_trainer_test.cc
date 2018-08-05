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

#include "flags.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "util.h"

DECLARE_string(data_dir);

namespace sentencepiece {
namespace unigram {
namespace {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(UnigramTrainerTest, TrainerModelTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  const TrainerModel model(trainer_spec, normalizer_spec);
  EXPECT_EQ(EncodeResult(), model.Encode("test"));
}

TEST(UnigramTrainerTest, EndToEndTest) {
  const test::ScopedTempFile sf("tmp_model");
  const std::string input =
      util::JoinPath(FLAGS_data_dir, "wagahaiwa_nekodearu.txt");

  EXPECT_OK(SentencePieceTrainer::Train(string_util::StrCat(
      "--model_prefix=", sf.filename(), " --input=", input,
      " --vocab_size=8000 --normalization_rule_name=identity",
      " --model_type=unigram --user_defined_symbols=<user>",
      " --control_symbols=<ctrl>")));

  SentencePieceProcessor sp;
  EXPECT_OK(sp.Load(string_util::StrCat(sf.filename(), ".model")));
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
  // TODO(taku): Temporally disable this test on Windows.
#ifndef OS_WIN
  EXPECT_EQ(WS
            " 吾輩 《 わが はい 》 は 猫 である 。 名前 はまだ 無い 。 "
            "どこ で 生 れた か とん と 見当 《 けん とう 》 が つか ぬ 。 "
            "何でも 薄 暗 い じめ じめ した 所で ニャーニャー "
            "泣 い ていた 事 だけは 記憶 している 。",
            string_util::Join(tok, " "));
#endif
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
