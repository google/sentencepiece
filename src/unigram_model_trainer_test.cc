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
#include "builder.h"
#include "normalizer.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(UnigramTrainerTest, TrainerModelTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  const TrainerModel model(trainer_spec, normalizer_spec);
  EXPECT_EQ(EncodeResult(), model.Encode("test"));
}

TEST(UnigramTrainerTest, EndToEndTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  normalizer_spec = normalizer::Builder::GetNormalizerSpec("nfkc");
  trainer_spec.add_input("../data/wagahaiwa_nekodearu.txt");

  constexpr int kVocabSize = 8000;
  trainer_spec.set_vocab_size(kVocabSize);
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);

  trainer_spec.add_control_symbols("<ctrl>");
  trainer_spec.add_user_defined_symbols("<user>");

  test::ScopedTempFile sf("tmp_model");
  trainer_spec.set_model_prefix(sf.filename());
  unigram::Trainer trainer(trainer_spec, normalizer_spec);
  EXPECT_OK(trainer.Train());

  SentencePieceProcessor sp;
  EXPECT_OK(sp.Load(std::string(sf.filename()) + ".model"));
  EXPECT_EQ(kVocabSize, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  const int uid = sp.PieceToId("<user>");
  EXPECT_TRUE(sp.IsControl(cid));
  EXPECT_FALSE(sp.IsUnknown(uid));

  std::vector<std::string> tok;

  sp.Encode("", &tok);
  EXPECT_TRUE(tok.empty());

  sp.Encode(
      "吾輩《わがはい》は猫である。名前はまだ無い。"
      "どこで生れたかとんと見当《けんとう》がつかぬ。"
      "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している"
      "。",
      &tok);
  EXPECT_EQ(WS
            " 吾輩 《 わが はい 》 は 猫 である 。 名前 はまだ 無い 。 "
            "どこ で 生 れた か とん と 見当 《 けん とう 》 が つか ぬ 。 "
            "何でも 薄 暗 い じめ じめ した 所で ニャーニャー "
            "泣 い ていた 事 だけは 記憶 している 。",
            string_util::Join(tok, " "));
}
}  // namespace unigram
}  // namespace sentencepiece
