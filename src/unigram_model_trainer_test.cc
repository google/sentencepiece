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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(UnigramTrainerTest, TrainerModelTest) {
  const TrainerModel model;
  EXPECT_TRUE(model.status().ok());
  EXPECT_EQ(0, model.GetPieceSize());
}

TEST(UnigramTrainerTest, InvalidNodeTest) {
  Lattice lattice;
  lattice.SetSentence("ABC");
  EXPECT_EQ(lattice.Insert(0, 0), nullptr);
  EXPECT_EQ(lattice.Insert(0, -1), nullptr);
  EXPECT_EQ(lattice.Insert(-1, 1), nullptr);
  EXPECT_EQ(lattice.Insert(0, 4), nullptr);
  EXPECT_EQ(lattice.Insert(2, 2), nullptr);

  auto* n0 = lattice.Insert(0, 1);
  auto* n1 = lattice.Insert(1, 1);
  auto* n2 = lattice.Insert(2, 1);
  ASSERT_NE(n0, nullptr);
  ASSERT_NE(n1, nullptr);
  ASSERT_NE(n2, nullptr);
  n0->score = 1.0f;
  n1->score = 1.0f;
  n2->score = 1.0f;
  EXPECT_EQ(lattice.Viterbi().first.size(), 3);
}

TEST(UnigramTrainerTest, EmptySentenceTest) {
  Lattice lattice;
  lattice.SetSentence("");
  EXPECT_TRUE(lattice.Viterbi().first.empty());
}

namespace {
// Joins the surfaces of `nodes` with a single space.
std::string GetTokenized(const Lattice& lattice,
                         const std::vector<Lattice::Node*>& nodes) {
  std::vector<std::string> tokens;
  for (const auto* node : nodes) {
    const char* begin = lattice.surface(node->pos);
    const char* end = lattice.surface(node->pos + node->length);
    tokens.emplace_back(begin, end - begin);
  }
  return absl::StrJoin(tokens, " ");
}

void InsertWithScoreAndId(Lattice* lattice, int pos, int length, float score,
                          int id = -1) {
  auto* node = lattice->Insert(pos, length);
  ASSERT_NE(node, nullptr);
  node->score = score;
  node->id = id;
}
}  // namespace

TEST(UnigramTrainerTest, LatticeSetSentenceTest) {
  Lattice lattice;

  EXPECT_EQ(0, lattice.size());
  EXPECT_EQ(0, lattice.utf8_size());

  lattice.SetSentence("");
  EXPECT_EQ(0, lattice.size());
  EXPECT_EQ(0, lattice.utf8_size());
  EXPECT_STREQ("", lattice.sentence());
  EXPECT_STREQ("", lattice.surface(0));

  lattice.SetSentence("test");
  EXPECT_EQ(4, lattice.size());
  EXPECT_EQ(4, lattice.utf8_size());
  EXPECT_STREQ("test", lattice.sentence());
  EXPECT_STREQ("test", lattice.surface(0));
  EXPECT_STREQ("est", lattice.surface(1));
  EXPECT_STREQ("st", lattice.surface(2));
  EXPECT_STREQ("t", lattice.surface(3));

  // Only the EOS node starts at the end of the sentence.
  ASSERT_EQ(1, lattice.begin_nodes(4).size());
  EXPECT_EQ(-1, lattice.begin_nodes(4).front()->id);

  lattice.SetSentence("テストab");
  EXPECT_EQ(5, lattice.size());
  EXPECT_EQ(11, lattice.utf8_size());
  EXPECT_STREQ("テストab", lattice.sentence());
  EXPECT_STREQ("テストab", lattice.surface(0));
  EXPECT_STREQ("ストab", lattice.surface(1));
  EXPECT_STREQ("トab", lattice.surface(2));
  EXPECT_STREQ("ab", lattice.surface(3));
  EXPECT_STREQ("b", lattice.surface(4));

  lattice.Clear();
  EXPECT_EQ(0, lattice.size());
  EXPECT_EQ(0, lattice.utf8_size());
}

TEST(UnigramTrainerTest, LatticeInsertTest) {
  Lattice lattice;
  lattice.SetSentence("ABあい");

  Lattice::Node* node[7];
  node[0] = lattice.Insert(0, 1);
  node[1] = lattice.Insert(1, 1);
  node[2] = lattice.Insert(2, 1);
  node[3] = lattice.Insert(3, 1);
  node[4] = lattice.Insert(0, 2);
  node[5] = lattice.Insert(1, 2);
  node[6] = lattice.Insert(2, 2);
  for (const auto* n : node) ASSERT_NE(n, nullptr);

  EXPECT_EQ("A", GetTokenized(lattice, {node[0]}));
  EXPECT_EQ("B", GetTokenized(lattice, {node[1]}));
  EXPECT_EQ("あ", GetTokenized(lattice, {node[2]}));
  EXPECT_EQ("い", GetTokenized(lattice, {node[3]}));
  EXPECT_EQ("AB", GetTokenized(lattice, {node[4]}));
  EXPECT_EQ("Bあ", GetTokenized(lattice, {node[5]}));
  EXPECT_EQ("あい", GetTokenized(lattice, {node[6]}));

  const int expected_pos[] = {0, 1, 2, 3, 0, 1, 2};
  const int expected_length[] = {1, 1, 1, 1, 2, 2, 2};
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_pos[i], node[i]->pos) << i;
    EXPECT_EQ(expected_length[i], node[i]->length) << i;
    // BOS and EOS take the first two node ids.
    EXPECT_EQ(i + 2, node[i]->node_id) << i;
  }

  EXPECT_EQ(2, lattice.begin_nodes(0).size());
  EXPECT_EQ(2, lattice.begin_nodes(1).size());
  EXPECT_EQ(2, lattice.begin_nodes(2).size());
  EXPECT_EQ(1, lattice.begin_nodes(3).size());
  EXPECT_EQ(1, lattice.begin_nodes(4).size());  // EOS

  EXPECT_EQ(node[0], lattice.begin_nodes(0)[0]);
  EXPECT_EQ(node[4], lattice.begin_nodes(0)[1]);
  EXPECT_EQ(node[1], lattice.begin_nodes(1)[0]);
  EXPECT_EQ(node[5], lattice.begin_nodes(1)[1]);
  EXPECT_EQ(node[2], lattice.begin_nodes(2)[0]);
  EXPECT_EQ(node[6], lattice.begin_nodes(2)[1]);
  EXPECT_EQ(node[3], lattice.begin_nodes(3)[0]);
}

TEST(UnigramTrainerTest, LatticeViterbiFromIncompleteLatticeTest) {
  Lattice lattice;
  lattice.SetSentence("ABC");
  EXPECT_TRUE(lattice.Viterbi().first.empty());

  // Still incomplete.
  ASSERT_NE(lattice.Insert(0, 1), nullptr);
  EXPECT_TRUE(lattice.Viterbi().first.empty());

  ASSERT_NE(lattice.Insert(1, 1), nullptr);
  ASSERT_NE(lattice.Insert(2, 1), nullptr);
  EXPECT_EQ(3, lattice.Viterbi().first.size());
}

TEST(UnigramTrainerTest, LatticeViterbiTest) {
  Lattice lattice;
  lattice.SetSentence("ABC");

  InsertWithScoreAndId(&lattice, 0, 1, 0.0);  // A
  InsertWithScoreAndId(&lattice, 1, 1, 0.0);  // B
  InsertWithScoreAndId(&lattice, 2, 1, 0.0);  // C
  EXPECT_EQ("A B C", GetTokenized(lattice, lattice.Viterbi().first));

  InsertWithScoreAndId(&lattice, 0, 2, 2.0);  // AB
  EXPECT_EQ("AB C", GetTokenized(lattice, lattice.Viterbi().first));

  InsertWithScoreAndId(&lattice, 1, 2, 5.0);  // BC
  EXPECT_EQ("A BC", GetTokenized(lattice, lattice.Viterbi().first));

  InsertWithScoreAndId(&lattice, 0, 3, 10.0);  // ABC
  const auto best = lattice.Viterbi();
  EXPECT_EQ("ABC", GetTokenized(lattice, best.first));
  EXPECT_NEAR(10.0, best.second, 0.001);
}

TEST(UnigramTrainerTest, LatticePopulateMarginalTest) {
  Lattice lattice;
  lattice.SetSentence("ABC");

  InsertWithScoreAndId(&lattice, 0, 1, 1.0, 0);  // A
  InsertWithScoreAndId(&lattice, 1, 1, 1.2, 1);  // B
  InsertWithScoreAndId(&lattice, 2, 1, 2.5, 2);  // C
  InsertWithScoreAndId(&lattice, 0, 2, 3.0, 3);  // AB
  InsertWithScoreAndId(&lattice, 1, 2, 4.0, 4);  // BC
  InsertWithScoreAndId(&lattice, 0, 3, 2.0, 5);  // ABC

  std::vector<float> probs(6, 0.0);

  // Expand all paths:
  // A B C : exp(1.0 + 1.2 + 2.5) => path1
  // AB  C : exp(3.0 + 2.5)       => path2
  // A  BC : exp(1.0 + 4.0)       => path3
  // ABC   : exp(2.0)             => path4
  const float p1 = std::exp(1.0 + 1.2 + 2.5);
  const float p2 = std::exp(3.0 + 2.5);
  const float p3 = std::exp(1.0 + 4.0);
  const float p4 = std::exp(2.0);
  const float Z = p1 + p2 + p3 + p4;

  const float logZ = lattice.PopulateMarginal(1.0, &probs);

  EXPECT_NEAR((p1 + p3) / Z, probs[0], 0.001);  // A
  EXPECT_NEAR(p1 / Z, probs[1], 0.001);         // B
  EXPECT_NEAR((p1 + p2) / Z, probs[2], 0.001);  // C
  EXPECT_NEAR(p2 / Z, probs[3], 0.001);         // AB
  EXPECT_NEAR(p3 / Z, probs[4], 0.001);         // BC
  EXPECT_NEAR(p4 / Z, probs[5], 0.001);         // ABC
  EXPECT_NEAR(std::log(static_cast<double>(Z)), logZ, 0.001);
}

TEST(UnigramTrainerTest, PruneUnreachableSentencePiecesTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;

  TrainerModel model;
  // "a": -1.0, "b": -2.0
  // "ab": -4.0 (shadowed by "a" + "b" = -3.0 -> unreachable)
  // "c": -1.0, "d": -1.0
  // "cd": -1.5 (better than "c" + "d" = -2.0 -> reachable)
  TrainerModel::SentencePieces pieces = {{"a", -1.0f},  {"b", -2.0f},
                                         {"ab", -4.0f}, {"c", -1.0f},
                                         {"d", -1.0f},  {"cd", -1.5f}};
  EXPECT_TRUE(model.SetSentencePieces(std::move(pieces)).ok());

  Trainer trainer(trainer_spec, normalizer_spec);
  auto pruned = trainer.PruneUnreachableSentencePieces(model);

  std::vector<std::string> pruned_names;
  for (const auto& p : pruned) {
    pruned_names.push_back(p.first);
  }

  std::vector<std::string> expected = {"a", "b", "c", "d", "cd"};
  EXPECT_EQ(expected, pruned_names);
}

struct TrainerResult {
  std::string sentence_pieces;
  std::vector<std::pair<std::string, float>> seed_pieces_and_probs;
};

TrainerResult RunTrainer(const std::vector<std::string>& input, int size) {
  const std::string input_file =
      filesystem::JoinPath(::testing::TempDir(), "input");
  const std::string model_prefix =
      filesystem::JoinPath(::testing::TempDir(), "model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto& line : input) {
      output->WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  std::vector<std::pair<std::string, float>> seed_pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec);
    ABSL_EXPECT_OK(trainer.LoadSentences());
    TrainerModel::SentencePieces res = trainer.MakeSeedSentencePieces();

    for (const auto& piece : res) {
      seed_pieces.emplace_back(piece.first, piece.second);
    }
  }

  std::vector<std::string> pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec);
    EXPECT_TRUE(trainer.Train().ok());

    SentencePieceProcessor processor;
    EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

    const auto& model = processor.model_proto();

    // remove <unk>, <s>, </s>
    for (int i = 3; i < model.pieces_size(); ++i) {
      pieces.emplace_back(model.pieces(i).piece());
    }
  }

  TrainerResult res;
  res.seed_pieces_and_probs = seed_pieces;
  std::sort(pieces.begin(), pieces.end());
  res.sentence_pieces = absl::StrJoin(pieces, " ");
  return res;
}

TEST(UnigramTrainerTest, BasicTest) {
  const auto& res = RunTrainer(
      {"magnanimity \t 5", "Pineapple \t 6", "i have an apple and a pen \t 1",
       "Overly \t 6", "Available \t 3"},
      30);

  // Check seed pieces.
  EXPECT_EQ(53, res.seed_pieces_and_probs.size());

  // Check final pieces.
  EXPECT_EQ(
      "A Available O Overly P Pineapple a b d e g h i l m magnanimity n p r t "
      "v "
      "y ▁ ▁an",
      res.sentence_pieces);
}

namespace {

static constexpr char kTestInputData[] = "wagahaiwa_nekodearu.txt";

TEST(UnigramTrainerTest, EndToEndTest) {
  const std::string input =
      filesystem::JoinPath(::testing::SrcDir(), kTestInputData);

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=",
                       filesystem::JoinPath(::testing::TempDir(), "tmp_model"),
                       " --input=", input,
                       " --vocab_size=8000 --normalization_rule_name=identity",
                       " --model_type=unigram --user_defined_symbols=<user>",
                       " --control_symbols=<ctrl> --max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  EXPECT_TRUE(
      sp.Load(filesystem::JoinPath(::testing::TempDir(), "tmp_model.model"))
          .ok());
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  const int uid = sp.PieceToId("<user>");
  EXPECT_TRUE(sp.IsControl(cid));
  EXPECT_FALSE(sp.IsUnknown(uid));

  std::vector<std::string> tok;

  EXPECT_TRUE(sp.Encode("", &tok).ok());
  EXPECT_TRUE(tok.empty());

  EXPECT_TRUE(sp.Encode("吾輩《わがはい》は猫である。名前はまだ無い。"
                        "どこで生れたかとんと見当《けんとう》がつかぬ。"
                        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だ"
                        "けは記憶している"
                        "。",
                        &tok)
                  .ok());
  // TODO(taku): Temporally disable this test on Windows.
#if !defined(_WIN32) || defined(__CYGWIN__)
  LOG(INFO) << "[" << absl::StrJoin(tok, " ") << std::endl;
  EXPECT_EQ(
      WS
      " 吾輩 《 わが はい 》 は猫である 。 名前はまだ 無い 。 どこで 生れ たか "
      "とんと 見当 《 け ん とう 》 が つかぬ 。 何でも 薄 暗 いじめ じめ "
      "した 所で ニャーニャー 泣 いていた 事 だけは 記憶 している 。",
      absl::StrJoin(tok, " "));
#endif
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
