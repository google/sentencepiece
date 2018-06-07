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

#include "model_interface.h"
#include "model_factory.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
namespace {

#define WS "\xe2\x96\x81"

const std::vector<TrainerSpec::ModelType> kModelTypes = {
    TrainerSpec::UNIGRAM, TrainerSpec::BPE, TrainerSpec::WORD,
    TrainerSpec::CHAR};

ModelProto MakeBaseModelProto(TrainerSpec::ModelType type) {
  ModelProto model_proto;
  auto *sp1 = model_proto.add_pieces();
  auto *sp2 = model_proto.add_pieces();
  auto *sp3 = model_proto.add_pieces();
  model_proto.mutable_trainer_spec()->set_model_type(type);

  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("<unk>");
  sp2->set_type(ModelProto::SentencePiece::CONTROL);
  sp2->set_piece("<s>");
  sp3->set_type(ModelProto::SentencePiece::CONTROL);
  sp3->set_piece("</s>");

  return model_proto;
}

void AddPiece(ModelProto *model_proto, const std::string &piece,
              float score = 0.0) {
  auto *sp = model_proto->add_pieces();
  sp->set_piece(piece);
  sp->set_score(score);
}

TEST(ModelInterfaceTest, SetModelInterfaceTest) {
  for (const auto type : kModelTypes) {
    ModelProto model_proto = MakeBaseModelProto(type);
    AddPiece(&model_proto, "a");
    AddPiece(&model_proto, "b");
    AddPiece(&model_proto, "c");
    AddPiece(&model_proto, "d");

    auto model = ModelFactory::Create(model_proto);
    EXPECT_EQ(model_proto.SerializeAsString(),
              model->model_proto().SerializeAsString());
  }
}

TEST(ModelInterfaceTest, PieceToIdTest) {
  for (const auto type : kModelTypes) {
    ModelProto model_proto = MakeBaseModelProto(type);

    AddPiece(&model_proto, "a", 0.1);  // 3
    AddPiece(&model_proto, "b", 0.2);  // 4
    AddPiece(&model_proto, "c", 0.3);  // 5
    AddPiece(&model_proto, "d", 0.4);  // 6
    AddPiece(&model_proto, "e", 0.5);  // 7
    model_proto.mutable_pieces(6)->set_type(ModelProto::SentencePiece::UNUSED);
    model_proto.mutable_pieces(7)->set_type(
        ModelProto::SentencePiece::USER_DEFINED);

    auto model = ModelFactory::Create(model_proto);

    EXPECT_EQ(model_proto.SerializeAsString(),
              model->model_proto().SerializeAsString());

    EXPECT_EQ(0, model->PieceToId("<unk>"));
    EXPECT_EQ(1, model->PieceToId("<s>"));
    EXPECT_EQ(2, model->PieceToId("</s>"));
    EXPECT_EQ(3, model->PieceToId("a"));
    EXPECT_EQ(4, model->PieceToId("b"));
    EXPECT_EQ(5, model->PieceToId("c"));
    EXPECT_EQ(6, model->PieceToId("d"));
    EXPECT_EQ(7, model->PieceToId("e"));
    EXPECT_EQ(0, model->PieceToId("f"));  // unk
    EXPECT_EQ(0, model->PieceToId(""));   // unk

    EXPECT_EQ("<unk>", model->IdToPiece(0));
    EXPECT_EQ("<s>", model->IdToPiece(1));
    EXPECT_EQ("</s>", model->IdToPiece(2));
    EXPECT_EQ("a", model->IdToPiece(3));
    EXPECT_EQ("b", model->IdToPiece(4));
    EXPECT_EQ("c", model->IdToPiece(5));
    EXPECT_EQ("d", model->IdToPiece(6));
    EXPECT_EQ("e", model->IdToPiece(7));

    EXPECT_TRUE(model->IsUnknown(0));
    EXPECT_FALSE(model->IsUnknown(1));
    EXPECT_FALSE(model->IsUnknown(2));
    EXPECT_FALSE(model->IsUnknown(3));
    EXPECT_FALSE(model->IsUnknown(4));
    EXPECT_FALSE(model->IsUnknown(5));
    EXPECT_FALSE(model->IsUnknown(6));
    EXPECT_FALSE(model->IsUnknown(7));

    EXPECT_FALSE(model->IsControl(0));
    EXPECT_TRUE(model->IsControl(1));
    EXPECT_TRUE(model->IsControl(2));
    EXPECT_FALSE(model->IsControl(3));
    EXPECT_FALSE(model->IsControl(4));
    EXPECT_FALSE(model->IsControl(5));
    EXPECT_FALSE(model->IsControl(6));
    EXPECT_FALSE(model->IsControl(7));

    EXPECT_FALSE(model->IsUnused(0));
    EXPECT_FALSE(model->IsUnused(1));
    EXPECT_FALSE(model->IsUnused(2));
    EXPECT_FALSE(model->IsUnused(3));
    EXPECT_FALSE(model->IsUnused(4));
    EXPECT_FALSE(model->IsUnused(5));
    EXPECT_TRUE(model->IsUnused(6));
    EXPECT_FALSE(model->IsUnused(7));

    EXPECT_FALSE(model->IsUserDefined(0));
    EXPECT_FALSE(model->IsUserDefined(1));
    EXPECT_FALSE(model->IsUserDefined(2));
    EXPECT_FALSE(model->IsUserDefined(3));
    EXPECT_FALSE(model->IsUserDefined(4));
    EXPECT_FALSE(model->IsUserDefined(5));
    EXPECT_FALSE(model->IsUserDefined(6));
    EXPECT_TRUE(model->IsUserDefined(7));

    EXPECT_NEAR(0, model->GetScore(0), 0.0001);
    EXPECT_NEAR(0, model->GetScore(1), 0.0001);
    EXPECT_NEAR(0, model->GetScore(2), 0.0001);
    EXPECT_NEAR(0.1, model->GetScore(3), 0.0001);
    EXPECT_NEAR(0.2, model->GetScore(4), 0.0001);
    EXPECT_NEAR(0.3, model->GetScore(5), 0.0001);
    EXPECT_NEAR(0.4, model->GetScore(6), 0.0001);
    EXPECT_NEAR(0.5, model->GetScore(7), 0.0001);
  }
}

TEST(ModelInterfaceTest, InvalidModelTest) {
  // Empty piece.
  {
    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::UNIGRAM);
    AddPiece(&model_proto, "");
    auto model = ModelFactory::Create(model_proto);
    EXPECT_FALSE(model->status().ok());
  }

  // Duplicated pieces.
  {
    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::UNIGRAM);
    AddPiece(&model_proto, "a");
    AddPiece(&model_proto, "a");
    auto model = ModelFactory::Create(model_proto);
    EXPECT_FALSE(model->status().ok());
  }

  // Multiple unknowns.
  {
    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::UNIGRAM);
    model_proto.mutable_pieces(1)->set_type(ModelProto::SentencePiece::UNKNOWN);
    auto model = ModelFactory::Create(model_proto);
    EXPECT_FALSE(model->status().ok());
  }

  // No unknown.
  {
    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::UNIGRAM);
    model_proto.mutable_pieces(0)->set_type(ModelProto::SentencePiece::CONTROL);
    auto model = ModelFactory::Create(model_proto);
    EXPECT_FALSE(model->status().ok());
  }
}

std::string RandomString(int length) {
  const char kAlphaNum[] =
      "0123456789"
      "!@#$%^&*"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  const int kAlphaSize = sizeof(kAlphaNum) - 1;
  const int size = rand() % length + 1;
  std::string result;
  for (int i = 0; i < size; ++i) {
    result += kAlphaNum[rand() % kAlphaSize];
  }
  return result;
}

TEST(ModelInterfaceTest, PieceToIdStressTest) {
  for (const auto type : kModelTypes) {
    for (int i = 0; i < 100; ++i) {
      std::unordered_map<std::string, int> expected_p2i;
      std::unordered_map<int, std::string> expected_i2p;
      ModelProto model_proto = MakeBaseModelProto(type);
      for (int n = 0; n < 1000; ++n) {
        const std::string piece = RandomString(10);
        if (expected_p2i.find(piece) != expected_p2i.end()) {
          continue;
        }
        expected_p2i[piece] = model_proto.pieces_size();
        expected_i2p[model_proto.pieces_size()] = piece;
        AddPiece(&model_proto, piece);
      }

      auto model = ModelFactory::Create(model_proto);
      for (const auto &it : expected_p2i) {
        EXPECT_EQ(it.second, model->PieceToId(it.first));
      }
      for (const auto &it : expected_i2p) {
        EXPECT_EQ(it.second, model->IdToPiece(it.first));
      }
    }
  }
}

TEST(ModelInterfaceTest, SplitIntoWordsTest) {
  {
    const auto v = SplitIntoWords(WS "this" WS "is" WS "a" WS "pen");
    EXPECT_EQ(4, v.size());
    EXPECT_EQ(WS "this", v[0]);
    EXPECT_EQ(WS "is", v[1]);
    EXPECT_EQ(WS "a", v[2]);
    EXPECT_EQ(WS "pen", v[3]);
  }

  {
    const auto v = SplitIntoWords("this" WS "is" WS "a" WS "pen");
    EXPECT_EQ(4, v.size());
    EXPECT_EQ("this", v[0]);
    EXPECT_EQ(WS "is", v[1]);
    EXPECT_EQ(WS "a", v[2]);
    EXPECT_EQ(WS "pen", v[3]);
  }

  {
    const auto v = SplitIntoWords(WS "this" WS WS "is");
    EXPECT_EQ(3, v.size());
    EXPECT_EQ(WS "this", v[0]);
    EXPECT_EQ(WS, v[1]);
    EXPECT_EQ(WS "is", v[2]);
  }

  {
    const auto v = SplitIntoWords("");
    EXPECT_TRUE(v.empty());
  }

  {
    const auto v = SplitIntoWords("hello");
    EXPECT_EQ(1, v.size());
    EXPECT_EQ("hello", v[0]);
  }
}

TEST(ModelInterfaceTest, PrefixMatcherTest) {
  {
    const PrefixMatcher matcher({"abc", "ab", "xy", "京都"});
    bool found;
    EXPECT_EQ(1, matcher.PrefixMatch("test", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(3, matcher.PrefixMatch("abcd", &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(2, matcher.PrefixMatch("abxy", &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(1, matcher.PrefixMatch("x", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(2, matcher.PrefixMatch("xyz", &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(6, matcher.PrefixMatch("京都大学", &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(3, matcher.PrefixMatch("東京大学", &found));
    EXPECT_FALSE(found);
  }

  {
    const PrefixMatcher matcher({});
    bool found;
    EXPECT_EQ(1, matcher.PrefixMatch("test", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(1, matcher.PrefixMatch("abcd", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(1, matcher.PrefixMatch("abxy", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(1, matcher.PrefixMatch("x", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(1, matcher.PrefixMatch("xyz", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(3, matcher.PrefixMatch("京都大学", &found));
    EXPECT_FALSE(found);
    EXPECT_EQ(3, matcher.PrefixMatch("東京大学", &found));
    EXPECT_FALSE(found);
  }
}

}  // namespace
}  // namespace sentencepiece
