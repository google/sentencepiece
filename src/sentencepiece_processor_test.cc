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

#include "sentencepiece_processor.h"
#include <unordered_map>
#include "builder.h"
#include "model_interface.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "stringpiece.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
using port::MakeUnique;

// Space symbol
#define WS "\xe2\x96\x81"

class MockModel : public ModelInterface {
 public:
  void SetEncodeResult(StringPiece input,
                       const std::vector<std::pair<StringPiece, int>> &output) {
    input_ = input;
    output_ = output;
  }

  std::vector<std::pair<StringPiece, int>> Encode(
      StringPiece normalized) const {
    EXPECT_EQ(normalized, input_);
    return output_;
  }

  bool IsControl(int id) const { return id == 1 || id == 2; }

  bool IsUnknown(int id) const { return id == 0; }

  int GetPieceSize() const { return 10; }

  int PieceToId(StringPiece piece) const { return 0; }

  std::string IdToPiece(int id) const { return ""; }

  float GetScore(int id) const { return 0.0; }

 private:
  StringPiece input_;
  std::vector<std::pair<StringPiece, int>> output_;
};

std::vector<std::string> GetSpVec(
    const std::vector<std::pair<StringPiece, int>> &pieces) {
  std::vector<std::string> sps;
  for (const auto &p : pieces) {
    sps.emplace_back(p.first.to_string());
  }
  return sps;
}

std::vector<std::string> GetSpVec(const SentencePieceText &spt) {
  std::vector<std::string> sps;
  for (auto &sp : spt.pieces()) {
    sps.emplace_back(sp.piece());
  }
  return sps;
}

NormalizerSpec MakeDefaultNormalizerSpec() {
  return normalizer::Builder::GetNormalizerSpec("nfkc");
}

TEST(SentencepieceProcessorTest, EncodeTest) {
  const StringPiece kInput = WS "ABC" WS "DEF";
  SentencePieceProcessor sp;

  const auto normalization_spec = MakeDefaultNormalizerSpec();

  {
    auto mock = MakeUnique<MockModel>();

    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "ABC", 3}, {WS "DE", 4}, {"F", 0}, {"</s>", 2}};
    mock->SetEncodeResult(kInput, result);

    sp.SetModel(std::move(mock));
    sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalization_spec));

    std::vector<std::string> output;
    sp.Encode("ABC DEF", &output);
    EXPECT_EQ(GetSpVec(result), output);

    SentencePieceText spt;
    sp.Encode("ABC DEF", &spt);
    EXPECT_EQ(4, spt.pieces_size());
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(result[i].first, spt.pieces(i).piece());
    }

    EXPECT_EQ("ABC", spt.pieces(0).surface());
    EXPECT_EQ(" DE", spt.pieces(1).surface());
    EXPECT_EQ("F", spt.pieces(2).surface());
    EXPECT_EQ("", spt.pieces(3).surface());  // </s>

    EXPECT_EQ(3, spt.pieces(0).id());
    EXPECT_EQ(4, spt.pieces(1).id());
    EXPECT_EQ(0, spt.pieces(2).id());
    EXPECT_EQ(2, spt.pieces(3).id());

    EXPECT_EQ(0, spt.pieces(0).begin());
    EXPECT_EQ(3, spt.pieces(0).end());
    EXPECT_EQ(3, spt.pieces(1).begin());
    EXPECT_EQ(6, spt.pieces(1).end());
    EXPECT_EQ(6, spt.pieces(2).begin());
    EXPECT_EQ(7, spt.pieces(2).end());
    EXPECT_EQ(7, spt.pieces(3).begin());
    EXPECT_EQ(7, spt.pieces(3).end());
  }

  // Unknown sequences.
  {
    auto mock = MakeUnique<MockModel>();

    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "ABC", 3}, {WS "D", 4}, {"E", 0}, {"F", 0}, {"</s>", 2}};
    const std::vector<std::pair<StringPiece, int>> expected = {
        {WS "ABC", 3}, {WS "D", 4}, {"EF", 0}, {"</s>", 2}};

    mock->SetEncodeResult(kInput, result);
    sp.SetModel(std::move(mock));
    sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalization_spec));

    std::vector<std::string> output;
    sp.Encode("ABC DEF", &output);
    EXPECT_EQ(GetSpVec(expected), output);

    SentencePieceText spt;
    sp.Encode("ABC DEF", &spt);
    EXPECT_EQ(4, spt.pieces_size());
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(expected[i].first, spt.pieces(i).piece());
    }

    EXPECT_EQ("ABC", spt.pieces(0).surface());
    EXPECT_EQ(" D", spt.pieces(1).surface());
    EXPECT_EQ("EF", spt.pieces(2).surface());
    EXPECT_EQ("", spt.pieces(3).surface());  // </s>

    EXPECT_EQ(3, spt.pieces(0).id());
    EXPECT_EQ(4, spt.pieces(1).id());
    EXPECT_EQ(0, spt.pieces(2).id());
    EXPECT_EQ(2, spt.pieces(3).id());

    EXPECT_EQ(0, spt.pieces(0).begin());
    EXPECT_EQ(3, spt.pieces(0).end());
    EXPECT_EQ(3, spt.pieces(1).begin());
    EXPECT_EQ(5, spt.pieces(1).end());
    EXPECT_EQ(5, spt.pieces(2).begin());
    EXPECT_EQ(7, spt.pieces(2).end());
    EXPECT_EQ(7, spt.pieces(3).begin());
    EXPECT_EQ(7, spt.pieces(3).end());
  }

  // Crash if
  // ModelInterface::Encode() returns shorter results.
  {
    auto mock = MakeUnique<MockModel>();
    const std::vector<std::pair<StringPiece, int>> result = {{WS "ABC", 3}};
    mock->SetEncodeResult(kInput, result);
    sp.SetModel(std::move(mock));
    sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalization_spec));
    SentencePieceText spt;
    // Expects crash.
    EXPECT_DEATH(sp.Encode("ABC DEF", &spt));
  }

  // Crash if
  // ModelInterface::Encode() returns longer results.
  {
    auto mock = MakeUnique<MockModel>();
    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "ABC", 3}, {WS "DE", 4}, {"F", 5}, {"G", 6}};
    mock->SetEncodeResult(kInput, result);
    sp.SetModel(std::move(mock));
    sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalization_spec));
    SentencePieceText spt;
    // Expects crash.
    EXPECT_DEATH(sp.Encode("ABC DEF", &spt));
  }

  // Crash if
  // ModelInterface::Encode() returns an empty piece.
  {
    auto mock = MakeUnique<MockModel>();
    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "ABC", 3}, {WS "DE", 4}, {"", 5}, {"F", 6}};
    mock->SetEncodeResult(kInput, result);
    sp.SetModel(std::move(mock));
    sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalization_spec));
    SentencePieceText spt;
    // Expects crash.
    EXPECT_DEATH(sp.Encode("ABC DEF", &spt));
  }

  // Halfwidth to Fullwidith katakana normalization.
  {
    auto mock = MakeUnique<MockModel>();
    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "グー", 3}, {"グル", 4}, {"</s>", 2}};
    const StringPiece input = WS "グーグル";
    mock->SetEncodeResult(input, result);
    sp.SetModel(std::move(mock));
    std::vector<std::string> output;
    sp.Encode("ｸﾞｰｸﾞﾙ", &output);
    EXPECT_EQ(GetSpVec(result), output);

    SentencePieceText spt;
    sp.Encode("ｸﾞｰｸﾞﾙ", &spt);
    EXPECT_EQ(3, spt.pieces_size());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result[i].first, spt.pieces(i).piece());
    }

    EXPECT_EQ("ｸﾞｰ", spt.pieces(0).surface());
    EXPECT_EQ("ｸﾞﾙ", spt.pieces(1).surface());
    EXPECT_EQ("", spt.pieces(2).surface());

    EXPECT_EQ(3, spt.pieces(0).id());
    EXPECT_EQ(4, spt.pieces(1).id());
    EXPECT_EQ(2, spt.pieces(2).id());

    EXPECT_EQ(0, spt.pieces(0).begin());
    EXPECT_EQ(9, spt.pieces(0).end());
    EXPECT_EQ(9, spt.pieces(1).begin());
    EXPECT_EQ(18, spt.pieces(1).end());
    EXPECT_EQ(18, spt.pieces(2).begin());  // </s>
    EXPECT_EQ(18, spt.pieces(2).end());
  }

  // One to many normalization.
  {
    auto mock = MakeUnique<MockModel>();
    const std::vector<std::pair<StringPiece, int>> result = {
        {WS "株式", 3}, {"会社", 4}, {"</s>", 2}};
    const StringPiece input = WS "株式会社";
    mock->SetEncodeResult(input, result);
    sp.SetModel(std::move(mock));
    std::vector<std::string> output;
    sp.Encode("㍿", &output);
    EXPECT_EQ(GetSpVec(result), output);

    SentencePieceText spt;
    sp.Encode("㍿", &spt);
    EXPECT_EQ(3, spt.pieces_size());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(result[i].first, spt.pieces(i).piece());
    }

    EXPECT_EQ("", spt.pieces(0).surface());
    EXPECT_EQ("㍿", spt.pieces(1).surface());
    EXPECT_EQ("", spt.pieces(2).surface());

    EXPECT_EQ(3, spt.pieces(0).id());
    EXPECT_EQ(4, spt.pieces(1).id());
    EXPECT_EQ(2, spt.pieces(2).id());

    EXPECT_EQ(0, spt.pieces(0).begin());  // 株式
    EXPECT_EQ(0, spt.pieces(0).end());
    EXPECT_EQ(0, spt.pieces(1).begin());  // 会社
    EXPECT_EQ(3, spt.pieces(1).end());
    EXPECT_EQ(3, spt.pieces(2).begin());  // </s>
    EXPECT_EQ(3, spt.pieces(2).end());
  }
}

TEST(SentencepieceProcessorTest, DecodeTest) {
  class DecodeMockModel : public ModelInterface {
   public:
    std::vector<std::pair<StringPiece, int>> Encode(
        StringPiece normalized) const override {
      return {};
    }

    int GetPieceSize() const override { return 7; }

    int PieceToId(StringPiece piece) const override {
      static std::unordered_map<StringPiece, int, StringPieceHash> kMap = {
          {"<unk>", 0}, {"<s>", 1}, {"</s>", 2},    {WS "ABC", 3},
          {WS "DE", 4}, {"F", 5},   {"G" WS "H", 6}};
      return port::FindWithDefault(kMap, piece, 0);
    }

    std::string IdToPiece(int id) const override {
      static std::vector<std::string> kMap = {
          "<unk>", "<s>", "</s>", WS "ABC", WS "DE", "F", "G" WS "H"};
      return kMap[id];
    }

    bool IsUnknown(int id) const override { return (id == 0); }

    bool IsControl(int id) const override { return (id == 1 || id == 2); }

    float GetScore(int id) const override { return 0.0; }
  };

  SentencePieceProcessor sp;
  auto mock = MakeUnique<DecodeMockModel>();
  //  std::unique_ptr<ModelInterface> mock(new DecodeMockModel);
  sp.SetModel(std::move(mock));

  const auto normalizaiton_spec = MakeDefaultNormalizerSpec();
  sp.SetNormalizer(MakeUnique<normalizer::Normalizer>(normalizaiton_spec));

  const std::vector<std::string> input = {"<s>", WS "ABC",   "<unk>", WS "DE",
                                          "F",   "G" WS "H", "I",     "</s>"};
  SentencePieceText spt;

  sp.Decode(input, &spt);
  EXPECT_EQ("ABC \xE2\x81\x87  DEFG HI", spt.text());
  EXPECT_EQ(8, spt.pieces_size());

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(input[i], spt.pieces(i).piece());
  }

  EXPECT_EQ("", spt.pieces(0).surface());
  EXPECT_EQ("ABC", spt.pieces(1).surface());
  EXPECT_EQ(" \xE2\x81\x87 ", spt.pieces(2).surface());
  EXPECT_EQ(" DE", spt.pieces(3).surface());
  EXPECT_EQ("F", spt.pieces(4).surface());
  EXPECT_EQ("G H", spt.pieces(5).surface());
  EXPECT_EQ("I", spt.pieces(6).surface());
  EXPECT_EQ("", spt.pieces(7).surface());

  EXPECT_EQ(0, spt.pieces(0).begin());
  EXPECT_EQ(0, spt.pieces(0).end());
  EXPECT_EQ(0, spt.pieces(1).begin());
  EXPECT_EQ(3, spt.pieces(1).end());
  EXPECT_EQ(3, spt.pieces(2).begin());
  EXPECT_EQ(8, spt.pieces(2).end());
  EXPECT_EQ(8, spt.pieces(3).begin());
  EXPECT_EQ(11, spt.pieces(3).end());
  EXPECT_EQ(11, spt.pieces(4).begin());
  EXPECT_EQ(12, spt.pieces(4).end());
  EXPECT_EQ(12, spt.pieces(5).begin());
  EXPECT_EQ(15, spt.pieces(5).end());
  EXPECT_EQ(15, spt.pieces(6).begin());
  EXPECT_EQ(16, spt.pieces(6).end());
  EXPECT_EQ(16, spt.pieces(7).begin());
  EXPECT_EQ(16, spt.pieces(7).end());
}

void AddPiece(ModelProto *model_proto, StringPiece piece, float score = 0.0) {
  auto *sp = model_proto->add_pieces();
  sp->set_piece(piece.to_string());
  sp->set_score(score);
}

TEST(SentencePieceProcessorTest, LoadInvalidModelTest) {
  SentencePieceProcessor sp;
  EXPECT_DEATH(sp.LoadOrDie(""));
  EXPECT_DEATH(sp.LoadOrDie("__UNKNOWN_FILE__"));
  std::istringstream ss("__UNKNOWN_STREAM__");
  EXPECT_DEATH(sp.LoadOrDie(&ss));
}

TEST(SentencePieceProcessorTest, EndToEndTest) {
  ModelProto model_proto;
  auto *sp1 = model_proto.add_pieces();
  auto *sp2 = model_proto.add_pieces();
  auto *sp3 = model_proto.add_pieces();

  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("<unk>");
  sp2->set_type(ModelProto::SentencePiece::CONTROL);
  sp2->set_piece("<s>");
  sp3->set_type(ModelProto::SentencePiece::CONTROL);
  sp3->set_piece("</s>");

  AddPiece(&model_proto, "a", 0.0);
  AddPiece(&model_proto, "b", 0.3);
  AddPiece(&model_proto, "c", 0.2);
  AddPiece(&model_proto, "ab", 1.0);
  AddPiece(&model_proto, "\xE2\x96\x81", 3.0);  // kSpaceSymbol

  *(model_proto.mutable_normalizer_spec()) = MakeDefaultNormalizerSpec();

  test::ScopedTempFile sf("model");

  {
    std::ofstream ofs(sf.filename(), OUTPUT_MODE);
    CHECK(model_proto.SerializeToOstream(&ofs));
  }

  SentencePieceProcessor sp;
  sp.Load(sf.filename());

  EXPECT_EQ(model_proto.SerializeAsString(),
            sp.model_proto().SerializeAsString());

  EXPECT_EQ(8, sp.GetPieceSize());
  EXPECT_EQ(0, sp.PieceToId("<unk>"));
  EXPECT_EQ(1, sp.PieceToId("<s>"));
  EXPECT_EQ(2, sp.PieceToId("</s>"));
  EXPECT_EQ(3, sp.PieceToId("a"));
  EXPECT_EQ(4, sp.PieceToId("b"));
  EXPECT_EQ(5, sp.PieceToId("c"));
  EXPECT_EQ(6, sp.PieceToId("ab"));
  EXPECT_EQ(7, sp.PieceToId("\xE2\x96\x81"));

  EXPECT_EQ("<unk>", sp.IdToPiece(0));
  EXPECT_EQ("<s>", sp.IdToPiece(1));
  EXPECT_EQ("</s>", sp.IdToPiece(2));
  EXPECT_EQ("a", sp.IdToPiece(3));
  EXPECT_EQ("b", sp.IdToPiece(4));
  EXPECT_EQ("c", sp.IdToPiece(5));
  EXPECT_EQ("ab", sp.IdToPiece(6));
  EXPECT_EQ("\xE2\x96\x81", sp.IdToPiece(7));

  EXPECT_NEAR(0.0, sp.GetScore(0), 0.001);
  EXPECT_NEAR(0.0, sp.GetScore(1), 0.001);
  EXPECT_NEAR(0.0, sp.GetScore(2), 0.001);
  EXPECT_NEAR(0.0, sp.GetScore(3), 0.001);
  EXPECT_NEAR(0.3, sp.GetScore(4), 0.001);
  EXPECT_NEAR(0.2, sp.GetScore(5), 0.001);
  EXPECT_NEAR(1.0, sp.GetScore(6), 0.001);
  EXPECT_NEAR(3.0, sp.GetScore(7), 0.001);

  EXPECT_TRUE(sp.IsUnknown(0));
  EXPECT_FALSE(sp.IsUnknown(1));
  EXPECT_FALSE(sp.IsUnknown(2));
  EXPECT_FALSE(sp.IsUnknown(3));
  EXPECT_FALSE(sp.IsUnknown(4));
  EXPECT_FALSE(sp.IsUnknown(5));
  EXPECT_FALSE(sp.IsUnknown(6));
  EXPECT_FALSE(sp.IsUnknown(7));

  EXPECT_FALSE(sp.IsControl(0));
  EXPECT_TRUE(sp.IsControl(1));
  EXPECT_TRUE(sp.IsControl(2));
  EXPECT_FALSE(sp.IsControl(3));
  EXPECT_FALSE(sp.IsControl(4));
  EXPECT_FALSE(sp.IsControl(5));
  EXPECT_FALSE(sp.IsControl(6));
  EXPECT_FALSE(sp.IsControl(7));

  {
    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {WS, "ab", "c"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {7, 6, 5};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("bos");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", WS, "ab", "c"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 7, 6, 5};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("eos");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {WS, "ab", "c", "</s>"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {7, 6, 5, 2};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("reverse");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"c", "ab", WS};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {5, 6, 7};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("bos:eos");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", WS, "ab", "c",
                                                   "</s>"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 7, 6, 5, 2};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("reverse:bos:eos");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", "c", "ab", WS,
                                                   "</s>"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 5, 6, 7, 2};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    sp.SetEncodeExtraOptions("bos:eos:reverse");

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"</s>", "c", "ab", WS,
                                                   "<s>"};
    sp.Encode("abc", &sps);
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {2, 5, 6, 7, 1};
    sp.Encode("abc", &ids);
    EXPECT_EQ(expected_id, ids);
  }

  {
    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("abc", output);
  }

  {
    sp.SetDecodeExtraOptions("bos");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("abc", output);
  }

  {
    sp.SetDecodeExtraOptions("eos");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("abc", output);
  }

  {
    sp.SetDecodeExtraOptions("reverse");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("cba", output);
  }

  {
    sp.SetDecodeExtraOptions("bos:eos");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("abc", output);
  }

  {
    sp.SetDecodeExtraOptions("reverse:bos:eos");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("cba", output);
  }

  {
    sp.SetDecodeExtraOptions("bos:eos:reverse");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("cba", output);
  }

  {
    sp.SetDecodeExtraOptions("reverse:reverse");

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    sp.Decode(sps, &output);
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    sp.Decode(ids, &output);
    EXPECT_EQ("abc", output);
  }

  EXPECT_DEATH(sp.SetEncodeExtraOptions("foo"));
  EXPECT_DEATH(sp.SetDecodeExtraOptions("foo"));
}
}  // namespace sentencepiece
