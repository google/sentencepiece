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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "builder.h"
#include "filesystem.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"
#include "util.h"

namespace sentencepiece {

// Space symbol
#define WS "\xe2\x96\x81"

using EncodeResult = std::vector<std::pair<absl::string_view, int>>;

std::vector<std::string> GetSpVec(const EncodeResult& pieces) {
  std::vector<std::string> sps;
  for (const auto& p : pieces) {
    sps.emplace_back(std::string(p.first));
  }
  return sps;
}

std::vector<int> GetIdVec(const EncodeResult& pieces) {
  std::vector<int> ids;
  for (const auto& p : pieces) {
    ids.emplace_back(p.second);
  }
  return ids;
}

std::vector<std::string> GetSpVec(const SentencePieceText& spt) {
  std::vector<std::string> sps;
  for (auto& sp : spt.pieces()) {
    sps.emplace_back(sp.piece());
  }
  return sps;
}

NormalizerSpec MakeDefaultNormalizerSpec() {
  return SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc");
}

TEST(SentencepieceProcessorTest, StatusTest) {
  SentencePieceProcessor sp;
  EXPECT_FALSE(sp.status().ok());
}

TEST(SentencepieceProcessorTest, EncodeTest) {
  const auto normalization_spec = MakeDefaultNormalizerSpec();

  auto MakeBaseProto = [&]() {
    ModelProto model_proto;
    *model_proto.mutable_normalizer_spec() = normalization_spec;
    auto* unk = model_proto.add_pieces();
    unk->set_piece("<unk>");
    unk->set_type(ModelProto::SentencePiece::UNKNOWN);
    auto* bos = model_proto.add_pieces();
    bos->set_piece("<s>");
    bos->set_type(ModelProto::SentencePiece::CONTROL);
    auto* eos = model_proto.add_pieces();
    eos->set_piece("</s>");
    eos->set_type(ModelProto::SentencePiece::CONTROL);
    return model_proto;
  };

  {
    SentencePieceProcessor sp;
    ModelProto model_proto = MakeBaseProto();
    auto* p3 = model_proto.add_pieces();
    p3->set_piece(WS "ABC");
    p3->set_score(3.0);
    auto* p4 = model_proto.add_pieces();
    p4->set_piece(WS "DE");
    p4->set_score(2.0);
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    const EncodeResult result = {
        {WS "ABC", 3}, {WS "DE", 4}, {"F", 0}, {"</s>", 2}};

    std::vector<std::string> output;
    EXPECT_TRUE(sp.Encode("ABC DEF", &output).ok());
    EXPECT_EQ(GetSpVec(result), output);

    std::vector<int> ids;
    EXPECT_TRUE(sp.Encode("ABC DEF", &ids).ok());
    EXPECT_EQ(GetIdVec(result), ids);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("ABC DEF", &spt).ok());
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
    SentencePieceProcessor sp;
    ModelProto model_proto = MakeBaseProto();
    auto* p3 = model_proto.add_pieces();
    p3->set_piece(WS "ABC");
    p3->set_score(3.0);
    auto* p4 = model_proto.add_pieces();
    p4->set_piece(WS "D");
    p4->set_score(2.0);
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    const EncodeResult expected = {
        {WS "ABC", 3}, {WS "D", 4}, {"EF", 0}, {"</s>", 2}};

    std::vector<std::string> output;
    EXPECT_TRUE(sp.Encode("ABC DEF", &output).ok());
    EXPECT_EQ(GetSpVec(expected), output);

    std::vector<int> ids;
    EXPECT_TRUE(sp.Encode("ABC DEF", &ids).ok());
    EXPECT_EQ(GetIdVec(expected), ids);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("ABC DEF", &spt).ok());
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

  // Byte-fallback.
  {
    SentencePieceProcessor sp;
    ModelProto model_proto = MakeBaseProto();
    model_proto.mutable_trainer_spec()->set_byte_fallback(true);
    auto* p3 = model_proto.add_pieces();
    p3->set_piece(WS "ABC");
    p3->set_score(3.0);
    auto* p4 = model_proto.add_pieces();
    p4->set_piece(WS "D");
    p4->set_score(2.0);
    for (int i = 0; i < 256; ++i) {
      auto* bp = model_proto.add_pieces();
      bp->set_piece(ByteToPiece(i));
      bp->set_type(ModelProto::SentencePiece::BYTE);
    }
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    const EncodeResult expected = {{WS "ABC", 3},
                                   {WS "D", 4},
                                   {"<0x45>", sp.PieceToId("<0x45>")},
                                   {"<0x46>", sp.PieceToId("<0x46>")},
                                   {"<0xE3>", sp.PieceToId("<0xE3>")},
                                   {"<0x81>", sp.PieceToId("<0x81>")},
                                   {"<0x82>", sp.PieceToId("<0x82>")},
                                   {"</s>", 2}};

    std::vector<std::string> output;
    EXPECT_TRUE(sp.Encode("ABC DEFあ", &output).ok());
    EXPECT_EQ(GetSpVec(expected), output);

    std::vector<int> ids;
    EXPECT_TRUE(sp.Encode("ABC DEFあ", &ids).ok());
    EXPECT_EQ(GetIdVec(expected), ids);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("ABC DEFあ", &spt).ok());
    EXPECT_EQ(8, spt.pieces_size());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(expected[i].first, spt.pieces(i).piece());
    }

    EXPECT_EQ("ABC", spt.pieces(0).surface());
    EXPECT_EQ(" D", spt.pieces(1).surface());
    EXPECT_EQ("E", spt.pieces(2).surface());
    EXPECT_EQ("F", spt.pieces(3).surface());
    EXPECT_EQ("", spt.pieces(4).surface());    // あ
    EXPECT_EQ("", spt.pieces(5).surface());    // あ
    EXPECT_EQ("あ", spt.pieces(6).surface());  // あ
    EXPECT_EQ("", spt.pieces(7).surface());    // </s>

    EXPECT_EQ(0, spt.pieces(0).begin());
    EXPECT_EQ(3, spt.pieces(0).end());
    EXPECT_EQ(3, spt.pieces(1).begin());
    EXPECT_EQ(5, spt.pieces(1).end());
    EXPECT_EQ(5, spt.pieces(2).begin());
    EXPECT_EQ(6, spt.pieces(2).end());
    EXPECT_EQ(6, spt.pieces(3).begin());
    EXPECT_EQ(7, spt.pieces(3).end());
    EXPECT_EQ(7, spt.pieces(4).begin());  // あ
    EXPECT_EQ(7, spt.pieces(4).end());
    EXPECT_EQ(7, spt.pieces(5).begin());  // あ
    EXPECT_EQ(7, spt.pieces(5).end());
    EXPECT_EQ(7, spt.pieces(6).begin());  // あ
    EXPECT_EQ(10, spt.pieces(6).end());
    EXPECT_EQ(10, spt.pieces(7).begin());  // </s>
    EXPECT_EQ(10, spt.pieces(7).end());
  }

  // Halfwidth to Fullwidth katakana normalization.
  {
    SentencePieceProcessor sp;
    ModelProto model_proto = MakeBaseProto();
    auto* p3 = model_proto.add_pieces();
    p3->set_piece(WS "グー");
    p3->set_score(3.0);
    auto* p4 = model_proto.add_pieces();
    p4->set_piece("グル");
    p4->set_score(2.0);
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    const EncodeResult result = {{WS "グー", 3}, {"グル", 4}, {"</s>", 2}};
    std::vector<std::string> output;
    EXPECT_TRUE(sp.Encode("ｸﾞｰｸﾞﾙ", &output).ok());
    EXPECT_EQ(GetSpVec(result), output);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("ｸﾞｰｸﾞﾙ", &spt).ok());
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
    SentencePieceProcessor sp;
    ModelProto model_proto = MakeBaseProto();
    auto* p3 = model_proto.add_pieces();
    p3->set_piece(WS "株式");
    p3->set_score(3.0);
    auto* p4 = model_proto.add_pieces();
    p4->set_piece("会社");
    p4->set_score(2.0);
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    const EncodeResult result = {{WS "株式", 3}, {"会社", 4}, {"</s>", 2}};
    std::vector<std::string> output;
    EXPECT_TRUE(sp.Encode("㍿", &output).ok());
    EXPECT_EQ(GetSpVec(result), output);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("㍿", &spt).ok());
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

namespace {
ModelProto MakeNBestUnigramModelProto() {
  ModelProto model_proto;
  *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
  auto* unk = model_proto.add_pieces();
  unk->set_piece("<unk>");
  unk->set_type(ModelProto::SentencePiece::UNKNOWN);
  auto* bos = model_proto.add_pieces();
  bos->set_piece("<s>");
  bos->set_type(ModelProto::SentencePiece::CONTROL);
  auto* eos = model_proto.add_pieces();
  eos->set_piece("</s>");
  eos->set_type(ModelProto::SentencePiece::CONTROL);

  auto add_piece = [&](absl::string_view piece, float score) {
    auto* sp = model_proto.add_pieces();
    sp->set_piece(std::string(piece));
    sp->set_score(score);
  };
  add_piece(WS "ABC", -1.0f);
  add_piece(WS "DE", -1.0f);
  add_piece("F", -1.0f);
  add_piece(WS "AB", -1.5f);
  add_piece("C", -1.5f);
  add_piece(WS "D", -1.5f);
  add_piece("EF", -1.5f);
  return model_proto;
}
}  // namespace

TEST(SentencepieceProcessorTest, NBestEncodeTest) {
  SentencePieceProcessor uninit_sp;
  std::vector<std::vector<std::string>> output;
  EXPECT_FALSE(uninit_sp.NBestEncode("ABC DEF", 2, &output).ok());

  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(MakeNBestUnigramModelProto()));
  ABSL_ASSERT_OK(sp.SetEncodeExtraOptions("eos"));

  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 2, &output));
  ASSERT_EQ(2u, output.size());
  EXPECT_EQ(std::vector<std::string>({WS "ABC", WS "DE", "F", "</s>"}),
            output[0]);
  EXPECT_EQ(std::vector<std::string>({WS "ABC", WS "D", "EF", "</s>"}),
            output[1]);

  std::vector<std::vector<int>> ids;
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 2, &ids));
  EXPECT_EQ(output.size(), ids.size());
  EXPECT_EQ(std::vector<int>({3, 4, 5, 2}), ids[0]);
  EXPECT_EQ(std::vector<int>({3, 8, 9, 2}), ids[1]);

  NBestSentencePieceText spt;
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 2, &spt));
  ASSERT_EQ(2, spt.nbests_size());
  EXPECT_EQ(4, spt.nbests(0).pieces_size());
  EXPECT_NEAR(-3.0f, spt.nbests(0).score(), 1e-4f);
  EXPECT_NEAR(-4.0f, spt.nbests(1).score(), 1e-4f);

  // Exact enumeration of the top 4 segmentations in descending score order:
  //   1. [▁ABC(-1.0), ▁DE(-1.0), F(-1.0)]            = -3.0
  //   2. [▁ABC(-1.0), ▁D(-1.5),  EF(-1.5)]           = -4.0
  //   3. [▁AB(-1.5),  C(-1.5),   ▁DE(-1.0), F(-1.0)] = -5.0
  //   4. [▁AB(-1.5),  C(-1.5),   ▁D(-1.5), EF(-1.5)] = -6.0
  std::vector<std::vector<std::string>> top4;
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 4, &top4));
  ASSERT_EQ(4u, top4.size());
  EXPECT_EQ(std::vector<std::string>({WS "ABC", WS "DE", "F", "</s>"}),
            top4[0]);
  EXPECT_EQ(std::vector<std::string>({WS "ABC", WS "D", "EF", "</s>"}),
            top4[1]);
  EXPECT_EQ(std::vector<std::string>({WS "AB", "C", WS "DE", "F", "</s>"}),
            top4[2]);
  EXPECT_EQ(std::vector<std::string>({WS "AB", "C", WS "D", "EF", "</s>"}),
            top4[3]);

  // Boundary checks on nbest_size (1 <= nbest_size <= 512).
  EXPECT_FALSE(sp.NBestEncode("ABC DEF", -1, &output).ok());
  EXPECT_FALSE(sp.NBestEncode("ABC DEF", 0, &output).ok());
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 512, &output));
  ASSERT_GE(output.size(), 4u);
  EXPECT_EQ(top4[0], output[0]);
  EXPECT_EQ(top4[3], output[3]);
  EXPECT_FALSE(sp.NBestEncode("ABC DEF", 513, &output).ok());
  EXPECT_FALSE(sp.NBestEncode("ABC DEF", 1024, &output).ok());
}

TEST(SentencepieceProcessorTest, NBestEncodeEmptyAndExtraOptionsTest) {
  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(MakeNBestUnigramModelProto()));
  ABSL_ASSERT_OK(sp.SetEncodeExtraOptions("eos"));

  for (const char* empty_in : {"", "   "}) {
    SCOPED_TRACE(testing::Message() << "empty_in=\"" << empty_in << "\"");
    NBestSentencePieceText empty_spt;
    ABSL_ASSERT_OK(sp.NBestEncode(empty_in, 5, &empty_spt));
    ASSERT_EQ(1, empty_spt.nbests_size());
    EXPECT_FLOAT_EQ(0.0f, empty_spt.nbests(0).score());
    ASSERT_EQ(1, empty_spt.nbests(0).pieces_size());
    EXPECT_EQ("</s>", empty_spt.nbests(0).pieces(0).piece());
  }

  ABSL_ASSERT_OK(sp.SetEncodeExtraOptions("reverse:bos:eos"));
  std::vector<std::vector<std::string>> rev_out;
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 2, &rev_out));
  ASSERT_EQ(2u, rev_out.size());
  EXPECT_EQ(std::vector<std::string>({"<s>", "F", WS "DE", WS "ABC", "</s>"}),
            rev_out[0]);

  ABSL_ASSERT_OK(sp.SetEncodeExtraOptions("unk"));
  std::vector<std::vector<std::string>> unk_out;
  ABSL_ASSERT_OK(sp.NBestEncode("XYZ", 5, &unk_out));
  ASSERT_EQ(1u, unk_out.size());
  EXPECT_EQ(std::vector<std::string>({"<unk>"}), unk_out[0]);
}

TEST(SentencepieceProcessorTest, NBestEncodeFlatbufferAndModelTypeTest) {
  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(MakeNBestUnigramModelProto()));
  ABSL_ASSERT_OK(sp.SetEncodeExtraOptions("eos"));
  std::vector<std::vector<std::string>> expected;
  ABSL_ASSERT_OK(sp.NBestEncode("ABC DEF", 4, &expected));

  for (const bool as_view : {false, true}) {
    SCOPED_TRACE(testing::Message() << "as_view=" << as_view);
    SentencePieceProcessor fb_sp;
    if (as_view) {
      ABSL_ASSERT_OK(fb_sp.LoadFromFlatbufferView(sp.serialized_flatbuffer()));
    } else {
      ABSL_ASSERT_OK(
          fb_sp.LoadFromSerializedFlatbuffer(sp.serialized_flatbuffer()));
    }
    ABSL_ASSERT_OK(fb_sp.SetEncodeExtraOptions("eos"));
    std::vector<std::vector<std::string>> fb_out;
    ABSL_ASSERT_OK(fb_sp.NBestEncode("ABC DEF", 4, &fb_out));
    EXPECT_EQ(expected, fb_out);
  }

  for (const auto mtype :
       {TrainerSpec::BPE, TrainerSpec::CHAR, TrainerSpec::WORD}) {
    SCOPED_TRACE(testing::Message() << "model_type=" << mtype);
    ModelProto non_uni;
    non_uni.mutable_trainer_spec()->set_model_type(mtype);
    auto* unk_p = non_uni.add_pieces();
    unk_p->set_piece("<unk>");
    unk_p->set_type(ModelProto::SentencePiece::UNKNOWN);
    auto* a_p = non_uni.add_pieces();
    a_p->set_piece("a");
    a_p->set_score(-1.0f);
    SentencePieceProcessor non_uni_sp;
    ABSL_ASSERT_OK(non_uni_sp.Load(non_uni));
    std::vector<std::vector<std::string>> non_uni_out;
    EXPECT_FALSE(non_uni_sp.NBestEncode("a", 2, &non_uni_out).ok());
  }

  SentencePieceProcessor botchan_sp;
  ABSL_ASSERT_OK(botchan_sp.Load(filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model")));
  for (const char* text : {"hello world", "I have a pen."}) {
    SCOPED_TRACE(testing::Message() << "text=" << text);
    NBestSentencePieceText botchan_nbest;
    ABSL_ASSERT_OK(botchan_sp.NBestEncode(text, 10, &botchan_nbest));
    ASSERT_EQ(10, botchan_nbest.nbests_size());
    absl::flat_hash_set<std::string> unique_cands;
    for (int i = 0; i < botchan_nbest.nbests_size(); ++i) {
      SCOPED_TRACE(testing::Message() << "rank=" << i);
      if (i > 0) {
        EXPECT_LE(botchan_nbest.nbests(i).score(),
                  botchan_nbest.nbests(i - 1).score());
      }
      unique_cands.insert(
          absl::StrJoin(GetSpVec(botchan_nbest.nbests(i)), "|"));
      std::string decoded;
      ABSL_ASSERT_OK(
          botchan_sp.Decode(GetSpVec(botchan_nbest.nbests(i)), &decoded));
      EXPECT_EQ(text, decoded);
    }
    EXPECT_EQ(10u, unique_cands.size());
  }
}

TEST(SentencepieceProcessorTest, NBestEncodeSpecialPiecesAndByteFallbackTest) {
  for (const bool byte_fallback : {false, true}) {
    SCOPED_TRACE(testing::Message() << "byte_fallback=" << byte_fallback);
    ModelProto fb_proto;
    fb_proto.mutable_normalizer_spec()->set_name("identity");
    fb_proto.mutable_normalizer_spec()->set_add_dummy_prefix(false);
    fb_proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
    fb_proto.mutable_normalizer_spec()->set_escape_whitespaces(false);
    fb_proto.mutable_trainer_spec()->set_byte_fallback(byte_fallback);
    auto* u = fb_proto.add_pieces();
    u->set_piece("<unk>");
    u->set_type(ModelProto::SentencePiece::UNKNOWN);
    auto* bos = fb_proto.add_pieces();
    bos->set_piece("<s>");
    bos->set_type(ModelProto::SentencePiece::CONTROL);
    auto* unused = fb_proto.add_pieces();
    unused->set_piece("<unused>");
    unused->set_type(ModelProto::SentencePiece::UNUSED);
    unused->set_score(100.0f);
    if (byte_fallback) {
      for (int i = 0; i < 256; ++i) {
        auto* bp = fb_proto.add_pieces();
        bp->set_piece(ByteToPiece(i));
        bp->set_type(ModelProto::SentencePiece::BYTE);
      }
    }
    auto* ud = fb_proto.add_pieces();
    ud->set_piece("<UD>");
    ud->set_type(ModelProto::SentencePiece::USER_DEFINED);
    auto add_fb_piece = [&](absl::string_view piece, float score) {
      auto* p = fb_proto.add_pieces();
      p->set_piece(std::string(piece));
      p->set_score(score);
    };
    // min_score = -20.0 -> unk_score = -30.0.
    // On "ABXY" (where X, Y are OOV and "ABXY" is not in vocab), 1-best is
    // "AB" (-1.0) + OOV("X") (-30.0) + OOV("Y") (-30.0) = -61.0.
    add_fb_piece("C", -20.0f);
    add_fb_piece("AB", -1.0f);
    add_fb_piece("<", -1.0f);
    add_fb_piece("U", -1.0f);
    add_fb_piece("D", -1.0f);
    add_fb_piece(">", -1.0f);

    SentencePieceProcessor fb_sp;
    ABSL_ASSERT_OK(fb_sp.Load(fb_proto));

    NBestSentencePieceText nbest_oov;
    ABSL_ASSERT_OK(fb_sp.NBestEncode("ABXY", 5, &nbest_oov));
    ASSERT_GE(nbest_oov.nbests_size(), 1);
    EXPECT_NEAR(-61.0f, nbest_oov.nbests(0).score(), 1e-4f);
    const auto& rank0 = nbest_oov.nbests(0);
    EXPECT_EQ("ABXY", rank0.text());
    if (byte_fallback) {
      ASSERT_EQ(3, rank0.pieces_size());
      EXPECT_EQ(std::vector<std::string>({"AB", "<0x58>", "<0x59>"}),
                GetSpVec(rank0));
      EXPECT_EQ("X", rank0.pieces(1).surface());
      EXPECT_EQ(2u, rank0.pieces(1).begin());
      EXPECT_EQ(3u, rank0.pieces(1).end());
      EXPECT_EQ("Y", rank0.pieces(2).surface());
      EXPECT_EQ(3u, rank0.pieces(2).begin());
      EXPECT_EQ(4u, rank0.pieces(2).end());
      std::string dec;
      ABSL_ASSERT_OK(fb_sp.Decode(GetSpVec(rank0), &dec));
      EXPECT_EQ("ABXY", dec);
    } else {
      ASSERT_EQ(2, rank0.pieces_size());
      EXPECT_EQ(std::vector<std::string>({"AB", "XY"}), GetSpVec(rank0));
      EXPECT_EQ(0, rank0.pieces(1).id());
      EXPECT_EQ("XY", rank0.pieces(1).surface());
      EXPECT_EQ(2u, rank0.pieces(1).begin());
      EXPECT_EQ(4u, rank0.pieces(1).end());
    }

    // Literal CONTROL ("<s>"), UNUSED ("<unused>"), and BYTE ("<0x41>") strings
    // in input text are never matched as those vocabulary token IDs.
    for (const char* literal : {"<s>", "<unused>", "<0x41>"}) {
      SCOPED_TRACE(testing::Message() << "literal=" << literal);
      NBestSentencePieceText lit_nbest;
      ABSL_ASSERT_OK(fb_sp.NBestEncode(literal, 5, &lit_nbest));
      for (const auto& cand : lit_nbest.nbests()) {
        for (const auto& p : cand.pieces()) {
          EXPECT_FALSE(fb_sp.IsControl(p.id()));
          EXPECT_FALSE(fb_sp.IsUnused(p.id()));
          if (!byte_fallback) {
            EXPECT_FALSE(fb_sp.IsByte(p.id()));
          } else if (fb_sp.IsByte(p.id())) {
            EXPECT_LE(p.end() - p.begin(), 1u);
          }
        }
      }
    }

    NBestSentencePieceText nbest_ud;
    ABSL_ASSERT_OK(fb_sp.NBestEncode("<UD>", 5, &nbest_ud));
    ASSERT_EQ(2, nbest_ud.nbests_size());
    EXPECT_NEAR(0.3f, nbest_ud.nbests(0).score(), 1e-4f);
    EXPECT_NEAR(-4.0f, nbest_ud.nbests(1).score(), 1e-4f);
  }
}

TEST(SentencepieceProcessorTest, NBestEncodeByteFallbackWithoutUnkPieceTest) {
  ModelProto no_unk_proto;
  no_unk_proto.mutable_normalizer_spec()->set_name("identity");
  no_unk_proto.mutable_normalizer_spec()->set_add_dummy_prefix(false);
  no_unk_proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
  no_unk_proto.mutable_normalizer_spec()->set_escape_whitespaces(false);
  no_unk_proto.mutable_trainer_spec()->set_byte_fallback(true);
  auto* p0 = no_unk_proto.add_pieces();
  p0->set_piece("AB");  // id == 0 (NORMAL, must NOT be mistaken for unk_id)
  p0->set_score(-1.0f);
  for (int i = 0; i < 256; ++i) {
    auto* bp = no_unk_proto.add_pieces();
    bp->set_piece(ByteToPiece(i));
    bp->set_type(ModelProto::SentencePiece::BYTE);
  }
  SentencePieceProcessor no_unk_sp;
  ABSL_ASSERT_OK(no_unk_sp.Load(no_unk_proto));
  EXPECT_EQ(-1, no_unk_sp.unk_id());

  SentencePieceText enc_spt;
  NBestSentencePieceText nbest_no_unk;
  ABSL_ASSERT_OK(no_unk_sp.Encode("ABあ", &enc_spt));
  ABSL_ASSERT_OK(no_unk_sp.NBestEncode("ABあ", 5, &nbest_no_unk));
  ASSERT_GE(nbest_no_unk.nbests_size(), 1);
  EXPECT_EQ(GetSpVec(enc_spt), GetSpVec(nbest_no_unk.nbests(0)));
  EXPECT_EQ(std::vector<std::string>({"AB", "<0xE3>", "<0x81>", "<0x82>"}),
            GetSpVec(nbest_no_unk.nbests(0)));
  EXPECT_EQ("", nbest_no_unk.nbests(0).pieces(1).surface());
  EXPECT_EQ(2u, nbest_no_unk.nbests(0).pieces(1).begin());
  EXPECT_EQ(2u, nbest_no_unk.nbests(0).pieces(1).end());
  EXPECT_EQ("あ", nbest_no_unk.nbests(0).pieces(3).surface());
  EXPECT_EQ(2u, nbest_no_unk.nbests(0).pieces(3).begin());
  EXPECT_EQ(5u, nbest_no_unk.nbests(0).pieces(3).end());
}

TEST(SentencepieceProcessorTest, NBestEncodeScoreTiesTest) {
  // "aaaaa" over {"a"(-1), "aa"(-2), "aaa"(-3), "aaaa"(-4), "aaaaa"(-5)} has
  // 2^4 = 16 distinct segmentations, all with exact score -5.0.
  ModelProto tie_proto;
  tie_proto.mutable_normalizer_spec()->set_name("identity");
  tie_proto.mutable_normalizer_spec()->set_add_dummy_prefix(false);
  tie_proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
  tie_proto.mutable_normalizer_spec()->set_escape_whitespaces(false);
  auto* u = tie_proto.add_pieces();
  u->set_piece("<unk>");
  u->set_type(ModelProto::SentencePiece::UNKNOWN);
  for (int len = 1; len <= 5; ++len) {
    auto* p = tie_proto.add_pieces();
    p->set_piece(std::string(len, 'a'));
    p->set_score(-static_cast<float>(len));
  }
  SentencePieceProcessor tie_sp;
  ABSL_ASSERT_OK(tie_sp.Load(tie_proto));

  for (const int k : {1, 3, 7, 16, 32}) {
    SCOPED_TRACE(testing::Message() << "k=" << k);
    NBestSentencePieceText tie_nbest;
    ABSL_ASSERT_OK(tie_sp.NBestEncode("aaaaa", k, &tie_nbest));
    const int expected_n = std::min(k, 16);
    ASSERT_EQ(expected_n, tie_nbest.nbests_size());
    absl::flat_hash_set<std::string> unique_segs;
    for (int i = 0; i < expected_n; ++i) {
      SCOPED_TRACE(testing::Message() << "rank=" << i);
      EXPECT_NEAR(-5.0f, tie_nbest.nbests(i).score(), 1e-5f);
      unique_segs.insert(absl::StrJoin(GetSpVec(tie_nbest.nbests(i)), "|"));
    }
    EXPECT_EQ(static_cast<size_t>(expected_n), unique_segs.size());
  }
}

TEST(SentencepieceProcessorTest, SampleEncodeTest) {
  ModelProto model_proto;
  *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
  auto* unk = model_proto.add_pieces();
  unk->set_piece("<unk>");
  unk->set_type(ModelProto::SentencePiece::UNKNOWN);
  auto* bos = model_proto.add_pieces();
  bos->set_piece("<s>");
  bos->set_type(ModelProto::SentencePiece::CONTROL);
  auto* eos = model_proto.add_pieces();
  eos->set_piece("</s>");
  eos->set_type(ModelProto::SentencePiece::CONTROL);

  auto add_piece = [&](absl::string_view piece, float score) {
    auto* sp = model_proto.add_pieces();
    sp->set_piece(std::string(piece));
    sp->set_score(score);
  };
  add_piece(WS "ABC", -1.0f);
  add_piece(WS "DE", -1.0f);
  add_piece("F", -1.0f);
  add_piece(WS "AB", -2.0f);
  add_piece("C", -2.0f);

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(model_proto).ok());

  std::vector<std::string> output;
  EXPECT_TRUE(sp.SampleEncode("ABC DEF", -1, 0.5, &output).ok());
  EXPECT_FALSE(output.empty());

  std::vector<int> ids;
  EXPECT_TRUE(sp.SampleEncode("ABC DEF", -1, 0.5, &ids).ok());
  EXPECT_FALSE(ids.empty());

  SentencePieceText spt;
  EXPECT_TRUE(sp.SampleEncode("ABC DEF", -1, 0.5, &spt).ok());
  EXPECT_GT(spt.pieces_size(), 0);

  // Verify SetRandomGeneratorSeed determinism.
  SetRandomGeneratorSeed(12345);
  std::vector<int> ids1;
  for (int i = 0; i < 10; ++i) {
    std::vector<int> tmp;
    EXPECT_TRUE(sp.SampleEncode("ABC DEF", -1, 0.5, &tmp).ok());
    ids1.insert(ids1.end(), tmp.begin(), tmp.end());
  }
  SetRandomGeneratorSeed(12345);
  std::vector<int> ids2;
  for (int i = 0; i < 10; ++i) {
    std::vector<int> tmp;
    EXPECT_TRUE(sp.SampleEncode("ABC DEF", -1, 0.5, &tmp).ok());
    ids2.insert(ids2.end(), tmp.begin(), tmp.end());
  }
  EXPECT_EQ(ids1, ids2);
}

TEST(SentencepieceProcessorTest, DecodeTest) {
  auto MakeDecodeProto = [&]() {
    ModelProto model_proto;
    *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
    auto add_piece = [&](absl::string_view piece,
                         ModelProto::SentencePiece::Type type =
                             ModelProto::SentencePiece::NORMAL) {
      auto* sp = model_proto.add_pieces();
      sp->set_piece(std::string(piece));
      sp->set_type(type);
    };
    add_piece("<unk>", ModelProto::SentencePiece::UNKNOWN);
    add_piece("<s>", ModelProto::SentencePiece::CONTROL);
    add_piece("</s>", ModelProto::SentencePiece::CONTROL);
    add_piece(WS "ABC");
    add_piece(WS "DE");
    add_piece("F");
    add_piece("G" WS "H");
    return model_proto;
  };

  const std::vector<absl::string_view> input = {
      "<s>", WS "ABC", "<unk>", WS "DE", "F", "G" WS "H", "I", "</s>"};

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;

    EXPECT_TRUE(sp.Decode(input, &spt).ok());
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

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    proto.mutable_trainer_spec()->set_unk_surface("");
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ("ABC DEFG HI", spt.text());
    EXPECT_EQ(8, spt.pieces_size());
  }

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    proto.mutable_trainer_spec()->set_unk_surface("<UNK>");
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ("ABC<UNK> DEFG HI", spt.text());
    EXPECT_EQ(8, spt.pieces_size());
  }

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    proto.mutable_trainer_spec()->set_unk_surface("");
    proto.mutable_normalizer_spec()->set_add_dummy_prefix(false);
    proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ(" ABC DEFG HI", spt.text());
    EXPECT_EQ(8, spt.pieces_size());
  }
}

TEST(SentencepieceProcessorTest, DummyPrefixDecodeTest) {
  auto MakeDecodeProto = [&]() {
    ModelProto model_proto;
    *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
    auto add_piece = [&](absl::string_view piece,
                         ModelProto::SentencePiece::Type type =
                             ModelProto::SentencePiece::NORMAL) {
      auto* sp = model_proto.add_pieces();
      sp->set_piece(std::string(piece));
      sp->set_type(type);
    };
    add_piece("<unk>", ModelProto::SentencePiece::UNKNOWN);
    add_piece("<s>", ModelProto::SentencePiece::CONTROL);
    add_piece("</s>", ModelProto::SentencePiece::CONTROL);
    add_piece(WS "ABC");
    add_piece(WS "DE");
    add_piece("F");
    add_piece("G" WS "H");
    add_piece(WS);
    return model_proto;
  };

  // start the sequence with a whitespace token
  const std::vector<absl::string_view> input = {
      "<s>", WS, WS "ABC", "<unk>", WS "DE", "F", "G" WS "H", "I", "</s>"};

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    proto.mutable_trainer_spec()->set_unk_surface("");
    proto.mutable_normalizer_spec()->set_add_dummy_prefix(true);
    proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ(" ABC DEFG HI", spt.text());
    EXPECT_EQ(9, spt.pieces_size());
  }

  {
    SentencePieceProcessor sp;
    ModelProto proto = MakeDecodeProto();
    proto.mutable_trainer_spec()->set_unk_surface("");
    proto.mutable_normalizer_spec()->set_add_dummy_prefix(true);
    proto.mutable_normalizer_spec()->set_remove_extra_whitespaces(true);
    EXPECT_TRUE(sp.Load(proto).ok());

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ("ABC DEFG HI", spt.text());
    EXPECT_EQ(9, spt.pieces_size());
  }
}

TEST(SentencepieceProcessorTest, ByteFallbackDecodeTest) {
  ModelProto model_proto;
  *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
  model_proto.mutable_trainer_spec()->set_byte_fallback(true);
  auto add_piece = [&](absl::string_view piece,
                       ModelProto::SentencePiece::Type type =
                           ModelProto::SentencePiece::NORMAL) {
    auto* sp = model_proto.add_pieces();
    sp->set_piece(std::string(piece));
    sp->set_type(type);
  };
  add_piece("<unk>", ModelProto::SentencePiece::UNKNOWN);
  add_piece("<s>", ModelProto::SentencePiece::CONTROL);
  add_piece("</s>", ModelProto::SentencePiece::CONTROL);
  add_piece("A");
  add_piece("B");
  add_piece("C");
  for (int i = 0; i < 256; ++i) {
    add_piece(ByteToPiece(i), ModelProto::SentencePiece::BYTE);
  }

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(model_proto).ok());

  {
    const std::vector<absl::string_view> input = {
        "<s>",
        "A",
        "B",
        // "あ" -> 0xE3 0x81 0x82
        "<0xE3>",
        "<0x81>",
        "<0x82>",
        // "Z" -> 0x5A
        "<0x5A>",
        // "Ω" -> 0xCE 0xA9
        "<0xCE>",
        "<0xA9>",
        "C",
        // Invalid UTF-8 bytes.
        "<0xE0>",
        "<0x80>",
        // "い" -> 0xE3 0x81 0x84
        "<0xE3>",
        "<0x81>",
        "<0x84>",
        // REPLACEMENT CHARACTER as byte pieces.
        "<0xEF>",
        "<0xBF>",
        "<0xBD>",
    };

    SentencePieceText spt;
    EXPECT_TRUE(sp.Decode(input, &spt).ok());
    EXPECT_EQ("ABあZΩC\xEF\xBF\xBD\xEF\xBF\xBDい\xEF\xBF\xBD", spt.text());
    EXPECT_EQ(18, spt.pieces_size());

    for (int i = 0; i < 18; ++i) {
      EXPECT_EQ(input[i], spt.pieces(i).piece());
    }

    EXPECT_EQ("", spt.pieces(0).surface());
    EXPECT_EQ(0, spt.pieces(0).begin());
    EXPECT_EQ(0, spt.pieces(0).end());

    EXPECT_EQ("A", spt.pieces(1).surface());
    EXPECT_EQ(0, spt.pieces(1).begin());
    EXPECT_EQ(1, spt.pieces(1).end());

    EXPECT_EQ("B", spt.pieces(2).surface());
    EXPECT_EQ(1, spt.pieces(2).begin());
    EXPECT_EQ(2, spt.pieces(2).end());

    EXPECT_EQ("", spt.pieces(3).surface());
    EXPECT_EQ("", spt.pieces(4).surface());
    EXPECT_EQ("あ", spt.pieces(5).surface());
    EXPECT_EQ(2, spt.pieces(3).begin());
    EXPECT_EQ(2, spt.pieces(3).end());
    EXPECT_EQ(2, spt.pieces(4).begin());
    EXPECT_EQ(2, spt.pieces(4).end());
    EXPECT_EQ(2, spt.pieces(5).begin());
    EXPECT_EQ(5, spt.pieces(5).end());

    EXPECT_EQ("Z", spt.pieces(6).surface());
    EXPECT_EQ(5, spt.pieces(6).begin());
    EXPECT_EQ(6, spt.pieces(6).end());

    EXPECT_EQ("", spt.pieces(7).surface());
    EXPECT_EQ("Ω", spt.pieces(8).surface());
    EXPECT_EQ(6, spt.pieces(7).begin());
    EXPECT_EQ(6, spt.pieces(7).end());
    EXPECT_EQ(6, spt.pieces(8).begin());
    EXPECT_EQ(8, spt.pieces(8).end());

    EXPECT_EQ("C", spt.pieces(9).surface());
    EXPECT_EQ(8, spt.pieces(9).begin());
    EXPECT_EQ(9, spt.pieces(9).end());

    EXPECT_EQ("\xEF\xBF\xBD", spt.pieces(10).surface());
    EXPECT_EQ(9, spt.pieces(10).begin());
    EXPECT_EQ(12, spt.pieces(10).end());

    EXPECT_EQ("\xEF\xBF\xBD", spt.pieces(11).surface());
    EXPECT_EQ(12, spt.pieces(11).begin());
    EXPECT_EQ(15, spt.pieces(11).end());

    EXPECT_EQ("", spt.pieces(12).surface());
    EXPECT_EQ("", spt.pieces(13).surface());
    EXPECT_EQ("い", spt.pieces(14).surface());
    EXPECT_EQ(15, spt.pieces(12).begin());
    EXPECT_EQ(15, spt.pieces(12).end());
    EXPECT_EQ(15, spt.pieces(13).begin());
    EXPECT_EQ(15, spt.pieces(13).end());
    EXPECT_EQ(15, spt.pieces(14).begin());
    EXPECT_EQ(18, spt.pieces(14).end());

    EXPECT_EQ("", spt.pieces(15).surface());
    EXPECT_EQ("", spt.pieces(16).surface());
    EXPECT_EQ("\xEF\xBF\xBD", spt.pieces(17).surface());
    EXPECT_EQ(18, spt.pieces(15).begin());
    EXPECT_EQ(18, spt.pieces(15).end());
    EXPECT_EQ(18, spt.pieces(16).begin());
    EXPECT_EQ(18, spt.pieces(16).end());
    EXPECT_EQ(18, spt.pieces(17).begin());
    EXPECT_EQ(21, spt.pieces(17).end());
  }
}

void AddPiece(ModelProto* model_proto, absl::string_view piece,
              float score = 0.0) {
  auto* sp = model_proto->add_pieces();
  sp->set_piece(std::string(piece));
  sp->set_score(score);
}

TEST(SentencePieceProcessorTest, LoadInvalidModelTest) {
  SentencePieceProcessor sp;
  EXPECT_FALSE(sp.Load("").ok());
  EXPECT_FALSE(sp.Load("__UNKNOWN_FILE__").ok());
}

TEST(SentencePieceProcessorTest, LoadSerializedProtoTest) {
  ModelProto model_proto;
  auto* sp1 = model_proto.add_pieces();
  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("<unk>");
  AddPiece(&model_proto, WS, 0.0);
  *(model_proto.mutable_normalizer_spec()) = MakeDefaultNormalizerSpec();

  SentencePieceProcessor sp;
  EXPECT_FALSE(sp.LoadFromSerializedProto("__NOT_A_PROTO__").ok());
  EXPECT_TRUE(sp.LoadFromSerializedProto(model_proto.SerializeAsString()).ok());
  EXPECT_EQ(model_proto.SerializeAsString(),
            sp.model_proto().SerializeAsString());
}

TEST(SentencePieceProcessorTest, EndToEndTest) {
  ModelProto model_proto;
  auto* sp1 = model_proto.add_pieces();
  auto* sp2 = model_proto.add_pieces();
  auto* sp3 = model_proto.add_pieces();

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

  {
    auto output = filesystem::NewWritableFile(
        filesystem::JoinPath(::testing::TempDir(), "model"), true);
    output->Write(model_proto.SerializeAsString());
  }

  SentencePieceProcessor sp;
  EXPECT_TRUE(
      sp.Load(filesystem::JoinPath(::testing::TempDir(), "model")).ok());

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
  EXPECT_EQ(0, sp.PieceToId("unknown_piece"));

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

  EXPECT_EQ(0, sp.unk_id());
  EXPECT_EQ(1, sp.bos_id());
  EXPECT_EQ(2, sp.eos_id());
  EXPECT_EQ(-1, sp.pad_id());

  {
    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {WS, "ab", "c"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {7, 6, 5};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("bos").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", WS, "ab", "c"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 7, 6, 5};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("abc", &spt).ok());
    EXPECT_EQ(spt.pieces(0).begin(), 0);
    EXPECT_EQ(spt.pieces(0).end(), 0);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("eos").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {WS, "ab", "c", "</s>"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {7, 6, 5, 2};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);

    SentencePieceText spt;
    EXPECT_TRUE(sp.Encode("abc", &spt).ok());
    EXPECT_EQ(spt.pieces(3).piece(), "</s>");
    EXPECT_EQ(spt.pieces(3).begin(), 3);
    EXPECT_EQ(spt.pieces(3).end(), 3);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("reverse").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"c", "ab", WS};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {5, 6, 7};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("bos:eos").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", WS, "ab", "c",
                                                   "</s>"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 7, 6, 5, 2};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("reverse:bos:eos").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"<s>", "c", "ab", WS,
                                                   "</s>"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {1, 5, 6, 7, 2};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);
  }

  {
    EXPECT_TRUE(sp.SetEncodeExtraOptions("bos:eos:reverse").ok());

    std::vector<std::string> sps;
    const std::vector<std::string> expected_str = {"</s>", "c", "ab", WS,
                                                   "<s>"};
    EXPECT_TRUE(sp.Encode("abc", &sps).ok());
    EXPECT_EQ(expected_str, sps);

    std::vector<int> ids;
    const std::vector<int> expected_id = {2, 5, 6, 7, 1};
    EXPECT_TRUE(sp.Encode("abc", &ids).ok());
    EXPECT_EQ(expected_id, ids);
  }

  {
    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("abc", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("bos").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("abc", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("eos").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("abc", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("reverse").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("cba", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("bos:eos").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("abc", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("reverse:bos:eos").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("cba", output);
  }

  // Out of range
  {
    std::string output;
    const std::vector<int> ids = {3, 4, 127};
    EXPECT_FALSE(sp.Decode(ids, &output).ok());
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("bos:eos:reverse").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("cab", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("cba", output);
  }

  {
    EXPECT_TRUE(sp.SetDecodeExtraOptions("reverse:reverse").ok());

    std::string output;
    const std::vector<std::string> sps = {"ab", "c"};
    EXPECT_TRUE(sp.Decode(sps, &output).ok());
    EXPECT_EQ("abc", output);

    const std::vector<int> ids = {3, 4, 5};
    EXPECT_TRUE(sp.Decode(ids, &output).ok());
    EXPECT_EQ("abc", output);
  }

  EXPECT_TRUE(sp.SetEncodeExtraOptions("").ok());
  EXPECT_TRUE(sp.SetDecodeExtraOptions("").ok());

  EXPECT_FALSE(sp.SetEncodeExtraOptions("foo").ok());
  EXPECT_FALSE(sp.SetDecodeExtraOptions("foo").ok());

  auto RunTest = [&model_proto](const SentencePieceProcessor& sp) {
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
      EXPECT_TRUE(sp.Encode("abc", &sps).ok());
      EXPECT_EQ(expected_str, sps);

      std::vector<int> ids;
      const std::vector<int> expected_id = {7, 6, 5};
      EXPECT_TRUE(sp.Encode("abc", &ids).ok());
      EXPECT_EQ(expected_id, ids);
    }

    {
      std::string output;
      const std::vector<std::string> sps = {"ab", "c"};
      EXPECT_TRUE(sp.Decode(sps, &output).ok());
      EXPECT_EQ("abc", output);

      const std::vector<int> ids = {3, 4, 5};
      EXPECT_TRUE(sp.Decode(ids, &output).ok());
      EXPECT_EQ("abc", output);
    }
  };

  // Copies ModelProto.
  {
    SentencePieceProcessor sp;
    const ModelProto copied = model_proto;
    EXPECT_TRUE(sp.Load(copied).ok());
    RunTest(sp);
  }

  // Moves ModelProto.
  {
    SentencePieceProcessor sp;
    auto moved = std::make_unique<ModelProto>();
    *moved = model_proto;
    EXPECT_TRUE(sp.Load(std::move(moved)).ok());
    EXPECT_EQ(&sp.model_proto(), &sp.model_proto());
    RunTest(sp);
  }
}

TEST(SentencePieceProcessorTest, SkipNormalizationTest) {
  ModelProto model_proto;
  auto* sp1 = model_proto.add_pieces();
  auto* sp2 = model_proto.add_pieces();

  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("<unk>");
  sp2->set_type(ModelProto::SentencePiece::USER_DEFINED);
  sp2->set_piece("<USER>");

  AddPiece(&model_proto, "a", 0.0);
  AddPiece(&model_proto, "b", 0.3);
  AddPiece(&model_proto, "c", 0.2);
  AddPiece(&model_proto, "u", 0.2);
  AddPiece(&model_proto, "s", 0.2);
  AddPiece(&model_proto, "e", 0.2);
  AddPiece(&model_proto, "r", 0.2);

  *(model_proto.mutable_normalizer_spec()) =
      SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc_cf");

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(model_proto).ok());

  std::vector<std::string> pieces;
  EXPECT_TRUE(sp.Encode("AB<USER>C<uSEr>", &pieces).ok());
  for (const auto& sp : pieces) LOG(INFO) << sp;
  EXPECT_EQ(std::vector<std::string>(
                {WS, "a", "b", "<USER>", "c", "<", "u", "s", "e", "r", ">"}),
            pieces);
}

TEST(SentencePieceProcessorTest, OverrideSpecialPieceTest) {
  ModelProto model_proto;
  auto* sp1 = model_proto.add_pieces();
  auto* sp2 = model_proto.add_pieces();
  auto* sp3 = model_proto.add_pieces();

  model_proto.mutable_trainer_spec()->set_unk_piece("__UNK__");
  model_proto.mutable_trainer_spec()->set_bos_piece("__BOS__");
  model_proto.mutable_trainer_spec()->set_eos_piece("__EOS__");
  model_proto.mutable_trainer_spec()->set_pad_piece("__PAD__");

  // No BOS/EOS.
  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("__UNK__");
  sp2->set_type(ModelProto::SentencePiece::CONTROL);
  sp2->set_piece("__BOS__");
  sp3->set_type(ModelProto::SentencePiece::CONTROL);
  sp3->set_piece("__EOS__");

  AddPiece(&model_proto, "a", 0.0);
  AddPiece(&model_proto, "b", 0.3);

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(model_proto).ok());
  EXPECT_EQ(0, sp.unk_id());
  EXPECT_EQ(1, sp.bos_id());
  EXPECT_EQ(2, sp.eos_id());
  EXPECT_EQ(-1, sp.pad_id());

  EXPECT_EQ("__UNK__", sp.IdToPiece(sp.unk_id()));
  EXPECT_EQ("__BOS__", sp.IdToPiece(sp.bos_id()));
  EXPECT_EQ("__EOS__", sp.IdToPiece(sp.eos_id()));
}

TEST(SentencePieceProcessorTest, SpecialPiecesCombinationsTest) {
  auto test_with_model = [](ModelProto::SentencePiece::Type type,
                            bool expect_ok) {
    ModelProto model_proto;
    auto* sp1 = model_proto.add_pieces();
    auto* sp2 = model_proto.add_pieces();
    auto* sp3 = model_proto.add_pieces();

    model_proto.mutable_trainer_spec()->set_unk_piece("<unk>");
    model_proto.mutable_trainer_spec()->set_bos_piece("<s>");
    model_proto.mutable_trainer_spec()->set_eos_piece("</s>");

    sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
    sp1->set_piece("<unk>");

    sp2->set_type(type);
    sp2->set_piece("<s>");
    sp3->set_type(type);
    sp3->set_piece("</s>");

    AddPiece(&model_proto, "a", 0.0);

    SentencePieceProcessor sp;
    EXPECT_TRUE(sp.Load(model_proto).ok());

    if (expect_ok) {
      EXPECT_TRUE(sp.SetEncodeExtraOptions("bos:eos").ok());
      std::vector<std::string> sps;
      EXPECT_TRUE(sp.Encode("a", &sps).ok());
      EXPECT_EQ(std::vector<std::string>({"<s>", WS, "a", "</s>"}), sps);
    } else {
      EXPECT_FALSE(sp.SetEncodeExtraOptions("bos").ok());
      EXPECT_FALSE(sp.SetEncodeExtraOptions("eos").ok());
    }
  };

  // CONTROL should succeed
  test_with_model(ModelProto::SentencePiece::CONTROL, true);
  // USER_DEFINED should fail
  test_with_model(ModelProto::SentencePiece::USER_DEFINED, false);
  // NORMAL should fail
  test_with_model(ModelProto::SentencePiece::NORMAL, false);
  // UNUSED should fail
  test_with_model(ModelProto::SentencePiece::UNUSED, false);

  // Missing BOS/EOS should fail
  {
    ModelProto model_proto;
    auto* sp1 = model_proto.add_pieces();
    sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
    sp1->set_piece("<unk>");
    AddPiece(&model_proto, "a", 0.0);

    SentencePieceProcessor sp;
    EXPECT_TRUE(sp.Load(model_proto).ok());
    EXPECT_FALSE(sp.SetEncodeExtraOptions("bos").ok());
    EXPECT_FALSE(sp.SetEncodeExtraOptions("eos").ok());
  }
}

TEST(LoadModelProtoTest, EmptyFilename) {
  ModelProto model_proto;
  const auto status = io::LoadModelProto("", &model_proto);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "model file path should not be empty."));
}

TEST(LoadModelProtoTest, FileNotParseableAsModelProto) {
  ModelProto model_proto;
  const std::string filename =
      filesystem::JoinPath(::testing::TempDir(), "file");
  {
    auto output = filesystem::NewWritableFile(filename, true);
    ASSERT_TRUE(output->Write("12345"));
  }

  const auto status = io::LoadModelProto(filename, &model_proto);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      absl::StrCat("could not parse ModelProto from ", filename)));
}

TEST(LoadModelProtoTest, FileLoadsOk) {
  ModelProto model_proto;
  AddPiece(&model_proto, "a", 0.0);
  AddPiece(&model_proto, "b", 0.3);
  AddPiece(&model_proto, "c", 0.2);

  const std::string filename =
      filesystem::JoinPath(::testing::TempDir(), "file");
  {
    auto output = filesystem::NewWritableFile(filename, true);
    ASSERT_TRUE(output->Write(model_proto.SerializeAsString()));
  }
  ABSL_EXPECT_OK(io::LoadModelProto(filename, &model_proto));
}

TEST(SentencePieceProcessorTest, UninitializedModelProtoSafeTest) {
  SentencePieceProcessor sp;
  EXPECT_EQ(0, sp.model_proto().pieces_size());
  EXPECT_EQ("", sp.serialized_model_proto());

  EXPECT_FALSE(sp.Load(std::unique_ptr<ModelProto>(nullptr)).ok());
  EXPECT_EQ(0, sp.model_proto().pieces_size());
}

TEST(SentencePieceProcessorTest, WordAndCharModelProtoPreservationTest) {
  auto MakeBaseModelProto = [](TrainerSpec::ModelType model_type) {
    ModelProto model_proto;
    *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
    model_proto.mutable_trainer_spec()->set_model_type(model_type);
    auto* unk = model_proto.add_pieces();
    unk->set_piece("<unk>");
    unk->set_type(ModelProto::SentencePiece::UNKNOWN);
    return model_proto;
  };

  // 1. WORD model round-trip via serialized_model_proto()
  {
    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::WORD);
    AddPiece(&model_proto, WS "hello", -1.0);
    AddPiece(&model_proto, "World", -2.0);
    AddPiece(&model_proto, WS "world", -3.0);

    SentencePieceProcessor sp1;
    ABSL_ASSERT_OK(sp1.Load(model_proto));
    EXPECT_EQ(TrainerSpec::WORD, sp1.model_proto().trainer_spec().model_type());

    SentencePieceProcessor sp2;
    ABSL_ASSERT_OK(sp2.LoadFromSerializedProto(sp1.serialized_model_proto()));
    EXPECT_EQ(TrainerSpec::WORD, sp2.model_proto().trainer_spec().model_type());

    std::vector<std::string> pieces1, pieces2;
    ABSL_ASSERT_OK(sp1.Encode("helloWorld hello world", &pieces1));
    ABSL_ASSERT_OK(sp2.Encode("helloWorld hello world", &pieces2));
    EXPECT_EQ(
        std::vector<std::string>({WS "helloWorld", WS "hello", WS "world"}),
        pieces1);
    EXPECT_EQ(pieces1, pieces2);

    // Also test ParallelEncode with chunk_len=1 on WORD model
    ThreadPool pool(4);
    std::vector<std::string> par_pieces;
    std::vector<int> ids1, par_ids;
    ABSL_ASSERT_OK(sp1.Encode("helloWorld hello world", &ids1));
    ABSL_ASSERT_OK(
        sp1.ParallelEncode("helloWorld hello world", 1, pool, &par_pieces));
    ABSL_ASSERT_OK(
        sp1.ParallelEncode("helloWorld hello world", 1, pool, &par_ids));
    EXPECT_EQ(pieces1, par_pieces);
    EXPECT_EQ(ids1, par_ids);
  }

  // 2. CHAR model round-trip and multi-char NORMAL piece rejection
  {
    ModelProto invalid_char_proto = MakeBaseModelProto(TrainerSpec::CHAR);
    AddPiece(&invalid_char_proto, WS, -1.0);
    AddPiece(&invalid_char_proto, "a", -2.0);
    AddPiece(&invalid_char_proto, "ab", -0.5);  // Invalid multi-char NORMAL
    SentencePieceProcessor sp_invalid;
    EXPECT_FALSE(sp_invalid.Load(invalid_char_proto).ok());

    ModelProto model_proto = MakeBaseModelProto(TrainerSpec::CHAR);
    AddPiece(&model_proto, WS, -1.0);
    AddPiece(&model_proto, "a", -2.0);
    AddPiece(&model_proto, "b", -3.0);

    SentencePieceProcessor sp1;
    ABSL_ASSERT_OK(sp1.Load(model_proto));
    EXPECT_EQ(TrainerSpec::CHAR, sp1.model_proto().trainer_spec().model_type());

    SentencePieceProcessor sp2;
    ABSL_ASSERT_OK(sp2.LoadFromSerializedProto(sp1.serialized_model_proto()));
    EXPECT_EQ(TrainerSpec::CHAR, sp2.model_proto().trainer_spec().model_type());

    std::vector<std::string> pieces1, pieces2;
    ABSL_ASSERT_OK(sp1.Encode("ab", &pieces1));
    ABSL_ASSERT_OK(sp2.Encode("ab", &pieces2));
    EXPECT_EQ(std::vector<std::string>({WS, "a", "b"}), pieces1);
    EXPECT_EQ(pieces1, pieces2);
  }
}

TEST(SentencePieceProcessorTest, ReloadClearsDenormalizerAndAcceptsValidModel) {
  auto make_reload_model = [](bool with_denormalizer) {
    auto model_proto = std::make_unique<ModelProto>();
    auto* unk = model_proto->add_pieces();
    unk->set_piece("<unk>");
    unk->set_type(ModelProto::SentencePiece::UNKNOWN);
    unk->set_score(0.0f);
    auto* piece_a = model_proto->add_pieces();
    piece_a->set_piece("a");
    piece_a->set_type(ModelProto::SentencePiece::NORMAL);
    piece_a->set_score(-1.0f);
    model_proto->mutable_trainer_spec()->set_model_type(TrainerSpec::UNIGRAM);
    model_proto->mutable_normalizer_spec()->set_name("identity");
    model_proto->mutable_normalizer_spec()->set_add_dummy_prefix(false);
    model_proto->mutable_normalizer_spec()->set_remove_extra_whitespaces(false);
    if (with_denormalizer) {
      std::string blob(1029, '\0');
      blob[1] = '\x04';
      blob[5] = '\x04';
      model_proto->mutable_denormalizer_spec()->set_precompiled_charsmap(blob);
      model_proto->mutable_denormalizer_spec()->set_add_dummy_prefix(false);
      model_proto->mutable_denormalizer_spec()->set_remove_extra_whitespaces(
          false);
    }
    return model_proto;
  };

  SentencePieceProcessor sp;
  std::string control_decoded;
  ASSERT_TRUE(sp.Load(make_reload_model(true)).ok());
  ASSERT_TRUE(sp.Decode(std::vector<std::string>{"a"}, &control_decoded).ok());
  EXPECT_EQ("a", control_decoded);

  std::string reloaded_decoded;
  ASSERT_TRUE(sp.Load(make_reload_model(false)).ok());
  ASSERT_TRUE(sp.Decode(std::vector<std::string>{"a"}, &reloaded_decoded).ok());
  EXPECT_EQ("a", reloaded_decoded);
}

TEST(SentencePieceProcessorTest, FailedReloadResetsCachedSpecialIds) {
  auto model_proto = std::make_unique<ModelProto>();
  auto add_piece = [&](absl::string_view piece,
                       ModelProto::SentencePiece::Type type) {
    auto* sp = model_proto->add_pieces();
    sp->set_piece(std::string(piece));
    sp->set_type(type);
    sp->set_score(0.0f);
  };
  add_piece("<unk>", ModelProto::SentencePiece::UNKNOWN);
  add_piece("<s>", ModelProto::SentencePiece::CONTROL);
  add_piece("</s>", ModelProto::SentencePiece::CONTROL);
  add_piece("a", ModelProto::SentencePiece::NORMAL);
  model_proto->mutable_trainer_spec()->set_model_type(TrainerSpec::UNIGRAM);
  model_proto->mutable_normalizer_spec()->set_name("identity");

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(std::move(model_proto)).ok());
  EXPECT_NE(-1, sp.unk_id());
  EXPECT_NE(-1, sp.bos_id());
  EXPECT_NE(-1, sp.eos_id());

  // A model without pieces fails to load. The ids cached from the previous
  // model must not survive.
  auto broken = std::make_unique<ModelProto>();
  EXPECT_FALSE(sp.Load(std::move(broken)).ok());
  EXPECT_EQ(-1, sp.unk_id());
  EXPECT_EQ(-1, sp.bos_id());
  EXPECT_EQ(-1, sp.eos_id());
  EXPECT_EQ(-1, sp.pad_id());
}

TEST(SentencePieceProcessorTest, DecodeIdsMatchesDecodePiecesTest) {
  auto make_proto = [](bool empty_unk_surface) {
    ModelProto model_proto;
    *model_proto.mutable_normalizer_spec() = MakeDefaultNormalizerSpec();
    model_proto.mutable_trainer_spec()->set_byte_fallback(true);
    if (empty_unk_surface) {
      model_proto.mutable_trainer_spec()->set_unk_surface("");
    }
    auto add_piece = [&](absl::string_view piece,
                         ModelProto::SentencePiece::Type type =
                             ModelProto::SentencePiece::NORMAL) {
      auto* sp = model_proto.add_pieces();
      sp->set_piece(std::string(piece));
      sp->set_type(type);
    };
    add_piece("<unk>", ModelProto::SentencePiece::UNKNOWN);
    add_piece("<s>", ModelProto::SentencePiece::CONTROL);
    add_piece("</s>", ModelProto::SentencePiece::CONTROL);
    for (int i = 0; i < 256; ++i) {
      add_piece(absl::StrFormat("<0x%02X>", i),
                ModelProto::SentencePiece::BYTE);
    }
    add_piece(WS "ABC");
    add_piece(WS "DE");
    add_piece("F");
    add_piece("G" WS "H");
    return model_proto;
  };

  const std::vector<std::vector<std::string>> inputs = {
      {},
      {"<unk>", WS "ABC", WS "DE"},
      {"<unk>", "<unk>", WS "DE"},
      {"<s>", WS "ABC", "<unk>", WS "DE", "F", "G" WS "H", "</s>"},
      {WS "ABC", "<0xE3>", "<0x81>", "<0x82>", "F"},  // U+3042
      {"<0xFF>", WS "DE", "<0xE3>", "<0x81>", "<unk>", "G" WS "H"},
  };

  // TODO(taku): Add `true` (an explicitly empty unk_surface) once the
  // Protobuf-to-FlatBuffer conversion preserves it. Decode(ids) currently
  // emits the default unk_surface for such a model while Decode(pieces) emits
  // nothing.
  for (const bool empty_unk_surface : {false}) {
    SentencePieceProcessor sp;
    ABSL_ASSERT_OK(sp.Load(make_proto(empty_unk_surface)));
    for (const char* options :
         {"", "bos", "eos", "reverse", "bos:eos", "reverse:bos:eos"}) {
      ABSL_ASSERT_OK(sp.SetDecodeExtraOptions(options));
      for (const auto& input : inputs) {
        std::vector<int> ids;
        for (const auto& piece : input) {
          ids.push_back(sp.PieceToId(piece));
          if (piece != "<unk>") {
            ASSERT_NE(ids.back(), sp.unk_id()) << piece;
          }
        }
        SentencePieceText from_pieces, from_ids;
        const std::vector<absl::string_view> views(input.begin(), input.end());
        ABSL_ASSERT_OK(sp.Decode(views, &from_pieces));
        ABSL_ASSERT_OK(sp.Decode(ids, &from_ids));
        const std::string context = absl::StrCat(
            "options=", options, " empty_unk_surface=", empty_unk_surface,
            " input=", absl::StrJoin(input, " "));
        EXPECT_EQ(from_pieces.text(), from_ids.text()) << context;
        ASSERT_EQ(from_pieces.pieces_size(), from_ids.pieces_size()) << context;
        for (int i = 0; i < from_ids.pieces_size(); ++i) {
          const auto& a = from_pieces.pieces(i);
          const auto& b = from_ids.pieces(i);
          EXPECT_EQ(a.piece(), b.piece()) << context << " i=" << i;
          EXPECT_EQ(a.id(), b.id()) << context << " i=" << i;
          EXPECT_EQ(a.surface(), b.surface()) << context << " i=" << i;
          EXPECT_EQ(a.begin(), b.begin()) << context << " i=" << i;
          EXPECT_EQ(a.end(), b.end()) << context << " i=" << i;
        }
        std::string text;
        ABSL_ASSERT_OK(sp.Decode(ids, &text));
        EXPECT_EQ(from_ids.text(), text) << context;
      }
    }
  }
}

// The tests below port the scenarios of the runtime unit tests that existed in
// v0.2.3 (unigram_model_test.cc, bpe_model_test.cc, char_model_test.cc,
// word_model_test.cc, and model_interface_test.cc) to the public
// SentencePieceProcessor API.
namespace {

ModelProto MakeRuntimeBaseProto(TrainerSpec::ModelType model_type,
                                bool byte_fallback = false) {
  ModelProto model_proto;
  auto add = [&model_proto](const char* piece,
                            ModelProto::SentencePiece::Type type) {
    auto* sp = model_proto.add_pieces();
    sp->set_piece(piece);
    sp->set_type(type);
  };
  add("<unk>", ModelProto::SentencePiece::UNKNOWN);
  add("<s>", ModelProto::SentencePiece::CONTROL);
  add("</s>", ModelProto::SentencePiece::CONTROL);
  model_proto.mutable_trainer_spec()->set_model_type(model_type);
  if (byte_fallback) {
    model_proto.mutable_trainer_spec()->set_byte_fallback(true);
  }
  // Identity normalization, so that the input reaches the model unchanged.
  auto* normalizer_spec = model_proto.mutable_normalizer_spec();
  normalizer_spec->set_name("identity");
  normalizer_spec->set_add_dummy_prefix(false);
  normalizer_spec->set_remove_extra_whitespaces(false);
  normalizer_spec->set_escape_whitespaces(false);
  return model_proto;
}

void AddRuntimePiece(
    ModelProto* model_proto, absl::string_view piece, float score = 0.0,
    ModelProto::SentencePiece::Type type = ModelProto::SentencePiece::NORMAL) {
  auto* sp = model_proto->add_pieces();
  sp->set_piece(std::string(piece));
  sp->set_score(score);
  if (type != ModelProto::SentencePiece::NORMAL) sp->set_type(type);
}

void AddRuntimeBytePieces(ModelProto* model_proto, int size) {
  for (int i = 0; i < size; ++i) {
    AddRuntimePiece(model_proto, absl::StrFormat("<0x%02X>", i), 0.0,
                    ModelProto::SentencePiece::BYTE);
  }
}

void ExpectEncode(const ModelProto& model_proto, absl::string_view input,
                  const std::vector<std::string>& expected_pieces,
                  const std::vector<int>& expected_ids) {
  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(model_proto));
  std::vector<std::string> pieces;
  std::vector<int> ids;
  ABSL_ASSERT_OK(sp.Encode(input, &pieces));
  ABSL_ASSERT_OK(sp.Encode(input, &ids));
  EXPECT_EQ(expected_pieces, pieces) << input;
  EXPECT_EQ(expected_ids, ids) << input;
}

}  // namespace

TEST(SentencePieceProcessorTest, UnigramEncodeWithUnusedTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM);
  AddRuntimePiece(&model_proto, "abcd", 10.0);  // 3
  AddRuntimePiece(&model_proto, "abc", 5.0);    // 4
  AddRuntimePiece(&model_proto, "ab", 2.0);     // 5
  AddRuntimePiece(&model_proto, "cd", 1.0);     // 6
  AddRuntimePiece(&model_proto, "a");           // 7
  AddRuntimePiece(&model_proto, "b");           // 8
  AddRuntimePiece(&model_proto, "c");           // 9
  AddRuntimePiece(&model_proto, "d");           // 10

  ExpectEncode(model_proto, "abcd", {"abcd"}, {3});

  model_proto.mutable_pieces(3)->set_type(ModelProto::SentencePiece::UNUSED);
  ExpectEncode(model_proto, "abcd", {"abc", "d"}, {4, 10});

  model_proto.mutable_pieces(5)->set_type(ModelProto::SentencePiece::UNUSED);
  ExpectEncode(model_proto, "abcd", {"abc", "d"}, {4, 10});

  // Unigram finds the best path without the unused nodes.
  model_proto.mutable_pieces(4)->set_type(ModelProto::SentencePiece::UNUSED);
  model_proto.mutable_pieces(5)->set_type(ModelProto::SentencePiece::NORMAL);
  ExpectEncode(model_proto, "abcd", {"ab", "cd"}, {5, 6});
}

TEST(SentencePieceProcessorTest, ControlAndUnknownSymbolsAreNotMatchedTest) {
  for (const auto model_type : {TrainerSpec::UNIGRAM, TrainerSpec::BPE}) {
    const bool is_bpe = model_type == TrainerSpec::BPE;
    ModelProto model_proto = MakeRuntimeBaseProto(model_type);
    AddRuntimePiece(&model_proto, "<", -1.0);              // 3
    AddRuntimePiece(&model_proto, "s", -1.0);              // 4
    AddRuntimePiece(&model_proto, ">", -1.0);              // 5
    AddRuntimePiece(&model_proto, "<s", 0.0);              // 6
    AddRuntimePiece(&model_proto, "u", -1.0);              // 7
    AddRuntimePiece(&model_proto, "n", -1.0);              // 8
    AddRuntimePiece(&model_proto, "k", -1.0);              // 9
    AddRuntimePiece(&model_proto, "<u", 0.0);              // 10
    if (is_bpe) AddRuntimePiece(&model_proto, "nk", 0.0);  // 11
    AddRuntimePiece(&model_proto, "nk>", 0.0);             // 11 or 12

    // "<s>" and "<unk>" in the input are not matched as CONTROL/UNKNOWN.
    ExpectEncode(model_proto, "<s>", {"<s", ">"}, {6, 5});
    ExpectEncode(model_proto, "<unk>", {"<u", "nk>"}, {10, is_bpe ? 12 : 11});
  }
}

TEST(SentencePieceProcessorTest, UnigramSpecialSymbolsNoSegmentTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM, true);
  AddRuntimeBytePieces(&model_proto, 256);   // 3..258
  AddRuntimePiece(&model_proto, "a", -1.0);  // 259
  AddRuntimePiece(&model_proto, "b", -1.0);  // 260
  AddRuntimePiece(&model_proto, "c", -1.0);  // 261
  AddRuntimePiece(&model_proto, "abc", 0.0,
                  ModelProto::SentencePiece::UNUSED);  // 262
  for (const char* c : {"<", "u", "n", "k", "s", ">"}) {
    AddRuntimePiece(&model_proto, c, -1.0);  // 263..268
  }
  AddRuntimePiece(&model_proto, "<s", 0.0);  // 269
  AddRuntimePiece(&model_proto, "<u", 0.0);  // 270
  AddRuntimePiece(&model_proto, "nk", 0.0);  // 271
  for (const char* c : {"0", "x", "A"}) {
    AddRuntimePiece(&model_proto, c, -1.0);  // 272..274
  }
  AddRuntimePiece(&model_proto, "<0", 0.0);  // 275
  AddRuntimePiece(&model_proto, "x0", 0.0);  // 276
  AddRuntimePiece(&model_proto, "A>", 0.0);  // 277

  // CONTROL, UNKNOWN, UNUSED, and BYTE pieces are not matched in the input.
  ExpectEncode(model_proto, "<s>", {"<s", ">"}, {269, 268});
  ExpectEncode(model_proto, "<unk>", {"<u", "nk", ">"}, {270, 271, 268});
  ExpectEncode(model_proto, "abc", {"a", "b", "c"}, {259, 260, 261});
  ExpectEncode(model_proto, "<0x0A>", {"<0", "x0", "A>"}, {275, 276, 277});
}

TEST(SentencePieceProcessorTest, UnigramLongTextScoreRecenteringTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM);
  AddRuntimePiece(&model_proto, "ab", -10.0);
  AddRuntimePiece(&model_proto, "cd", -10.0);
  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(model_proto));

  // 30,000 "ab" pairs accumulate a score of -300,000.
  std::string input;
  for (int i = 0; i < 30000; ++i) input += "ab";
  std::vector<std::string> pieces;
  ABSL_ASSERT_OK(sp.Encode(input, &pieces));
  ASSERT_EQ(30000, pieces.size());
  for (const auto& piece : pieces) ASSERT_EQ("ab", piece);
}

TEST(SentencePieceProcessorTest, PieceToIdEmptyPieceTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM);
  AddRuntimePiece(&model_proto, "a", 0.1);   // 3
  AddRuntimePiece(&model_proto, "b", 0.2);   // 4
  AddRuntimePiece(&model_proto, "ab", 0.3);  // 5
  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(model_proto));

  // A zero-length view resolves to <unk> even when data() points at a key.
  EXPECT_EQ(0, sp.PieceToId(""));
  EXPECT_EQ(0, sp.PieceToId(absl::string_view("ab", 0)));
  EXPECT_EQ(0, sp.PieceToId(absl::string_view("a", 0)));
  EXPECT_EQ(3, sp.PieceToId("a"));
  EXPECT_EQ(5, sp.PieceToId("ab"));
}

TEST(SentencePieceProcessorTest, BpeEncodeAmbiguousTest) {
  // v0.2.3 used the scores -0.1 .. -0.5. BPE scores are truncated to integer
  // merge ranks in 0.3.0, so the same order is expressed with integers.
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::BPE);
  AddRuntimePiece(&model_proto, "aa", -1.0);  // 3
  AddRuntimePiece(&model_proto, "bb", -2.0);  // 4
  AddRuntimePiece(&model_proto, "ab", -3.0);  // 5
  AddRuntimePiece(&model_proto, "a", -4.0);   // 6
  AddRuntimePiece(&model_proto, "b", -5.0);   // 7

  // Leftmost symbols are merged first.
  ExpectEncode(model_proto, "aaa", {"aa", "a"}, {3, 6});
  // "bb" is merged earlier than "ab".
  ExpectEncode(model_proto, "aabb", {"aa", "bb"}, {3, 4});
  ExpectEncode(model_proto, "aaabbb", {"aa", "a", "bb", "b"}, {3, 6, 4, 7});
  ExpectEncode(model_proto, "aaaba", {"aa", "ab", "a"}, {3, 5, 6});
  // A broken UTF-8 byte is encoded as <unk> with U+FFFD as its surface.
  ExpectEncode(model_proto, "\xE3", {"\xEF\xBF\xBD"}, {0});
}

TEST(SentencePieceProcessorTest, BpeEncodeWithUnusedTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::BPE);
  AddRuntimePiece(&model_proto, "abcd", 10.0);  // 3
  AddRuntimePiece(&model_proto, "abc", 5.0);    // 4
  AddRuntimePiece(&model_proto, "ab", 2.0);     // 5
  AddRuntimePiece(&model_proto, "cd", 1.0);     // 6
  AddRuntimePiece(&model_proto, "a", 0.0);      // 7
  AddRuntimePiece(&model_proto, "b", 0.0);      // 8
  AddRuntimePiece(&model_proto, "c", 0.0);      // 9
  AddRuntimePiece(&model_proto, "d", 0.0);      // 10

  ExpectEncode(model_proto, "abcd", {"abcd"}, {3});

  model_proto.mutable_pieces(3)->set_type(ModelProto::SentencePiece::UNUSED);
  ExpectEncode(model_proto, "abcd", {"abc", "d"}, {4, 10});

  // Intermediate UNUSED piece "ab" still merges to form NORMAL parent "abc".
  model_proto.mutable_pieces(5)->set_type(ModelProto::SentencePiece::UNUSED);
  ExpectEncode(model_proto, "abcd", {"abc", "d"}, {4, 10});

  // Both "abcd" and "abc" are UNUSED while "ab" and "cd" are NORMAL:
  // merges to "abcd" (blocking "cd") and resegments to ["ab", "c", "d"].
  model_proto.mutable_pieces(4)->set_type(ModelProto::SentencePiece::UNUSED);
  model_proto.mutable_pieces(5)->set_type(ModelProto::SentencePiece::NORMAL);
  ExpectEncode(model_proto, "abcd", {"ab", "c", "d"}, {5, 9, 10});
}

TEST(SentencePieceProcessorTest, BpeDeepUnusedMergeChainTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::BPE);
  constexpr int kDepth = 101;

  std::vector<std::string> chars;
  for (int i = 0; i <= kDepth + 1; ++i) {
    chars.push_back(string_util::UnicodeCharToUTF8(0x1000 + i));
    AddRuntimePiece(&model_proto, chars.back(), 0.0);
  }
  const int merge_start_index = model_proto.pieces_size();
  std::string merged = chars[0] + chars[1];
  AddRuntimePiece(&model_proto, merged, -1.0);
  for (int i = 1; i <= kDepth; ++i) {
    merged += chars[i + 1];
    AddRuntimePiece(&model_proto, merged, -static_cast<float>(i + 1));
  }
  for (int i = 0; i <= kDepth; ++i) {
    model_proto.mutable_pieces(merge_start_index + i)
        ->set_type(ModelProto::SentencePiece::UNUSED);
  }

  SentencePieceProcessor sp;
  ABSL_ASSERT_OK(sp.Load(model_proto));
  std::vector<std::string> pieces;
  std::vector<int> ids;
  const std::string input = absl::StrJoin(chars, "");
  ABSL_ASSERT_OK(sp.Encode(input, &pieces));
  ABSL_ASSERT_OK(sp.Encode(input, &ids));

  // Resegmentation stops at depth 100 (`kMaxBpeResegmentDepth`) and maps the
  // remaining un-resegmented UNUSED piece to <unk> (whereas v0.2.3 emitted the
  // invisible UNUSED token ID).
  ASSERT_EQ(static_cast<size_t>(kDepth + 1), pieces.size());
  ASSERT_EQ(pieces.size(), ids.size());
  EXPECT_EQ(sp.unk_id(), ids[0]);
  EXPECT_EQ(chars[0] + chars[1], pieces[0]);
  for (int i = 1; i <= kDepth; ++i) {
    EXPECT_EQ(chars[i + 1], pieces[i]);
  }
}

TEST(SentencePieceProcessorTest, CharEncodeReservedAndUnusedTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::CHAR);
  AddRuntimePiece(&model_proto, "a", 0.1);  // 3
  AddRuntimePiece(&model_proto, "x", 0.0, ModelProto::SentencePiece::CONTROL);
  AddRuntimePiece(&model_proto, "y", 0.0, ModelProto::SentencePiece::UNUSED);

  // CONTROL and UNUSED characters are encoded as <unk>, and consecutive
  // unknowns are merged.
  ExpectEncode(model_proto, "axy", {"a", "xy"}, {3, 0});
  ExpectEncode(model_proto, "xay", {"x", "a", "y"}, {0, 3, 0});
}

TEST(SentencePieceProcessorTest, WordEncodeReservedAndUnusedTest) {
  ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::WORD);
  AddRuntimePiece(&model_proto, WS "a", 0.1);  // 3
  AddRuntimePiece(&model_proto, WS "<s>", 0.0,
                  ModelProto::SentencePiece::CONTROL);
  AddRuntimePiece(&model_proto, WS "unused", 0.0,
                  ModelProto::SentencePiece::UNUSED);

  ExpectEncode(model_proto, WS "a" WS "<s>" WS "unused",
               {WS "a", WS "<s>" WS "unused"}, {3, 0});
  ExpectEncode(model_proto, WS "<s>" WS "a" WS "unused",
               {WS "<s>", WS "a", WS "unused"}, {0, 3, 0});
}

TEST(SentencePieceProcessorTest, ByteFallbackModelValidationTest) {
  {
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM, true);
    AddRuntimeBytePieces(&model_proto, 256);
    AddRuntimePiece(&model_proto, "a");
    SentencePieceProcessor sp;
    EXPECT_TRUE(sp.Load(model_proto).ok());
  }
  {  // `byte_fallback` is true, but there are not 256 byte pieces.
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM, true);
    AddRuntimeBytePieces(&model_proto, 10);
    AddRuntimePiece(&model_proto, "a");
    SentencePieceProcessor sp;
    EXPECT_FALSE(sp.Load(model_proto).ok());
  }
  {  // `byte_fallback` is false, but a byte piece is found.
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM);
    AddRuntimeBytePieces(&model_proto, 10);
    AddRuntimePiece(&model_proto, "a");
    SentencePieceProcessor sp;
    EXPECT_FALSE(sp.Load(model_proto).ok());
  }
}

TEST(SentencePieceProcessorTest, PieceToIdStressTest) {
  uint32_t state = 12345;
  auto next = [&state]() {
    state = state * 1103515245u + 12345u;
    return (state >> 16) & 0x7fff;
  };
  for (const auto model_type :
       {TrainerSpec::UNIGRAM, TrainerSpec::BPE, TrainerSpec::WORD}) {
    for (int iter = 0; iter < 5; ++iter) {
      ModelProto model_proto = MakeRuntimeBaseProto(model_type);
      absl::flat_hash_map<std::string, int> expected;
      for (int n = 0; n < 1000; ++n) {
        std::string piece;
        for (int k = 0; k < 10; ++k) {
          piece += static_cast<char>('a' + next() % 26);
        }
        if (!expected.emplace(piece, model_proto.pieces_size()).second) {
          continue;
        }
        AddRuntimePiece(&model_proto, piece);
      }
      SentencePieceProcessor sp;
      ABSL_ASSERT_OK(sp.Load(model_proto));
      for (const auto& [piece, id] : expected) {
        ASSERT_EQ(id, sp.PieceToId(piece));
        ASSERT_EQ(piece, sp.IdToPiece(id));
      }
    }
  }
}

TEST(SentencePieceProcessorTest, CharModelRejectsMultiCharPieceTest) {
  {
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::CHAR);
    AddRuntimePiece(&model_proto, "a");
    AddRuntimePiece(&model_proto, "\xE3\x81\x82");  // U+3042
    SentencePieceProcessor sp;
    EXPECT_TRUE(sp.Load(model_proto).ok());
  }
  {  // v0.2.3 loaded a CHAR model with a multi-character NORMAL piece.
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::CHAR);
    AddRuntimePiece(&model_proto, "ab");
    SentencePieceProcessor sp;
    EXPECT_FALSE(sp.Load(model_proto).ok());
  }
}

TEST(SentencePieceProcessorTest, TooLongPieceModelTest) {
  {
    ModelProto model_proto;
    model_proto.add_pieces()->set_piece(std::string(10000, 'a'));
    SentencePieceProcessor sp;
    EXPECT_FALSE(sp.Load(model_proto).ok());
  }
  // The limit is 4096 bytes (v0.2.3 accepted up to 7999 bytes).
  for (const int length : {4096, 4097, 10000}) {
    ModelProto model_proto = MakeRuntimeBaseProto(TrainerSpec::UNIGRAM);
    AddRuntimePiece(&model_proto, std::string(length, 'a'));
    SentencePieceProcessor sp;
    EXPECT_EQ(length <= 4096, sp.Load(model_proto).ok()) << length;
  }
}

}  // namespace sentencepiece
