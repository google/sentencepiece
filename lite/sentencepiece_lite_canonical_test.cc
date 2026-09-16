// Copyright 2026 Google LLC
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
// limitations under the License.

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"
#include "sentencepiece_processor.h"

namespace sentencepiece::lite {
namespace {

// Regression test: Decode must strip leading ▁ for suffix-whitespace models
// when add_dummy_prefix is true. Previously, lite's Decode incorrectly skipped
// bos_ws stripping for suffix models, producing extra leading spaces compared
// to canonical sentencepiece. This test pins lite Decode output against the
// canonical implementation.
TEST(SentencePieceLiteDecodeRegressionTest,
     DecodeStripsLeadingSpaceForSuffixModels) {
  // Construct a minimal Unigram model with treat_whitespace_as_suffix=true,
  // add_dummy_prefix=true, remove_extra_whitespaces=true.
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);
  trainer_spec->set_treat_whitespace_as_suffix(true);

  auto* normalizer_spec = proto.mutable_normalizer_spec();
  normalizer_spec->set_name("identity");
  normalizer_spec->set_add_dummy_prefix(true);
  normalizer_spec->set_remove_extra_whitespaces(true);
  normalizer_spec->set_escape_whitespaces(true);

  // Minimal vocabulary: <unk>, ▁, a, b, ▁a, ▁b
  auto add_piece = [&proto](
                       const char* piece, float score,
                       ::sentencepiece::ModelProto::SentencePiece::Type type =
                           ::sentencepiece::ModelProto::SentencePiece::NORMAL) {
    auto* p = proto.add_pieces();
    p->set_piece(piece);
    p->set_score(score);
    p->set_type(type);
  };
  add_piece("<unk>", 0,
            ::sentencepiece::ModelProto::SentencePiece::UNKNOWN);  // id 0
  add_piece("\xe2\x96\x81", 0);                                    // id 1: ▁
  add_piece("a", -1);                                              // id 2
  add_piece("b", -1);                                              // id 3
  add_piece(
      "\xe2\x96\x81"
      "a",
      -0.5);  // id 4: ▁a
  add_piece(
      "\xe2\x96\x81"
      "b",
      -0.5);  // id 5: ▁b

  // Set up lite processor.
  auto fbs_bytes_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(fbs_bytes_or.ok()) << fbs_bytes_or.status();
  std::string fb_data = std::move(*fbs_bytes_or);
  SentencePieceLiteProcessor lite(fb_data);
  ASSERT_EQ(lite.status(), StatusCode::kOk);

  // Set up canonical processor from the same proto.
  ::sentencepiece::SentencePieceProcessor canonical;
  ASSERT_TRUE(
      canonical.LoadFromSerializedProto(proto.SerializeAsString()).ok());

  // Test cases: various ID sequences that exercise the bos_ws stripping path.
  const std::vector<std::vector<int>> test_id_sequences = {
      {1},              // single ▁
      {1, 1},           // consecutive ▁
      {1, 1, 1, 4},     // ▁, ▁, ▁, ▁a
      {4, 5},           // ▁a, ▁b
      {1, 4, 5},        // ▁, ▁a, ▁b
      {2, 3},           // a, b (no ▁ prefix)
      {1, 1, 1, 1, 2},  // many ▁ then a
      {0},              // <unk>
      {1, 0},           // ▁, <unk>
  };

  for (const auto& ids : test_id_sequences) {
    SCOPED_TRACE(testing::PrintToString(ids));

    std::string lite_decoded;
    ASSERT_EQ(lite.Decode(ids, &lite_decoded), StatusCode::kOk);

    std::string canonical_decoded;
    ASSERT_TRUE(canonical.Decode(ids, &canonical_decoded).ok());

    EXPECT_EQ(lite_decoded, canonical_decoded)
        << "Lite Decode diverges from canonical for ids="
        << testing::PrintToString(ids);
  }
}

// Empty input, and input that is entirely whitespace once leading spaces are
// stripped, leave Normalize through early returns that bypass the offset
// bookkeeping of the main path. Both the output and the offset map are pinned
// against canonical sentencepiece, since the two runtimes are expected to agree
// byte for byte and the early returns are easy to get wrong in isolation.
TEST(SentencePieceLiteCanonicalTest, EmptyAndWhitespaceInputsMatchCanonical) {
  // With remove_extra_whitespaces the leading-space loop consumes the whole
  // input and returns early; without it the same input reaches the main path.
  for (const bool remove_extra_whitespaces : {true, false}) {
    SCOPED_TRACE(remove_extra_whitespaces ? "remove_extra_whitespaces"
                                          : "keep_extra_whitespaces");

    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);

    auto* normalizer_spec = proto.mutable_normalizer_spec();
    normalizer_spec->set_name("identity");
    normalizer_spec->set_add_dummy_prefix(true);
    normalizer_spec->set_remove_extra_whitespaces(remove_extra_whitespaces);
    normalizer_spec->set_escape_whitespaces(true);

    auto add_piece =
        [&proto](const char* piece, float score,
                 ::sentencepiece::ModelProto::SentencePiece::Type type =
                     ::sentencepiece::ModelProto::SentencePiece::NORMAL) {
          auto* p = proto.add_pieces();
          p->set_piece(piece);
          p->set_score(score);
          p->set_type(type);
        };
    add_piece("<unk>", 0,
              ::sentencepiece::ModelProto::SentencePiece::UNKNOWN);  // id 0
    add_piece("\xe2\x96\x81", 0);                                    // id 1: ▁
    add_piece("a", -1);                                              // id 2

    auto fbs_bytes_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(fbs_bytes_or.ok()) << fbs_bytes_or.status();
    std::string fb_data = std::move(*fbs_bytes_or);
    SentencePieceLiteProcessor lite(fb_data);
    ASSERT_EQ(lite.status(), StatusCode::kOk);

    ::sentencepiece::SentencePieceProcessor canonical;
    ASSERT_TRUE(
        canonical.LoadFromSerializedProto(proto.SerializeAsString()).ok());

    const std::vector<std::string> kInputs = {
        "",
        " ",
        "   ",
        "\xe2\x96\x81",  // the space symbol itself
        " a",
    };

    for (const std::string& input : kInputs) {
      SCOPED_TRACE(testing::PrintToString(input));

      std::string lite_normalized;
      std::vector<size_t> lite_offsets;
      ASSERT_EQ(lite.Normalize(input, &lite_normalized, &lite_offsets),
                StatusCode::kOk);

      std::string canonical_normalized;
      std::vector<size_t> canonical_offsets;
      ASSERT_TRUE(
          canonical.Normalize(input, &canonical_normalized, &canonical_offsets)
              .ok());

      EXPECT_EQ(lite_normalized, canonical_normalized);
      EXPECT_EQ(lite_offsets, canonical_offsets);

      std::vector<int> lite_ids;
      ASSERT_EQ(lite.Encode(input, &lite_ids), StatusCode::kOk);

      std::vector<int> canonical_ids;
      ASSERT_TRUE(canonical.Encode(input, &canonical_ids).ok());

      EXPECT_EQ(lite_ids, canonical_ids);

      std::string lite_decoded;
      ASSERT_EQ(lite.Decode(lite_ids, &lite_decoded), StatusCode::kOk);

      std::string canonical_decoded;
      ASSERT_TRUE(canonical.Decode(canonical_ids, &canonical_decoded).ok());

      EXPECT_EQ(lite_decoded, canonical_decoded);
    }
  }
}

}  // namespace
}  // namespace sentencepiece::lite
