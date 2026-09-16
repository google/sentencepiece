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

#include "sentencepiece_lite.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "flatbuffers/flatbuffers.h"
#include "sentencepiece_lite_generated.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"

namespace sentencepiece::lite {
namespace {

std::string GetFilePath(std::string_view path) {
  static constexpr std::string_view kPrefixes[] = {
      "",
      "data/",
      "../data/",
      "../../data/",
      "test_data/",
      "../test_data/",
      "../../test_data/",
      "lite/test_data/",
      "../lite/test_data/",
      "test_data/",
      "../test_data/",
  };
  for (const auto prefix : kPrefixes) {
    const std::string candidate = std::string(prefix) + std::string(path);
    if (std::ifstream(candidate, std::ios::binary).good()) {
      return candidate;
    }
  }
  const size_t last_slash = path.rfind('/');
  const std::string_view basename = (last_slash != std::string_view::npos)
                                        ? path.substr(last_slash + 1)
                                        : path;
  for (const auto prefix : kPrefixes) {
    const std::string candidate = std::string(prefix) + std::string(basename);
    if (std::ifstream(candidate, std::ios::binary).good()) {
      return candidate;
    }
  }
  std::string alias = std::string(basename);
  if (basename == "test_oss_model.model")
    alias = "botchan_en_unigram_1000.model";
  if (basename == "botchan_1000_bpe.model") alias = "botchan_en_bpe_1000.model";
  if (basename == "wagahaiwa_nekodearu_2000_bpe_byte.model")
    alias = "wagahaiwa_nekodearu_ja_bpe_byte_2000.model";

  for (const auto prefix : kPrefixes) {
    const std::string candidate = std::string(prefix) + alias;
    if (std::ifstream(candidate, std::ios::binary).good()) {
      return candidate;
    }
  }
  return std::string(path);
}

std::string ReadFile(std::string_view path) {
  const std::string actual_path = GetFilePath(path);
  std::ifstream input(actual_path, std::ios::binary);
  EXPECT_TRUE(input) << "Failed to open file: " << path << " (resolved to "
                     << actual_path << ")";
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

class SentencePieceLiteTest
    : public ::testing::TestWithParam<std::string_view> {
 protected:
  void SetUp() override {
    std::string model_path;
    if (GetParam() == "unigram") {
      model_path = "test_data/test_oss_model.model";
    } else if (GetParam() == "bpe") {
      model_path = "test_data/botchan_1000_bpe.model";
    } else {
      model_path =
          "test_data/"
          "wagahaiwa_nekodearu_2000_bpe_byte.model";
    }

    std::string model_bytes = ReadFile(model_path);
    ::sentencepiece::ModelProto proto;
    ASSERT_TRUE(proto.ParseFromString(model_bytes));
    proto_ = proto;
    auto fbs_bytes_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(fbs_bytes_or.ok()) << fbs_bytes_or.status();
    lite_model_data_ = std::move(*fbs_bytes_or);
    lite_processor_ =
        std::make_unique<sentencepiece::lite::SentencePieceLiteProcessor>(
            lite_model_data_);
    ASSERT_EQ(lite_processor_->status(), StatusCode::kOk);
  }

  ::sentencepiece::ModelProto proto_;
  std::string lite_model_data_;
  std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor>
      lite_processor_;
  const std::vector<std::string> test_inputs_ = {
      "hello world",
      "Wagahai wa Neko de Aru.",
      "this is a test.",
      "short",
      "",
      "  multiple   spaces  ",
      "🐱 hello 🐱",
      "∑ equation ∑",
  };
};

TEST_P(SentencePieceLiteTest, EncodeDecodeIdempotency) {
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);

    std::vector<int> lite_ids;
    ASSERT_EQ(lite_processor_->Encode(input, &lite_ids), StatusCode::kOk);

    std::string lite_decoded;
    ASSERT_EQ(lite_processor_->Decode(lite_ids, &lite_decoded),
              StatusCode::kOk);

    const bool byte_fallback = (lite_processor_->PieceToId("<0x00>") != -1);
    const int unk_id = lite_processor_->PieceToId("<unk>");
    if (!byte_fallback &&
        std::find(lite_ids.begin(), lite_ids.end(), unk_id) != lite_ids.end()) {
      continue;
    }

    std::vector<int> re_ids;
    ASSERT_EQ(lite_processor_->Encode(lite_decoded, &re_ids), StatusCode::kOk);
    EXPECT_EQ(re_ids, lite_ids);

    std::string re_decoded;
    ASSERT_EQ(lite_processor_->Decode(re_ids, &re_decoded), StatusCode::kOk);
    EXPECT_EQ(re_decoded, lite_decoded);
  }
}

TEST_P(SentencePieceLiteTest, NormalizeAndOffsetSemanticsTest) {
  // 1. Verify general monotonicity and boundary invariants for all inputs.
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);
    std::string norm;
    std::vector<size_t> offsets;
    ASSERT_EQ(lite_processor_->Normalize(input, &norm, &offsets),
              StatusCode::kOk);
    if (norm.empty()) {
      EXPECT_TRUE(offsets.empty());
      continue;
    }
    ASSERT_EQ(offsets.size(), norm.size() + 1);
    EXPECT_LE(offsets[0], input.size());
    for (size_t i = 0; i < norm.size(); ++i) {
      EXPECT_LE(offsets[i], offsets[i + 1]) << "Offsets must be monotonic.";
      EXPECT_LE(offsets[i + 1], input.size())
          << "Offsets must not exceed input size.";
    }
  }

  // 2. Explicitly verify semantic correctness of non-trivial normalization
  // (character expansion, fullwidth-to-halfwidth conversion, and whitespace
  // reduction) by reversing the offset mapping to extract and match original
  // substrings, and verifying independent span normalization equivalence.
  auto verify_substring_mapping = [&](std::string_view input,
                                      std::string_view target_norm_substr,
                                      std::string_view expected_orig_substr) {
    std::string norm;
    std::vector<size_t> offsets;
    ASSERT_EQ(lite_processor_->Normalize(input, &norm, &offsets),
              StatusCode::kOk);
    const size_t pos = norm.find(target_norm_substr);
    ASSERT_NE(pos, std::string::npos)
        << "Normalized text \"" << norm << "\" must contain \""
        << target_norm_substr << "\"";
    const size_t orig_begin = offsets[pos];
    const size_t orig_end = offsets[pos + target_norm_substr.size()];
    ASSERT_LE(orig_end, input.size());
    const std::string_view orig_span =
        input.substr(orig_begin, orig_end - orig_begin);
    EXPECT_EQ(orig_span, expected_orig_substr);

    std::string renorm_orig;
    ASSERT_EQ(lite_processor_->Normalize(orig_span, &renorm_orig),
              StatusCode::kOk);
    std::string_view renorm_view = renorm_orig;
    if (!target_norm_substr.starts_with("\xE2\x96\x81") &&
        renorm_view.starts_with("\xE2\x96\x81")) {
      renorm_view.remove_prefix(3);
    }
    if (renorm_view.empty() && target_norm_substr == "\xE2\x96\x81") {
      renorm_view = target_norm_substr;
    }
    EXPECT_NE(renorm_view.find(target_norm_substr), std::string_view::npos)
        << "Independently normalizing the original span must yield or "
           "contain the normalized span.";
  };

  // Case A: 1-to-2 character expansion ("㍻" -> "平成").
  verify_substring_mapping("㍻元年", "平成", "㍻");
  verify_substring_mapping("㍻元年", "元年", "元年");

  // Case B: Fullwidth-to-halfwidth conversion and space reduction
  // ("ＫＡＤＯＫＡＷＡ").
  verify_substring_mapping("ＫＡＤＯＫＡＷＡ   ABC ", "KADOKAWA",
                           "ＫＡＤＯＫＡＷＡ");
  verify_substring_mapping("ＫＡＤＯＫＡＷＡ   ABC ", "ABC", "ABC");

  // Case C: Leading, trailing, and multiple consecutive whitespace reduction.
  verify_substring_mapping("  multiple   spaces  ", "multiple", "multiple");
  verify_substring_mapping("  multiple   spaces  ", "spaces", "spaces");

  // 3. Verify span normalization equivalence across all tokens for all inputs:
  // For every token piece in the normalized text, extracting its corresponding
  // original span via offset and independently normalizing both spans must
  // yield identical normalized results (Normalize(orig_span) ==
  // Normalize(norm_span)).
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);
    std::string norm;
    std::vector<size_t> offsets;
    ASSERT_EQ(lite_processor_->Normalize(input, &norm, &offsets),
              StatusCode::kOk);
    if (norm.empty()) {
      continue;
    }

    std::vector<int> ids;
    std::vector<std::string_view> pieces;
    ASSERT_EQ(lite_processor_->EncodeNormalized(norm, &ids, &pieces),
              StatusCode::kOk);

    size_t curr_pos = 0;
    for (std::string_view piece : pieces) {
      if (curr_pos >= norm.size()) {
        break;
      }
      const size_t pos = norm.find(piece, curr_pos);
      if (pos == std::string_view::npos) {
        continue;
      }
      curr_pos = pos + piece.size();

      const size_t orig_begin = offsets[pos];
      const size_t orig_end = offsets[pos + piece.size()];
      if (orig_end > input.size() || orig_begin >= orig_end) {
        continue;
      }

      const std::string_view orig_span =
          std::string_view(input).substr(orig_begin, orig_end - orig_begin);
      std::string renorm_orig;
      ASSERT_EQ(lite_processor_->Normalize(orig_span, &renorm_orig),
                StatusCode::kOk);
      std::string_view renorm_view = renorm_orig;
      if (!piece.starts_with("\xE2\x96\x81") &&
          renorm_view.starts_with("\xE2\x96\x81")) {
        renorm_view.remove_prefix(3);
      }
      if (renorm_view.empty() && piece == "\xE2\x96\x81") {
        renorm_view = piece;
      }
      EXPECT_NE(renorm_view.find(piece), std::string_view::npos)
          << "Renormalized original span \"" << renorm_view
          << "\" must contain token piece \"" << piece << "\"";
    }
  }
}

TEST_P(SentencePieceLiteTest, EncodeAndDecodeWithPiecesTest) {
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);

    std::string normalized;
    ASSERT_EQ(lite_processor_->Normalize(input, &normalized), StatusCode::kOk);

    std::vector<int> ids;
    std::vector<std::string_view> enc_pieces;
    ASSERT_EQ(lite_processor_->EncodeNormalized(normalized, &ids, &enc_pieces),
              StatusCode::kOk);
    ASSERT_EQ(ids.size(), enc_pieces.size());

    std::string reconstructed_enc;
    for (const auto& p : enc_pieces) {
      if (!normalized.empty()) {
        EXPECT_GE(p.data(), normalized.data());
        EXPECT_LE(p.data() + p.size(), normalized.data() + normalized.size());
      }
      reconstructed_enc.append(p.data(), p.size());
    }
    EXPECT_EQ(reconstructed_enc, normalized);

    std::string decoded;
    std::vector<std::string_view> dec_pieces;
    ASSERT_EQ(
        lite_processor_->Decode(Span<const int>(ids), &decoded, &dec_pieces),
        StatusCode::kOk);
    ASSERT_EQ(ids.size(), dec_pieces.size());

    std::string reconstructed_dec;
    for (const auto& p : dec_pieces) {
      if (!decoded.empty()) {
        EXPECT_GE(p.data(), decoded.data());
        EXPECT_LE(p.data() + p.size(), decoded.data() + decoded.size());
      }
      reconstructed_dec.append(p.data(), p.size());
    }
    EXPECT_EQ(reconstructed_dec, decoded);
  }
}

TEST_P(SentencePieceLiteTest, EncodeNormalizedEquivalence) {
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);

    std::string normalized;
    ASSERT_EQ(lite_processor_->Normalize(input, &normalized), StatusCode::kOk);

    std::vector<int> lite_ids_from_raw;
    ASSERT_EQ(lite_processor_->EncodeNormalized(normalized, &lite_ids_from_raw),
              StatusCode::kOk);

    std::vector<int> lite_ids;
    ASSERT_EQ(lite_processor_->Encode(input, &lite_ids), StatusCode::kOk);

    EXPECT_EQ(lite_ids_from_raw, lite_ids);
  }
}

TEST_P(SentencePieceLiteTest, DecodeSpan) {
  std::string input = "hello world";
  std::vector<int> lite_ids;
  ASSERT_EQ(lite_processor_->Encode(input, &lite_ids), StatusCode::kOk);
  ASSERT_FALSE(lite_ids.empty());

  const int* raw_ids = lite_ids.data();
  const size_t raw_size = lite_ids.size();

  std::string decoded_from_raw;
  ASSERT_EQ(lite_processor_->Decode(Span<const int>(raw_ids, raw_size),
                                    &decoded_from_raw),
            StatusCode::kOk);

  std::string decoded_vector;
  ASSERT_EQ(lite_processor_->Decode(lite_ids, &decoded_vector),
            StatusCode::kOk);
  EXPECT_EQ(decoded_from_raw, decoded_vector);
}

TEST_P(SentencePieceLiteTest, ControlTokensNotMerged) {
  {
    std::vector<int> lite_ids;
    ASSERT_EQ(lite_processor_->Encode("<s>", &lite_ids), StatusCode::kOk);
    // Should not be merged into a single control token ID.
    EXPECT_GT(lite_ids.size(), 1U);
  }
  {
    std::vector<int> lite_ids;
    ASSERT_EQ(lite_processor_->Encode("</s>", &lite_ids), StatusCode::kOk);
    // Should not be merged into a single control token ID.
    EXPECT_GT(lite_ids.size(), 1U);
  }
}

TEST_P(SentencePieceLiteTest, VocabSizeAndPieceType) {
  const size_t vocab_size = lite_processor_->vocab_size();
  EXPECT_EQ(vocab_size, static_cast<size_t>(proto_.pieces_size()));
  EXPECT_GT(vocab_size, 0U);

  for (size_t i = 0; i < vocab_size; ++i) {
    const int type = lite_processor_->piece_type(static_cast<int>(i));
    const auto& sp = proto_.pieces(static_cast<int>(i));
    EXPECT_EQ(type, static_cast<int>(sp.type()));

    const std::string_view piece =
        lite_processor_->IdToPiece(static_cast<int>(i));
    EXPECT_EQ(piece, sp.piece());
    EXPECT_EQ(lite_processor_->PieceToId(piece), static_cast<int>(i));
  }

  // Check out of bounds behavior
  EXPECT_EQ(lite_processor_->piece_type(-1), -1);
  EXPECT_EQ(lite_processor_->piece_type(static_cast<int>(vocab_size)), -1);
  EXPECT_EQ(lite_processor_->IdToPiece(-1), "<unk>");
  EXPECT_EQ(lite_processor_->IdToPiece(static_cast<int>(vocab_size)), "<unk>");
  EXPECT_EQ(lite_processor_->PieceToId("NON_EXISTING_PIECE"), -1);
  EXPECT_EQ(lite_processor_->PieceToId(""), -1);
  EXPECT_EQ(lite_processor_->PieceToId(std::string_view("a", 0)), -1);
  EXPECT_EQ(lite_processor_->PieceToId(std::string_view()), -1);
  EXPECT_EQ(lite_processor_->GetScore(-1), 0.0f);
  EXPECT_EQ(lite_processor_->GetScore(static_cast<int>(vocab_size)), 0.0f);
  EXPECT_EQ(lite_processor_->unk_id(), 0);

  auto find_expected_control_id = [&](std::string_view name) -> int {
    for (int i = 0; i < proto_.pieces_size(); ++i) {
      if (proto_.pieces(i).piece() == name &&
          proto_.pieces(i).type() ==
              ::sentencepiece::ModelProto::SentencePiece::CONTROL) {
        return i;
      }
    }
    return -1;
  };
  EXPECT_EQ(lite_processor_->bos_id(), find_expected_control_id("<s>"));
  EXPECT_EQ(lite_processor_->eos_id(), find_expected_control_id("</s>"));
  EXPECT_EQ(lite_processor_->pad_id(), find_expected_control_id("<pad>"));
}

TEST_P(SentencePieceLiteTest, OwnedBufferConstructor) {
  std::vector<int> lite_ids;
  std::string input = "hello world";

  {
    std::string temp_model_data = lite_model_data_;
    SentencePieceLiteProcessor owned_processor(
        std::make_shared<std::string>(std::move(temp_model_data)));
    ASSERT_EQ(owned_processor.status(), StatusCode::kOk);
    ASSERT_EQ(owned_processor.Encode(input, &lite_ids), StatusCode::kOk);
  }

  std::vector<int> normal_ids;
  ASSERT_EQ(lite_processor_->Encode(input, &normal_ids), StatusCode::kOk);
  EXPECT_EQ(lite_ids, normal_ids);
}

TEST_P(SentencePieceLiteTest, IdempotencyOnLargeCorpus) {
  std::string corpus_path;
  if (GetParam() == "bpe_byte") {
    corpus_path =
        "third_party/sentencepiece/src/test_data/wagahaiwa_nekodearu.txt";
  } else {
    corpus_path = "third_party/sentencepiece/src/test_data/botchan.txt";
  }

  const std::string actual_corpus_path = GetFilePath(corpus_path);
  std::ifstream input_file(actual_corpus_path);
  ASSERT_TRUE(input_file) << "Failed to open corpus file: " << corpus_path
                          << " (resolved to " << actual_corpus_path << ")";

  std::string line;
  int count = 0;
  // Test up to 500 lines to keep test execution fast.
  while (std::getline(input_file, line) && count < 500) {
    ++count;
    SCOPED_TRACE("Line number: " + std::to_string(count));

    std::vector<int> lite_ids;
    ASSERT_EQ(lite_processor_->Encode(line, &lite_ids), StatusCode::kOk);

    std::string lite_decoded;
    ASSERT_EQ(lite_processor_->Decode(lite_ids, &lite_decoded),
              StatusCode::kOk);

    const bool byte_fallback = (lite_processor_->PieceToId("<0x00>") != -1);
    const int unk_id = lite_processor_->PieceToId("<unk>");
    if (!byte_fallback &&
        std::find(lite_ids.begin(), lite_ids.end(), unk_id) != lite_ids.end()) {
      continue;
    }

    std::vector<int> re_ids;
    ASSERT_EQ(lite_processor_->Encode(lite_decoded, &re_ids), StatusCode::kOk);
    EXPECT_EQ(re_ids, lite_ids);

    std::string re_decoded;
    ASSERT_EQ(lite_processor_->Decode(re_ids, &re_decoded), StatusCode::kOk);
    EXPECT_EQ(re_decoded, lite_decoded);
  }
}

TEST(SentencePieceLiteSecurityTest, CorruptedBuffer) {
  std::string corrupted_data = "invalid flatbuffer data";
  SentencePieceLiteProcessor processor(corrupted_data);
  EXPECT_NE(processor.status(), StatusCode::kOk);
}

TEST(SentencePieceLiteSecurityTest, MaliciousTrieBlob) {
  flatbuffers::FlatBufferBuilder fbb;  // NOLINT(misc-include-cleaner)

  std::vector<flatbuffers::Offset<flatbuffers::String>>  // NOLINT
      pieces_offsets;
  pieces_offsets.push_back(fbb.CreateString("<unk>"));
  auto pieces = fbb.CreateVector(pieces_offsets);

  std::vector<float> scores_data = {0.0};
  auto scores = fbb.CreateVector(scores_data);

  std::vector<int8_t> types_data = {static_cast<int8_t>(PieceType_UNKNOWN)};
  auto types = fbb.CreateVector(types_data);

  std::vector<uint32_t> pieces_trie_data = {10240000, 0, 0, 0};
  auto pieces_trie_blob = fbb.CreateVector(
      reinterpret_cast<const uint8_t*>(pieces_trie_data.data()),
      pieces_trie_data.size() * sizeof(uint32_t));

  ModelProtoBuilder builder(fbb);
  builder.add_pieces(pieces);
  builder.add_scores(scores);
  builder.add_types(types);
  builder.add_pieces_trie_blob(pieces_trie_blob);
  builder.add_model_type(ModelType_UNIGRAM);
  auto model_offset = builder.Finish();
  fbb.Finish(model_offset);

  std::string_view model_buffer(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());

  SentencePieceLiteProcessor processor(model_buffer);
  EXPECT_NE(processor.status(), StatusCode::kOk);
}

// Smallest model the converter accepts, so the tests below can vary one thing
// at a time.
::sentencepiece::ModelProto MinimalUnigramProto() {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* piece_unk = proto.add_pieces();
  piece_unk->set_piece("<unk>");
  piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* piece = proto.add_pieces();
  piece->set_piece("ab");
  piece->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  return proto;
}

TEST(SentencePieceLiteSecurityTest, RejectMisalignedModelBuffer) {
  auto model_or = ::sentencepiece::lite::ToFlatbuffer(MinimalUnigramProto());
  ASSERT_TRUE(model_or.ok()) << model_or.status();

  // The widest scalar in the schema is 4 bytes, and flatbuffers::Verifier only
  // checks offsets relative to the start of the buffer, so a buffer that is
  // itself misaligned passes verification and then yields misaligned loads.
  constexpr size_t kAlignment = 4;

  // Distance to the next address whose residue modulo kAlignment is `residue`.
  const auto shift_for = [](const char* base, size_t residue) {
    return (residue + kAlignment -
            reinterpret_cast<uintptr_t>(base) % kAlignment) %
           kAlignment;
  };

  std::vector<char> aligned_storage(model_or->size() + kAlignment);
  const size_t aligned_shift = shift_for(aligned_storage.data(), 0);
  std::memcpy(aligned_storage.data() + aligned_shift, model_or->data(),
              model_or->size());
  SentencePieceLiteProcessor aligned(std::string_view(
      aligned_storage.data() + aligned_shift, model_or->size()));
  EXPECT_EQ(aligned.status(), StatusCode::kOk);

  std::vector<char> misaligned_storage(model_or->size() + kAlignment);
  const size_t misaligned_shift = shift_for(misaligned_storage.data(), 1);
  std::memcpy(misaligned_storage.data() + misaligned_shift, model_or->data(),
              model_or->size());
  SentencePieceLiteProcessor misaligned(std::string_view(
      misaligned_storage.data() + misaligned_shift, model_or->size()));
  EXPECT_EQ(misaligned.status(), StatusCode::kInvalidArgument);
}

TEST(SentencePieceLiteSecurityTest, RejectOverlongUnkSurface) {
  // Mirrors kMaxPieceLength in sentencepiece_lite.cc.
  constexpr size_t kMaxPieceLength = 4096;

  const struct {
    std::string_view name;
    size_t length;
    StatusCode expected;
  } kCases[] = {
      {"at limit", kMaxPieceLength, StatusCode::kOk},
      {"over limit", kMaxPieceLength + 1, StatusCode::kInternal},
  };

  for (const auto& c : kCases) {
    SCOPED_TRACE(c.name);
    ::sentencepiece::ModelProto proto = MinimalUnigramProto();
    proto.mutable_trainer_spec()->set_unk_surface(std::string(c.length, 'x'));

    auto model_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(model_or.ok()) << model_or.status();
    SentencePieceLiteProcessor processor(*model_or);
    EXPECT_EQ(processor.status(), c.expected);
  }
}

TEST(SentencePieceLiteSecurityTest, RejectNullOutput) {
  auto model_or = ::sentencepiece::lite::ToFlatbuffer(MinimalUnigramProto());
  ASSERT_TRUE(model_or.ok()) << model_or.status();
  SentencePieceLiteProcessor processor(*model_or);
  ASSERT_EQ(processor.status(), StatusCode::kOk);

  EXPECT_EQ(processor.Normalize("ab", nullptr), StatusCode::kInvalidArgument);

  const std::vector<int> ids = {1};
  EXPECT_EQ(processor.Decode(ids, nullptr), StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, RejectNaNAndInfUnigramScores) {
  for (float bad_score : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);

    auto* piece_unk = proto.add_pieces();
    piece_unk->set_piece("<unk>");
    piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* piece1 = proto.add_pieces();
    piece1->set_piece("ab");
    piece1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
    piece1->set_score(bad_score);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(status_or.ok());
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
    EXPECT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kInternal);
  }
}

TEST(SentencePieceModelConvertersTest, RejectEmptyVocabulary) {
  ::sentencepiece::ModelProto proto;
  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, RejectDuplicatePiece) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece("foo");
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto* p2 = proto.add_pieces();
  p2->set_piece("foo");
  p2->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, RejectZeroLengthPiece) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece("");
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, RejectTooLongPiece) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece(std::string(5000, 'a'));
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, RejectOutOfBoundsUnkId) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(5);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(status_or.ok());
  ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
  EXPECT_NE(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
}

TEST(SentencePieceModelConvertersTest, SpecialIdsExtractionAndControlCheck) {
  // Test case 1: Standard model with UNK, BOS, EOS, PAD as CONTROL symbols
  {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);

    auto* p0 = proto.add_pieces();
    p0->set_piece("<unk>");
    p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* p1 = proto.add_pieces();
    p1->set_piece("<s>");
    p1->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto* p2 = proto.add_pieces();
    p2->set_piece("</s>");
    p2->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto* p3 = proto.add_pieces();
    p3->set_piece("<pad>");
    p3->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto* p4 = proto.add_pieces();
    p4->set_piece("hello");
    p4->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(status_or.ok());
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
    EXPECT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
    EXPECT_EQ(processor.unk_id(), 0);
    EXPECT_EQ(processor.bos_id(), 1);
    EXPECT_EQ(processor.eos_id(), 2);
    EXPECT_EQ(processor.pad_id(), 3);
  }

  // Test case 2: Custom special token names
  {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);
    trainer_spec->set_bos_piece("[BOS]");
    trainer_spec->set_eos_piece("[EOS]");
    trainer_spec->set_pad_piece("[PAD]");

    auto* p0 = proto.add_pieces();
    p0->set_piece("<unk>");
    p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* p1 = proto.add_pieces();
    p1->set_piece("[BOS]");
    p1->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto* p2 = proto.add_pieces();
    p2->set_piece("[EOS]");
    p2->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto* p3 = proto.add_pieces();
    p3->set_piece("[PAD]");
    p3->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(status_or.ok());
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
    EXPECT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
    EXPECT_EQ(processor.bos_id(), 1);
    EXPECT_EQ(processor.eos_id(), 2);
    EXPECT_EQ(processor.pad_id(), 3);
  }

  // Test case 3: <s> piece exists but is NORMAL or USER_DEFINED, not CONTROL
  {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);

    auto* p0 = proto.add_pieces();
    p0->set_piece("<unk>");
    p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* p1 = proto.add_pieces();
    p1->set_piece("<s>");
    p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

    auto* p2 = proto.add_pieces();
    p2->set_piece("</s>");
    p2->set_type(::sentencepiece::ModelProto::SentencePiece::USER_DEFINED);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(status_or.ok());
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
    EXPECT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
    // Because types are not CONTROL, bos_id and eos_id must be -1
    EXPECT_EQ(processor.bos_id(), -1);
    EXPECT_EQ(processor.eos_id(), -1);
    EXPECT_EQ(processor.pad_id(), -1);
  }

  // Test case 4: Uninitialized / error processor returns -1
  {
    std::string invalid_buffer = "invalid";
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(invalid_buffer);
    EXPECT_NE(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
    EXPECT_EQ(processor.unk_id(), -1);
    EXPECT_EQ(processor.bos_id(), -1);
    EXPECT_EQ(processor.eos_id(), -1);
    EXPECT_EQ(processor.pad_id(), -1);
  }
}

TEST(SentencePieceModelConvertersTest, AdversarialEmbeddedNullByteInPiece) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece(std::string("foo\0bar", 7));
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto* p2 = proto.add_pieces();
  p2->set_piece(std::string("foo\0baz", 7));
  p2->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, AdversarialTruncatedUtf8Bigram) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::BPE);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece("\xE2\x82");
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto* p2 = proto.add_pieces();
  p2->set_piece("\xF0\x9F\x98");
  p2->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  if (status_or.ok()) {
    ::sentencepiece::lite::SentencePieceLiteProcessor processor(*status_or);
  }
}

TEST(SentencePieceLiteSecurityTest, NullAndCorruptedModelErrorHandling) {
  ::sentencepiece::lite::SentencePieceLiteProcessor null_proc(
      std::shared_ptr<std::string>(nullptr));
  EXPECT_EQ(null_proc.status(),
            ::sentencepiece::lite::StatusCode::kInvalidArgument);
  std::string output;
  std::vector<int> ids;
  std::vector<std::string_view> out_views;
  EXPECT_EQ(null_proc.Normalize("hello", &output),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);
  EXPECT_EQ(null_proc.Encode("hello", &ids),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);
  EXPECT_EQ(null_proc.EncodeNormalized("hello", &ids),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);
  EXPECT_EQ(null_proc.PretokenizeAtSafeBoundaries("hello", &out_views),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);
  EXPECT_EQ(null_proc.Decode({1, 2}, &output),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);

  auto bad_buffer = std::make_shared<std::string>("corrupted flatbuffer blob");
  ::sentencepiece::lite::SentencePieceLiteProcessor bad_proc(bad_buffer);
  EXPECT_NE(bad_proc.status(), ::sentencepiece::lite::StatusCode::kOk);
  EXPECT_EQ(bad_proc.Encode("hello", &ids),
            ::sentencepiece::lite::StatusCode::kFailedPrecondition);
}

TEST(SentencePieceLiteSecurityTest, BpeUserDefinedTokens) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::BPE);
  trainer_spec->set_unk_id(0);

  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* p1 = proto.add_pieces();
  p1->set_piece("\xe2\x96\x81");  //   (space prefix)
  p1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  p1->set_score(0.0f);

  auto* p2 = proto.add_pieces();
  p2->set_piece("[SPECIAL]");
  p2->set_type(::sentencepiece::ModelProto::SentencePiece::USER_DEFINED);
  p2->set_score(0.0f);

  auto* p3 = proto.add_pieces();
  p3->set_piece("a");
  p3->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  p3->set_score(-1.0f);

  auto* p4 = proto.add_pieces();
  p4->set_piece("b");
  p4->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  p4->set_score(-2.0f);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(status_or.ok());
  ::sentencepiece::lite::SentencePieceLiteProcessor proc(*status_or);
  ASSERT_EQ(proc.status(), ::sentencepiece::lite::StatusCode::kOk);

  std::vector<int> ids;
  EXPECT_EQ(proc.EncodeNormalized("\xe2\x96\x81"
                                  "a[SPECIAL]b",
                                  &ids),
            ::sentencepiece::lite::StatusCode::kOk);
  ASSERT_EQ(ids.size(), 4U);
  EXPECT_EQ(ids[0], 1);
  EXPECT_EQ(ids[1], 3);  // a
  EXPECT_EQ(ids[2], 2);  // [SPECIAL]
  EXPECT_EQ(ids[3], 4);  // b
}

TEST(SentencePieceLiteSecurityTest, CorruptedTrieStructureValidation) {
  // Test various malformed Double-Array Trie structures that must be rejected
  // by DoubleArrayTrie::Validate() during initialization.
  auto build_and_test = [](const std::vector<uint32_t>& trie_data) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<flatbuffers::String>> pieces_data = {
        fbb.CreateString("<unk>")};
    auto pieces = fbb.CreateVector(pieces_data);
    std::vector<float> scores_data = {0.0f};
    auto scores = fbb.CreateVector(scores_data);
    std::vector<int8_t> types_data = {static_cast<int8_t>(PieceType_UNKNOWN)};
    auto types = fbb.CreateVector(types_data);
    auto pieces_trie_blob =
        fbb.CreateVector(reinterpret_cast<const uint8_t*>(trie_data.data()),
                         trie_data.size() * sizeof(uint32_t));

    ModelProtoBuilder builder(fbb);
    builder.add_pieces(pieces);
    builder.add_scores(scores);
    builder.add_types(types);
    builder.add_pieces_trie_blob(pieces_trie_blob);
    builder.add_model_type(ModelType_UNIGRAM);
    fbb.Finish(builder.Finish());

    std::string_view model_buffer(
        reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());
    SentencePieceLiteProcessor processor(model_buffer);
    EXPECT_NE(processor.status(), StatusCode::kOk);
    std::vector<int> ids;
    EXPECT_NE(processor.Encode("hello", &ids), StatusCode::kOk);
  };

  // Case 1: Root node has non-zero label (label must be 0 for root)
  build_and_test({1, 0, 0, 0});
  // Case 2: Root node has leaf bit set (root cannot be a leaf)
  build_and_test({(1U << 31), 0, 0, 0});
  // Case 3: Root node has zero offset (must have non-zero transition offset)
  build_and_test({0, 0, 0, 0});
  // Case 4: Internal node (i=1) has OOB child transition offset
  build_and_test({16, 1000000, 0, 0});
  // Case 5: Internal node claims leaf bit, but target is not leaf (label <=
  // 0xFF)
  build_and_test({16, (1U << 31) | 2, 10, 0});
  // Case 6: Leaf node (i=1, label > 0xFF) has OOB suffix link value
  build_and_test({16, 0x100 | 1000000, 0, 0});
}

TEST(SentencePieceModelConvertersTest, RejectUnsupportedModelTypes) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::WORD);
  trainer_spec->set_unk_id(0);
  auto* p0 = proto.add_pieces();
  p0->set_piece("<unk>");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto status_or_word = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or_word.ok());
  EXPECT_EQ(status_or_word.status().code(), absl::StatusCode::kInvalidArgument);

  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::CHAR);
  auto status_or_char = ::sentencepiece::lite::ToFlatbuffer(proto);
  EXPECT_FALSE(status_or_char.ok());
  EXPECT_EQ(status_or_char.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SentencePieceModelConvertersTest, LongestPrefixLookupRegressionTest) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::BPE);
  trainer_spec->set_unk_id(0);

  auto* piece_unk = proto.add_pieces();
  piece_unk->set_piece("<unk>");
  piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  for (const char* char_piece : {"f", "o", "b", "a", "r", "z"}) {
    auto* cp = proto.add_pieces();
    cp->set_piece(char_piece);
    cp->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
    cp->set_score(1);
  }

  // Overlapping user-defined pieces to test longest match preference
  auto* piece_foo = proto.add_pieces();
  piece_foo->set_piece("foo");
  piece_foo->set_type(::sentencepiece::ModelProto::SentencePiece::USER_DEFINED);
  piece_foo->set_score(1);

  auto* piece_foobar = proto.add_pieces();
  piece_foobar->set_piece("foobar");
  piece_foobar->set_type(
      ::sentencepiece::ModelProto::SentencePiece::USER_DEFINED);
  piece_foobar->set_score(1);

  auto* piece_foobarbaz = proto.add_pieces();
  piece_foobarbaz->set_piece("foobarbaz");
  piece_foobarbaz->set_type(
      ::sentencepiece::ModelProto::SentencePiece::USER_DEFINED);
  piece_foobarbaz->set_score(1);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(status_or.ok());
  const std::string& binary_model = *status_or;

  ::sentencepiece::lite::SentencePieceLiteProcessor processor(binary_model);
  ASSERT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);

  std::vector<int> ids;
  EXPECT_EQ(processor.Encode("foobarbazfoo", &ids),
            ::sentencepiece::lite::StatusCode::kOk);
  std::vector<int> expected_ids = {9, 7};
  EXPECT_EQ(ids, expected_ids);
}

TEST(SentencePieceModelConvertersTest, StrictBigramPreTokenizationTest) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* piece_unk = proto.add_pieces();
  piece_unk->set_piece("<unk>");
  piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto* piece1 = proto.add_pieces();
  piece1->set_piece("ab");
  piece1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  piece1->set_score(-1.0);

  auto* piece2 = proto.add_pieces();
  piece2->set_piece("bc");
  piece2->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  piece2->set_score(-1.0);

  auto* piece3 = proto.add_pieces();
  piece3->set_piece("xy");
  piece3->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  piece3->set_score(-1.0);

  for (const char* char_piece : {"a", "b", "c", "x", "y"}) {
    auto* cp = proto.add_pieces();
    cp->set_piece(char_piece);
    cp->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
    cp->set_score(-10.0);
  }

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(status_or.ok());

  const std::string& binary_model = *status_or;
  ::sentencepiece::lite::SentencePieceLiteProcessor processor(binary_model);
  ASSERT_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);

  std::vector<std::string_view> chunks;
  EXPECT_EQ(processor.PretokenizeAtSafeBoundaries("abcxy", &chunks),
            ::sentencepiece::lite::StatusCode::kOk);
  std::vector<std::string_view> expected_chunks = {"abc", "xy"};
  EXPECT_EQ(chunks, expected_chunks);

  std::vector<int> ids;
  EXPECT_EQ(processor.Encode("abcxy", &ids),
            ::sentencepiece::lite::StatusCode::kOk);
  EXPECT_FALSE(ids.empty());
}

// Verifies that Viterbi score re-centering (subtracting an offset from active
// hypotheses when cumulative log-prob exceeds a threshold to prevent underflow)
// preserves relative path scores. Even when threshold is lowered to 1.0f to
// trigger frequent resets, resulting token IDs must remain 100% invariant.
TEST_P(SentencePieceLiteTest, ScoreRecenteringThresholdInvariance) {
  if (GetParam() != "unigram") return;

  const std::string input =
      "I saw a girl with a telescope. "
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
      "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
      "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。";

  std::vector<int> baseline_ids;
  ASSERT_EQ(lite_processor_->Encode(input, &baseline_ids), StatusCode::kOk);
  EXPECT_FALSE(baseline_ids.empty());

  for (float threshold : {1.0f, 2.5f, 5.0f, 10.0f, 50.0f, 100.0f, 500.0f,
                          1000.0f, 10000.0f, 100000.0f}) {
    lite_processor_->SetScoreResetThresholdForTesting(threshold);
    std::vector<int> test_ids;
    EXPECT_EQ(lite_processor_->Encode(input, &test_ids), StatusCode::kOk)
        << "Failed at threshold: " << threshold;
    EXPECT_EQ(test_ids, baseline_ids)
        << "Token ID divergence at threshold: " << threshold;
  }
  lite_processor_->SetScoreResetThresholdForTesting(100000.0f);
}

TEST_P(SentencePieceLiteTest, SampleNormalizedTest) {
  for (const auto& input : test_inputs_) {
    SCOPED_TRACE(input);
    std::string normalized;
    ASSERT_EQ(lite_processor_->Normalize(input, &normalized), StatusCode::kOk);
    if (normalized.empty()) continue;

    // Simple deterministic 64-bit LCG (Linear Congruential Generator) using
    // Knuth's MMIX constants for testing purposes.
    uint64_t state = 123456789;
    auto uniform_sampler = [&state]() -> float {
      state = state * 6364136223846793005ULL + 1ULL;
      return static_cast<float>((state >> 33) & 0x7FFFFFFF) /
             static_cast<float>(0x80000000ULL);
    };

    // 1. In deterministic mode (alpha == 0.0f), SampleNormalized must produce
    // identical results to EncodeNormalized.
    std::vector<int> det_ids, enc_ids;
    std::vector<std::string_view> det_pieces, enc_pieces;
    ASSERT_EQ(lite_processor_->SampleNormalized(
                  normalized, 0.0f, uniform_sampler, &det_ids, &det_pieces),
              StatusCode::kOk);
    ASSERT_EQ(
        lite_processor_->EncodeNormalized(normalized, &enc_ids, &enc_pieces),
        StatusCode::kOk);
    EXPECT_EQ(det_ids, enc_ids);
    EXPECT_EQ(det_pieces, enc_pieces);

    // 2. With positive sampling temperature/dropout (alpha = 0.5f), running
    // multiple trials should yield valid segmentations and demonstrate
    // diversity on non-trivial strings.
    for (int trial = 0; trial < 10; ++trial) {
      std::vector<int> sample_ids;
      std::vector<std::string_view> sample_pieces;
      ASSERT_EQ(
          lite_processor_->SampleNormalized(normalized, 0.5f, uniform_sampler,
                                            &sample_ids, &sample_pieces),
          StatusCode::kOk);
      ASSERT_EQ(sample_ids.size(), sample_pieces.size());

      // Reconstructed pieces must match normalized text.
      std::string reconstructed;
      for (const auto& p : sample_pieces) {
        reconstructed.append(p.data(), p.size());
      }
      EXPECT_EQ(reconstructed, normalized);
    }

    // 3. Passing a negative alpha (< 0.0f) must return kInvalidArgument.
    std::vector<int> invalid_ids;
    EXPECT_EQ(lite_processor_->SampleNormalized(normalized, -0.1f,
                                                uniform_sampler, &invalid_ids),
              StatusCode::kInvalidArgument);
  }
}

TEST_P(SentencePieceLiteTest, PretokenizeWithFunctionRefReceiver) {
  if (lite_processor_ == nullptr ||
      lite_processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model non-existent or invalid";
  }

  std::string normalized;
  ASSERT_EQ(
      lite_processor_->Normalize("Hello World. This is a test.", &normalized),
      StatusCode::kOk);

  std::vector<std::string> chunks_vector;
  EXPECT_EQ(lite_processor_->PretokenizeAtSafeBoundaries(
                normalized,
                [&chunks_vector](std::string_view chunk) {
                  chunks_vector.push_back(std::string(chunk));
                }),
            StatusCode::kOk);

  std::vector<std::string_view> chunks_legacy;
  EXPECT_EQ(
      lite_processor_->PretokenizeAtSafeBoundaries(normalized, &chunks_legacy),
      StatusCode::kOk);

  ASSERT_EQ(chunks_vector.size(), chunks_legacy.size());
  for (size_t i = 0; i < chunks_vector.size(); ++i) {
    EXPECT_EQ(chunks_vector[i], chunks_legacy[i]);
  }
}

TEST_P(SentencePieceLiteTest, PretokenizeAtSafeBoundariesSWARAndMultilingual) {
  if (lite_processor_ == nullptr ||
      lite_processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model non-existent or invalid";
  }

  const std::vector<std::string> test_cases = {
      "The quick brown fox jumps over the lazy dog.",
      "Hello123World456! SWAR acceleration test with 日本語 and emojis 🎉.",
      "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.",
      "",
  };

  for (const auto& text : test_cases) {
    std::string normalized;
    ASSERT_EQ(lite_processor_->Normalize(text, &normalized), StatusCode::kOk);

    std::vector<std::string_view> chunks;
    EXPECT_EQ(lite_processor_->PretokenizeAtSafeBoundaries(normalized, &chunks),
              StatusCode::kOk);

    // 1. In standard Unigram/BPE models, contiguous ASCII letter sequences
    //    [a-zA-Z]+ are never split across chunk boundaries.
    if (GetParam() != "bpe_byte") {
      auto is_ascii_alpha = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
      };
      for (size_t i = 0; i + 1 < chunks.size(); ++i) {
        const bool ends_with_alpha =
            !chunks[i].empty() && is_ascii_alpha(chunks[i].back());
        const bool starts_with_alpha =
            !chunks[i + 1].empty() && is_ascii_alpha(chunks[i + 1].front());
        EXPECT_FALSE(ends_with_alpha && starts_with_alpha)
            << "Alpha sequence inappropriately split between chunks: '"
            << chunks[i] << "' and '" << chunks[i + 1] << "'";
      }
    }

    // 2. All chunks concatenated must exactly reconstruct the normalized text.
    std::string reconstructed;
    for (std::string_view chunk : chunks) {
      reconstructed.append(chunk.data(), chunk.size());
    }
    EXPECT_EQ(reconstructed, normalized);
  }
}

TEST(SentencePieceLiteNormalizerTest, IdentityNormalizerSWARAndOffsets) {
  auto create_identity_model = [](bool add_dummy_prefix,
                                  bool remove_extra_whitespaces,
                                  bool escape_whitespaces,
                                  bool treat_whitespace_as_suffix) {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);
    trainer_spec->set_treat_whitespace_as_suffix(treat_whitespace_as_suffix);

    auto* normalizer_spec = proto.mutable_normalizer_spec();
    normalizer_spec->set_name("identity");
    normalizer_spec->set_add_dummy_prefix(add_dummy_prefix);
    normalizer_spec->set_remove_extra_whitespaces(remove_extra_whitespaces);
    normalizer_spec->set_escape_whitespaces(escape_whitespaces);

    auto* piece_unk = proto.add_pieces();
    piece_unk->set_piece("<unk>");
    piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* piece1 = proto.add_pieces();
    piece1->set_piece("a");
    piece1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
    piece1->set_score(-1.0f);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    EXPECT_TRUE(status_or.ok()) << status_or.status();
    return std::move(status_or.value());
  };

  // 1. Standard configuration (dummy prefix, remove extra whitespaces, escape
  // whitespaces)
  std::string model_buf = create_identity_model(true, true, true, false);
  SentencePieceLiteProcessor processor(model_buf);
  ASSERT_EQ(processor.status(), StatusCode::kOk);

  struct TestCase {
    std::string input;
    std::string expected_norm;
  };

  const std::vector<TestCase> cases = {
      {"hello world",
       "\xE2\x96\x81"
       "hello\xE2\x96\x81"
       "world"},
      {"  hello   world  ",
       "\xE2\x96\x81"
       "hello\xE2\x96\x81"
       "world"},
      {"abcdefghijklmnopqrstuvwxyz0123456789",
       "\xE2\x96\x81"
       "abcdefghijklmnopqrstuvwxyz0123456789"},
      {"日本語のテスト",
       "\xE2\x96\x81"
       "日本語のテスト"},
      {"日本語 と English",
       "\xE2\x96\x81"
       "日本語\xE2\x96\x81"
       "と\xE2\x96\x81"
       "English"},
      {"", ""},
      {"   ", ""},
  };

  for (const auto& tc : cases) {
    std::string norm;
    std::vector<size_t> offsets;
    ASSERT_EQ(processor.Normalize(tc.input, &norm, &offsets), StatusCode::kOk);
    EXPECT_EQ(norm, tc.expected_norm);
    if (!norm.empty()) {
      ASSERT_EQ(offsets.size(), norm.size() + 1);
      EXPECT_LE(offsets.back(), tc.input.size());
      for (size_t i = 0; i < norm.size(); ++i) {
        EXPECT_LE(offsets[i], offsets[i + 1]);
      }
    }
  }

  // 2. Raw unescaped configuration (no dummy prefix, no whitespace removal, no
  // escape)
  std::string raw_model_buf = create_identity_model(false, false, false, false);
  SentencePieceLiteProcessor raw_processor(raw_model_buf);
  ASSERT_EQ(raw_processor.status(), StatusCode::kOk);
  std::string norm;
  std::vector<size_t> offsets;
  ASSERT_EQ(raw_processor.Normalize("  hello  world  ", &norm, &offsets),
            StatusCode::kOk);
  EXPECT_EQ(norm, "  hello  world  ");
  ASSERT_EQ(offsets.size(), norm.size() + 1);
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_EQ(offsets[i], i);
  }

  // 3. Suffix whitespace configuration
  std::string suffix_model_buf = create_identity_model(true, true, true, true);
  SentencePieceLiteProcessor suffix_processor(suffix_model_buf);
  ASSERT_EQ(suffix_processor.status(), StatusCode::kOk);
  std::string suffix_norm;
  ASSERT_EQ(suffix_processor.Normalize("hello world", &suffix_norm),
            StatusCode::kOk);
  EXPECT_EQ(suffix_norm, "hello\xE2\x96\x81world\xE2\x96\x81");
}

TEST(SentencePieceLiteNormalizerTest, All16NormalizerSpecPermutations) {
  auto create_model = [](bool add_dummy_prefix, bool remove_extra_whitespaces,
                         bool escape_whitespaces,
                         bool treat_whitespace_as_suffix) {
    ::sentencepiece::ModelProto proto;
    auto* trainer_spec = proto.mutable_trainer_spec();
    trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
    trainer_spec->set_unk_id(0);
    trainer_spec->set_treat_whitespace_as_suffix(treat_whitespace_as_suffix);

    auto* normalizer_spec = proto.mutable_normalizer_spec();
    normalizer_spec->set_name("identity");
    normalizer_spec->set_add_dummy_prefix(add_dummy_prefix);
    normalizer_spec->set_remove_extra_whitespaces(remove_extra_whitespaces);
    normalizer_spec->set_escape_whitespaces(escape_whitespaces);

    auto* piece_unk = proto.add_pieces();
    piece_unk->set_piece("<unk>");
    piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

    auto* piece1 = proto.add_pieces();
    piece1->set_piece("a");
    piece1->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
    piece1->set_score(-1.0f);

    auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    EXPECT_TRUE(status_or.ok()) << status_or.status();
    return std::move(status_or.value());
  };

  const std::vector<std::string> test_inputs = {
      "",
      " ",
      "   ",
      "hello",
      "  hello",
      "hello  ",
      "hello world",
      "  hello   world  ",
      "a",
      " a ",
      "日本語のテスト",
      "  日本語  の  テスト  ",
      "日本語 と English 123 !?",
      "A B C D E F G",
      "  A   B   C  ",
  };

  for (int mask = 0; mask < 16; ++mask) {
    const bool add_dummy_prefix = (mask & 1) != 0;
    const bool remove_extra_whitespaces = (mask & 2) != 0;
    const bool escape_whitespaces = (mask & 4) != 0;
    const bool treat_whitespace_as_suffix = (mask & 8) != 0;

    std::string model_buf =
        create_model(add_dummy_prefix, remove_extra_whitespaces,
                     escape_whitespaces, treat_whitespace_as_suffix);
    SentencePieceLiteProcessor processor(model_buf);
    ASSERT_EQ(processor.status(), StatusCode::kOk);

    const std::string_view space_symbol =
        escape_whitespaces ? "\xE2\x96\x81" : " ";

    for (const auto& input : test_inputs) {
      // Verify that the fast path (offset == nullptr, using
      // NormalizeIdentityFast) and the general path (offset != nullptr,
      // tracking character offsets) produce the exact same normalized string.
      std::string norm_fast;
      ASSERT_EQ(processor.Normalize(input, &norm_fast, nullptr),
                StatusCode::kOk);

      std::string norm_with_offsets;
      std::vector<size_t> offsets;
      ASSERT_EQ(processor.Normalize(input, &norm_with_offsets, &offsets),
                StatusCode::kOk);

      EXPECT_EQ(norm_fast, norm_with_offsets)
          << "Mismatch for mask=" << mask << " input=[" << input << "]";

      if (input.empty() ||
          (remove_extra_whitespaces &&
           input.find_first_not_of(' ') == std::string::npos)) {
        EXPECT_TRUE(norm_fast.empty());
      } else {
        if (!treat_whitespace_as_suffix && add_dummy_prefix) {
          EXPECT_TRUE(std::string_view(norm_fast).starts_with(space_symbol));
        }
        if (treat_whitespace_as_suffix && add_dummy_prefix) {
          EXPECT_TRUE(std::string_view(norm_fast).ends_with(space_symbol));
        }
        if (remove_extra_whitespaces) {
          EXPECT_FALSE(std::string_view(norm_fast).find("  ") !=  // NOLINT
                           std::string_view::npos &&
                       !escape_whitespaces);
        }

        ASSERT_EQ(offsets.size(), norm_fast.size() + 1);
        EXPECT_LE(offsets.back(), input.size());
        for (size_t i = 0; i < norm_fast.size(); ++i) {
          EXPECT_LE(offsets[i], offsets[i + 1]);
        }
      }
    }
  }
}

TEST(SentencePieceLiteNormalizerTest, InvalidUTF8ReplacementCharacterTest) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(0);

  auto* normalizer_spec = proto.mutable_normalizer_spec();
  normalizer_spec->set_name("identity");
  normalizer_spec->set_add_dummy_prefix(true);
  normalizer_spec->set_remove_extra_whitespaces(true);
  normalizer_spec->set_escape_whitespaces(true);

  auto* piece_unk = proto.add_pieces();
  piece_unk->set_piece("<unk>");
  piece_unk->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);

  auto status_or = ::sentencepiece::lite::ToFlatbuffer(proto);
  ASSERT_TRUE(status_or.ok()) << status_or.status();
  SentencePieceLiteProcessor processor(status_or.value());
  ASSERT_EQ(processor.status(), StatusCode::kOk);

  const std::string kRepl = "\xEF\xBF\xBD";

  struct InvalidCase {
    std::string input;
    std::string expected_norm;
  };

  const std::vector<InvalidCase> cases = {
      {"\xFF", "\xE2\x96\x81" + kRepl},
      {"\x80", "\xE2\x96\x81" + kRepl},
      {"\xC0\xAF", "\xE2\x96\x81" + kRepl + kRepl},
      {"hello\xFFworld", "\xE2\x96\x81hello" + kRepl + "world"},
      {"  hello \xFF world  ",
       "\xE2\x96\x81hello\xE2\x96\x81" + kRepl + "\xE2\x96\x81world"},
      {"テスト\x80です", "\xE2\x96\x81テスト" + kRepl + "です"},
      {"\xFF\xFE", "\xE2\x96\x81" + kRepl + kRepl},
  };

  for (const auto& tc : cases) {
    // Verify that the fast path (offset == nullptr) and general path (offset !=
    // nullptr) both produce the exact same normalized output with replacement
    // characters.
    std::string norm_fast;
    ASSERT_EQ(processor.Normalize(tc.input, &norm_fast, nullptr),
              StatusCode::kOk);
    EXPECT_EQ(norm_fast, tc.expected_norm);

    std::string norm_with_offsets;
    std::vector<size_t> offsets;
    ASSERT_EQ(processor.Normalize(tc.input, &norm_with_offsets, &offsets),
              StatusCode::kOk);
    EXPECT_EQ(norm_with_offsets, tc.expected_norm);
    EXPECT_EQ(norm_fast, norm_with_offsets);

    ASSERT_EQ(offsets.size(), norm_with_offsets.size() + 1);
    EXPECT_LE(offsets.back(), tc.input.size());
    for (size_t i = 0; i < norm_with_offsets.size(); ++i) {
      EXPECT_LE(offsets[i], offsets[i + 1]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(UnigramAndBPE, SentencePieceLiteTest,
                         ::testing::Values(std::string_view("unigram"),
                                           std::string_view("bpe"),
                                           std::string_view("bpe_byte")));

TEST(SentencePieceModelConvertersTest, TreatNullByteAsUnused) {
  // Build a test ModelProto with byte_fallback = true and a piece containing
  // '\0'
  ::sentencepiece::ModelProto proto;
  auto* ts = proto.mutable_trainer_spec();
  ts->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  ts->set_byte_fallback(true);
  ts->set_unk_id(0);

  // 0: <unk>
  auto* sp0 = proto.add_pieces();
  sp0->set_piece("<unk>");
  sp0->set_type(::sentencepiece::ModelProto::SentencePiece::UNKNOWN);
  sp0->set_score(0.0);

  // 1: <s> (control)
  auto* sp1 = proto.add_pieces();
  sp1->set_piece("<s>");
  sp1->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

  // 2: </s> (control)
  auto* sp2 = proto.add_pieces();
  sp2->set_piece("</s>");
  sp2->set_type(::sentencepiece::ModelProto::SentencePiece::CONTROL);

  // 3: piece with null byte "\0"
  auto* sp3 = proto.add_pieces();
  sp3->set_piece(std::string("\0", 1));
  sp3->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  sp3->set_score(-1.0);

  // 4..259: 256 byte pieces <0x00> .. <0xFF>
  for (int i = 0; i < 256; ++i) {
    char buf[16];
    snprintf(buf, sizeof(buf), "<0x%02X>", i);
    auto* sp = proto.add_pieces();
    sp->set_piece(buf);
    sp->set_type(::sentencepiece::ModelProto::SentencePiece::BYTE);
    sp->set_score(-2.0);
  }

  // 1. Without treat_null_byte_as_unused, conversion must fail.
  ConverterOptions default_options;
  auto fail_res = ToFlatbuffer(proto, default_options);
  EXPECT_FALSE(fail_res.ok());

  // 2. With treat_null_byte_as_unused = true, conversion must succeed.
  ConverterOptions sanitize_options;
  sanitize_options.treat_null_byte_as_unused = true;
  auto ok_res = ToFlatbuffer(proto, sanitize_options);
  ASSERT_TRUE(ok_res.ok());

  // Verify processor can load and encode correctly.
  SentencePieceLiteProcessor processor(*ok_res);
  ASSERT_EQ(processor.status(), StatusCode::kOk);

  // Encoding a string with '\0' should fall back to byte token for 0x00 (ID =
  // 4).
  std::string input(1, '\0');
  std::vector<int> ids;
  EXPECT_EQ(processor.EncodeNormalized(input, &ids), StatusCode::kOk);
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], 4);  // <0x00> is at index 4

  // 3. If byte_fallback is false, treat_null_byte_as_unused = true must fail.
  proto.mutable_trainer_spec()->set_byte_fallback(false);
  auto no_fb_res = ToFlatbuffer(proto, sanitize_options);
  EXPECT_FALSE(no_fb_res.ok());
}

TEST(SentencePieceLiteTest, ByteFallbackWithoutUnkId) {
  ::sentencepiece::ModelProto proto;
  auto* trainer_spec = proto.mutable_trainer_spec();
  trainer_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  trainer_spec->set_unk_id(-1);
  trainer_spec->set_byte_fallback(true);

  auto* p0 = proto.add_pieces();
  p0->set_piece("a");
  p0->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  p0->set_score(-1.0);

  // Add 256 byte pieces
  for (int i = 0; i < 256; ++i) {
    char buf[16];
    snprintf(buf, sizeof(buf), "<0x%02X>", i);
    auto* sp = proto.add_pieces();
    sp->set_piece(buf);
    sp->set_type(::sentencepiece::ModelProto::SentencePiece::BYTE);
    sp->set_score(-2.0);
  }

  // 1. With byte_fallback = true and unk_id = -1, loading must succeed.
  auto ok_res = ToFlatbuffer(proto);
  ASSERT_TRUE(ok_res.ok());
  SentencePieceLiteProcessor processor(*ok_res);
  EXPECT_EQ(processor.status(), StatusCode::kOk);
  EXPECT_EQ(processor.unk_id(), -1);
  EXPECT_EQ(processor.IdToPiece(-1), "<unk>");

  // Unknown character 'b' (0x62) falls back to byte token (1 + 0x62 = 99).
  std::vector<int> ids;
  EXPECT_EQ(processor.EncodeNormalized("b", &ids), StatusCode::kOk);
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], 1 + 0x62);

  std::string decoded;
  EXPECT_EQ(processor.Decode(ids, &decoded), StatusCode::kOk);
  EXPECT_EQ(decoded, "b");

  // 2. Without byte fallback pieces and unk_id = -1, initialization must fail.
  ::sentencepiece::ModelProto no_fb_proto;
  auto* no_fb_spec = no_fb_proto.mutable_trainer_spec();
  no_fb_spec->set_model_type(::sentencepiece::TrainerSpec::UNIGRAM);
  no_fb_spec->set_unk_id(-1);
  no_fb_spec->set_byte_fallback(false);
  auto* p_single = no_fb_proto.add_pieces();
  p_single->set_piece("a");
  p_single->set_type(::sentencepiece::ModelProto::SentencePiece::NORMAL);
  p_single->set_score(-1.0);

  auto no_fb_res = ToFlatbuffer(no_fb_proto);
  ASSERT_TRUE(no_fb_res.ok());
  SentencePieceLiteProcessor bad_processor(*no_fb_res);
  EXPECT_NE(bad_processor.status(), StatusCode::kOk);
}

}  // namespace
}  // namespace sentencepiece::lite
