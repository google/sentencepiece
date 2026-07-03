// Copyright 2024 nlcodec / Thamme Gowda
// Tests for the nlcodec fast BPE trainer integration.

#include <set>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "util.h"

ABSL_DECLARE_FLAG(bool, nlcodec_bpe);

namespace sentencepiece {
namespace nlcodec {
namespace {

static constexpr char kTestInputData[] = "wagahaiwa_nekodearu.txt";

// Helper: train BPE and return set of learned pieces (excluding meta tokens).
std::set<std::string> TrainAndGetPieces(
    const std::string &input_file, int vocab_size, bool use_nlcodec) {
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(),
                     use_nlcodec ? "nlcodec_model" : "default_model");

  absl::SetFlag(&FLAGS_nlcodec_bpe, use_nlcodec);

  EXPECT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix,
                       " --input=", input_file,
                       " --vocab_size=", std::to_string(vocab_size),
                       " --normalization_rule_name=identity",
                       " --model_type=bpe",
                       " --max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(model_prefix + ".model").ok());

  std::set<std::string> pieces;
  for (int i = 0; i < sp.GetPieceSize(); ++i) {
    if (!sp.IsUnknown(i) && !sp.IsControl(i))
      pieces.insert(sp.IdToPiece(i));
  }
  return pieces;
}

TEST(NlcodecBPETest, ProducesValidModel) {
  const std::string input =
      util::JoinPath(::testing::SrcDir(), kTestInputData);
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "nlcodec_test");

  absl::SetFlag(&FLAGS_nlcodec_bpe, true);
  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix,
                       " --input=", input,
                       " --vocab_size=3000",
                       " --normalization_rule_name=identity",
                       " --model_type=bpe",
                       " --max_sentence_length=2048"))
          .ok());
  absl::SetFlag(&FLAGS_nlcodec_bpe, false);

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());
  EXPECT_EQ(3000, sp.GetPieceSize());

  std::vector<std::string> tok;
  ASSERT_TRUE(sp.Encode("hello world", &tok).ok());
  EXPECT_FALSE(tok.empty());

  std::string decoded;
  ASSERT_TRUE(sp.Decode(tok, &decoded).ok());
  EXPECT_FALSE(decoded.empty());
}

TEST(NlcodecBPETest, VocabSizeMatchesDefault) {
  const std::string input =
      util::JoinPath(::testing::SrcDir(), kTestInputData);

  auto default_pieces = TrainAndGetPieces(input, 3000, false);
  auto nlcodec_pieces = TrainAndGetPieces(input, 3000, true);

  EXPECT_EQ(default_pieces.size(), nlcodec_pieces.size());
}

TEST(NlcodecBPETest, EncodesDecodesCorrectly) {
  const std::string input =
      util::JoinPath(::testing::SrcDir(), kTestInputData);
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "nlcodec_roundtrip");

  absl::SetFlag(&FLAGS_nlcodec_bpe, true);
  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix,
                       " --input=", input,
                       " --vocab_size=3000",
                       " --normalization_rule_name=identity",
                       " --model_type=bpe",
                       " --max_sentence_length=2048"))
          .ok());
  absl::SetFlag(&FLAGS_nlcodec_bpe, false);

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());

  const std::vector<std::string> test_strings = {
      "abracadabra",
      "hello world",
      "the quick brown fox",
  };

  for (const auto &s : test_strings) {
    std::vector<int> ids;
    ASSERT_TRUE(sp.Encode(s, &ids).ok());
    EXPECT_FALSE(ids.empty());

    std::string decoded;
    ASSERT_TRUE(sp.Decode(ids, &decoded).ok());
    EXPECT_FALSE(decoded.empty()) << "Decode produced empty string for: " << s;
  }
}

TEST(NlcodecBPETest, VocabOverlapsWithDefault) {
  const std::string input =
      util::JoinPath(::testing::SrcDir(), kTestInputData);

  auto default_pieces = TrainAndGetPieces(input, 3000, false);
  auto nlcodec_pieces = TrainAndGetPieces(input, 3000, true);

  int overlap = 0;
  for (const auto &p : default_pieces) {
    if (nlcodec_pieces.count(p)) overlap++;
  }

  double overlap_ratio =
      static_cast<double>(overlap) / default_pieces.size();

  EXPECT_GT(overlap_ratio, 0.5)
      << "Overlap too low: " << overlap << "/"
      << default_pieces.size() << " = " << overlap_ratio;

  LOG(INFO) << "Vocab overlap: " << overlap << "/"
            << default_pieces.size() << " = "
            << static_cast<int>(overlap_ratio * 100) << "%";
}

}  // namespace
}  // namespace nlcodec
}  // namespace sentencepiece
