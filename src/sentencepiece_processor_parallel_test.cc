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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "filesystem.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

namespace sentencepiece {

void ExpectSptEqual(const SentencePieceText& expected,
                    const SentencePieceText& actual) {
  EXPECT_EQ(expected.text(), actual.text());
  EXPECT_EQ(expected.pieces_size(), actual.pieces_size());
  EXPECT_NEAR(expected.score(), actual.score(), 1e-5);
  for (int i = 0; i < expected.pieces_size() && i < actual.pieces_size(); ++i) {
    const auto& e_piece = expected.pieces(i);
    const auto& a_piece = actual.pieces(i);
    EXPECT_EQ(e_piece.piece(), a_piece.piece()) << "at index " << i;
    EXPECT_EQ(e_piece.id(), a_piece.id()) << "at index " << i;
    EXPECT_EQ(e_piece.surface(), a_piece.surface()) << "at index " << i;
    EXPECT_EQ(e_piece.begin(), a_piece.begin()) << "at index " << i;
    EXPECT_EQ(e_piece.end(), a_piece.end()) << "at index " << i;
  }
}

class SentencePieceProcessorMaxLoops : public SentencePieceProcessor {
 public:
  SentencePieceProcessorMaxLoops(int max_loops) {
    //    max_reparse_loops_ = max_loops;
  }
};

std::string LoadTestData(const std::string& filename, int num_lines) {
  auto fs = filesystem::NewReadableFile(
      filesystem::JoinPath(::testing::SrcDir(), filename));
  CHECK(fs);
  CHECK_GT(num_lines, 0);
  std::string test_data, line;
  for (int i = 0; i < num_lines; ++i) {
    if (!fs->ReadLine(&line)) break;
    absl::StrAppend(&test_data, line);
    absl::StrAppend(&test_data, "\n");
  }
  return test_data;
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestEmptyString) {
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model");

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  std::vector<int> sequential_encode_ids;
  std::vector<int> parallel_encode_ids;

  // Check English tokenized correctly in parallel
  sequential_encode_ids = sp.EncodeAsIds("");

  ThreadPool thread_pool(4);

  for (const int chunk_size : {128, 256, 512}) {
    parallel_encode_ids.clear();
    CHECK_OK(
        sp.ParallelEncode("", chunk_size, thread_pool, &parallel_encode_ids));
    EXPECT_EQ(sequential_encode_ids, parallel_encode_ids);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestEn) {
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model");

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(4);

  std::vector<int> sequential_encode_ids;
  std::vector<int> parallel_encode_ids;

  // Check English tokenized correctly in parallel
  const std::string en_test_data = LoadTestData("botchan.txt", 20);
  CHECK_OK(sp.Encode(en_test_data, &sequential_encode_ids));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_ids.clear();
    CHECK_OK(sp.ParallelEncode(en_test_data, chunk_size, thread_pool,
                               &parallel_encode_ids));
    EXPECT_EQ(sequential_encode_ids, parallel_encode_ids);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestJaWithUNK) {
  // Check Japanese tokenized correctly in parallel
  const std::string test_model_file =
      filesystem::JoinPath(::testing::SrcDir(), "botchan_en_bpe_1000.model");

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(4);

  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  std::vector<int> sequential_encode_ids;
  std::vector<int> parallel_encode_ids;

  CHECK_OK(sp.Encode(ja_test_data, &sequential_encode_ids));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_ids.clear();
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &parallel_encode_ids));
    EXPECT_EQ(parallel_encode_ids.size(), sequential_encode_ids.size());
    EXPECT_EQ(sequential_encode_ids, parallel_encode_ids);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestJaWithByte) {
  // Check Japanese tokenized correctly in parallel
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "wagahaiwa_nekodearu_ja_bpe_byte_2000.model");

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(4);

  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  std::vector<int> sequential_encode_ids;
  std::vector<int> parallel_encode_ids;

  CHECK_OK(sp.Encode(ja_test_data, &sequential_encode_ids));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_ids.clear();
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &parallel_encode_ids));
    EXPECT_EQ(parallel_encode_ids.size(), sequential_encode_ids.size());
    EXPECT_EQ(sequential_encode_ids, parallel_encode_ids);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestJaWithByteIntoSPTZeroLoops) {
  // Check Japanese tokenized correctly in parallel
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "wagahaiwa_nekodearu_ja_bpe_byte_2000.model");

  SentencePieceProcessorMaxLoops sp(0);
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(4);

  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  SentencePieceText sequential_encode_spt;
  SentencePieceText parallel_encode_spt;

  CHECK_OK(sp.Encode(ja_test_data, &sequential_encode_spt));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_spt.Clear();
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &parallel_encode_spt));
    EXPECT_EQ(parallel_encode_spt.pieces_size(),
              sequential_encode_spt.pieces_size());
    ExpectSptEqual(sequential_encode_spt, parallel_encode_spt);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestJaWithByteIntoSPTOneLoop) {
  // Check Japanese tokenized correctly in parallel
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "wagahaiwa_nekodearu_ja_bpe_byte_2000.model");

  SentencePieceProcessorMaxLoops sp(1);
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(1);

  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  SentencePieceText sequential_encode_spt;
  SentencePieceText parallel_encode_spt;

  CHECK_OK(sp.Encode(ja_test_data, &sequential_encode_spt));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_spt.Clear();
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &parallel_encode_spt));
    ExpectSptEqual(sequential_encode_spt, parallel_encode_spt);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestJaWithByteIntoSPT) {
  // Check Japanese tokenized correctly in parallel
  std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "wagahaiwa_nekodearu_ja_bpe_byte_2000.model");

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(4);

  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  SentencePieceText sequential_encode_spt;
  SentencePieceText parallel_encode_spt;

  CHECK_OK(sp.Encode(ja_test_data, &sequential_encode_spt));

  for (int chunk_size = 128; chunk_size <= 512; ++chunk_size) {
    parallel_encode_spt.Clear();
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &parallel_encode_spt));
    ExpectSptEqual(sequential_encode_spt, parallel_encode_spt);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestBotchan) {
  std::string test_model_file =
      filesystem::JoinPath(::testing::SrcDir(), "botchan_en_bpe_1000.model");
  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  // Load all data.
  ThreadPool thread_pool(4);
  const std::string test_data = LoadTestData("botchan.txt", 1000000);

  SentencePieceText sequential_encode_spt;
  CHECK_OK(sp.Encode(test_data, &sequential_encode_spt));

  std::vector<int> chunk_sizes = {100, 1000, 10000};
  for (auto chunk_size : chunk_sizes) {
    SentencePieceText parallel_encode_spt;
    CHECK_OK(sp.ParallelEncode(test_data, chunk_size, thread_pool,
                               &parallel_encode_spt));
    ExpectSptEqual(sequential_encode_spt, parallel_encode_spt);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeTestExtraOptions) {
  const std::string test_model_file =
      filesystem::JoinPath(::testing::SrcDir(), "botchan_en_bpe_1000.model");
  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));
  CHECK_OK(sp.SetEncodeExtraOptions("bos:eos:reverse:unk_piece"));

  ThreadPool thread_pool(4);
  const std::string en_test_data = LoadTestData("botchan.txt", 25);
  const std::string ja_test_data = LoadTestData("wagahaiwa_nekodearu.txt", 25);

  for (const absl::string_view input :
       {absl::string_view(""), absl::string_view(en_test_data)}) {
    std::vector<int> expected_ids;
    std::vector<std::string> expected_pieces;
    SentencePieceText expected_spt;
    CHECK_OK(sp.Encode(input, &expected_ids));
    CHECK_OK(sp.Encode(input, &expected_pieces));
    CHECK_OK(sp.Encode(input, &expected_spt));

    for (const int chunk_size : {64, 128, 256}) {
      std::vector<int> actual_ids;
      std::vector<std::string> actual_pieces;
      SentencePieceText actual_spt;
      CHECK_OK(sp.ParallelEncode(input, chunk_size, thread_pool, &actual_ids));
      CHECK_OK(
          sp.ParallelEncode(input, chunk_size, thread_pool, &actual_pieces));
      CHECK_OK(sp.ParallelEncode(input, chunk_size, thread_pool, &actual_spt));
      EXPECT_EQ(expected_ids, actual_ids);
      EXPECT_EQ(expected_pieces, actual_pieces);
      ExpectSptEqual(expected_spt, actual_spt);
    }
  }

  std::vector<int> expected_ja_ids;
  std::vector<std::string> expected_ja_pieces;
  CHECK_OK(sp.Encode(ja_test_data, &expected_ja_ids));
  CHECK_OK(sp.Encode(ja_test_data, &expected_ja_pieces));
  for (const int chunk_size : {128, 256}) {
    std::vector<int> actual_ja_ids;
    std::vector<std::string> actual_ja_pieces;
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &actual_ja_ids));
    CHECK_OK(sp.ParallelEncode(ja_test_data, chunk_size, thread_pool,
                               &actual_ja_pieces));
    EXPECT_EQ(expected_ja_ids, actual_ja_ids);
    EXPECT_EQ(expected_ja_pieces, actual_ja_pieces);
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodePositiveControl) {
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model");
  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(test_model_file));

  ThreadPool thread_pool(2);
  const std::string input = "This is a positive control test for parallel encode.";
  std::vector<int> expected_ids;
  std::vector<int> actual_ids;
  CHECK_OK(sp.Encode(input, &expected_ids));
  CHECK_OK(sp.ParallelEncode(input, 8, thread_pool, &actual_ids));
  EXPECT_EQ(expected_ids, actual_ids);
}

TEST(SentencepieceProcessorTest, ParallelEncodeRejectsNonPositiveMaxPieceLength) {
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model");
  ModelProto base_model_proto;
  CHECK_OK(io::LoadModelProto(test_model_file, &base_model_proto));

  ThreadPool thread_pool(2);
  for (const int invalid_max_len : {-2, 0}) {
    ModelProto model_proto = base_model_proto;
    model_proto.mutable_trainer_spec()->set_max_sentencepiece_length(
        invalid_max_len);
    SentencePieceProcessor sp;
    CHECK_OK(sp.Load(model_proto));

    std::vector<int> ids;
    EXPECT_FALSE(
        sp.ParallelEncode("This is a test of parallel encoding.", 2,
                          thread_pool, &ids)
            .ok());
  }
}

TEST(SentencepieceProcessorTest, ParallelEncodeLargeMaxPieceLengthDoesNotOverflow) {
  const std::string test_model_file = filesystem::JoinPath(
      ::testing::SrcDir(), "botchan_en_unigram_1000.model");
  ModelProto model_proto;
  CHECK_OK(io::LoadModelProto(test_model_file, &model_proto));
  model_proto.mutable_trainer_spec()->set_max_sentencepiece_length(
      std::numeric_limits<int32_t>::max());

  SentencePieceProcessor sp;
  CHECK_OK(sp.Load(model_proto));

  ThreadPool thread_pool(2);
  const std::string input = "This is a forty character test sentence.";
  std::vector<int> expected_ids;
  std::vector<int> actual_ids;
  CHECK_OK(sp.Encode(input, &expected_ids));
  CHECK_OK(sp.ParallelEncode(input, 20, thread_pool, &actual_ids));
  EXPECT_EQ(expected_ids, actual_ids);
}

}  // namespace sentencepiece
