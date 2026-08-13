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

#include <string>
#include <vector>

#include "filesystem.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "testharness.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
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
      util::JoinPath(::testing::SrcDir(), filename));
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
  const std::string test_model_file =
      util::JoinPath(::testing::SrcDir(), "botchan_en_unigram_1000.model");

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
  const std::string test_model_file =
      util::JoinPath(::testing::SrcDir(), "botchan_en_unigram_1000.model");

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
      util::JoinPath(::testing::SrcDir(), "botchan_en_bpe_1000.model");

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
  const std::string test_model_file = util::JoinPath(
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
  const std::string test_model_file = util::JoinPath(
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
  const std::string test_model_file = util::JoinPath(
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
  std::string test_model_file = util::JoinPath(
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
      util::JoinPath(::testing::SrcDir(), "botchan_en_bpe_1000.model");
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

}  // namespace sentencepiece
