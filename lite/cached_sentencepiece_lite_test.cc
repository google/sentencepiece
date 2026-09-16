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

#include "cached_sentencepiece_lite.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "sentencepiece_lite.h"
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

class CachedSentencePieceLiteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string model_path = "test_data/botchan_1000_bpe.model";
    const std::string model_bytes = ReadFile(model_path);
    ::sentencepiece::ModelProto proto;
    ASSERT_TRUE(proto.ParseFromString(model_bytes));
    auto fbs_bytes_or = ::sentencepiece::lite::ToFlatbuffer(proto);
    ASSERT_TRUE(fbs_bytes_or.ok()) << fbs_bytes_or.status();
    model_buffer_ = std::move(*fbs_bytes_or);
    processor_ = std::make_unique<SentencePieceLiteProcessor>(model_buffer_);
    ASSERT_EQ(processor_->status(), StatusCode::kOk);
  }

  std::string model_buffer_;
  std::unique_ptr<SentencePieceLiteProcessor> processor_;
};

TEST_F(CachedSentencePieceLiteTest, DynamicSeedAndExplicitSeed) {
  TokenCache cache1;
  TokenCache cache2;
  EXPECT_NE(cache1.seed(), 0ULL);
  EXPECT_NE(cache2.seed(), 0ULL);

  TokenCache explicit_cache(0x123456789ABCDEF0ULL);
  EXPECT_EQ(explicit_cache.seed(), 0x123456789ABCDEF0ULL);
}

TEST_F(CachedSentencePieceLiteTest, CapacityAndCustomSizing) {
  TokenCache default_cache;
  EXPECT_EQ(default_cache.num_pairs(),
            static_cast<size_t>(TokenCache::Capacity::kMedium));
  EXPECT_EQ(default_cache.capacity(),
            static_cast<size_t>(TokenCache::Capacity::kMedium) * 2);

  TokenCache tiny_cache(TokenCache::Capacity::kTiny, 0x1234ULL);
  EXPECT_EQ(tiny_cache.num_pairs(), 1024U);
  EXPECT_EQ(tiny_cache.capacity(), 2048U);
  EXPECT_EQ(tiny_cache.seed(), 0x1234ULL);

  TokenCache small_cache(TokenCache::Capacity::kSmall);
  EXPECT_EQ(small_cache.num_pairs(), 4096U);
  EXPECT_EQ(small_cache.capacity(), 8192U);
  EXPECT_NE(small_cache.seed(), 0ULL);

  TokenCache med_cache(TokenCache::Capacity::kMedium);
  EXPECT_EQ(med_cache.num_pairs(), 16384U);
  EXPECT_EQ(med_cache.capacity(), 32768U);

  TokenCache large_cache(TokenCache::Capacity::kLarge);
  EXPECT_EQ(large_cache.num_pairs(), 131072U);
  EXPECT_EQ(large_cache.capacity(), 262144U);
}

TEST_F(CachedSentencePieceLiteTest, EncodeEquivalence) {
  if (processor_ == nullptr || processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model file not available";
  }

  const std::vector<std::string> test_texts = {
      "The quick brown fox jumps over the lazy dog.",
      "Hello world! This is a test of cached tokenization.",
      "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
      "Repeat word Repeat word Repeat word Repeat word.",
      "",
  };

  TokenCache cache;
  for (const auto& text : test_texts) {
    std::vector<int> cached_ids1;
    EXPECT_EQ(
        CachedSentencePieceLite::Encode(*processor_, cache, text, &cached_ids1),
        StatusCode::kOk);

    // Repeat Encode to test cache hits and ensure identical results
    std::vector<int> cached_ids2;
    EXPECT_EQ(
        CachedSentencePieceLite::Encode(*processor_, cache, text, &cached_ids2),
        StatusCode::kOk);
    EXPECT_EQ(cached_ids1, cached_ids2);

    // Test EncodeNormalized
    std::string normalized;
    ASSERT_EQ(processor_->Normalize(text, &normalized), StatusCode::kOk);
    std::vector<int> norm_ids;
    EXPECT_EQ(CachedSentencePieceLite::EncodeNormalized(*processor_, cache,
                                                        normalized, &norm_ids),
              StatusCode::kOk);
    EXPECT_EQ(norm_ids, cached_ids1);
  }
}

TEST_F(CachedSentencePieceLiteTest, CacheClear) {
  if (processor_ == nullptr || processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model file not available";
  }

  TokenCache cache;
  const std::string text = "Test clearing the token cache cleanly.";
  std::vector<int> ids1;
  EXPECT_EQ(CachedSentencePieceLite::Encode(*processor_, cache, text, &ids1),
            StatusCode::kOk);

  cache.Clear();

  std::vector<int> ids2;
  EXPECT_EQ(CachedSentencePieceLite::Encode(*processor_, cache, text, &ids2),
            StatusCode::kOk);
  EXPECT_EQ(ids1, ids2);
}

TEST_F(CachedSentencePieceLiteTest, ProcessorIsolationWithSharedCache) {
  if (processor_ == nullptr || processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model file not available";
  }

  const std::string model_path2 =
      "test_data/"
      "wagahaiwa_nekodearu_2000_bpe_byte.model";
  const std::string model_bytes2 = ReadFile(model_path2);
  ::sentencepiece::ModelProto proto2;
  ASSERT_TRUE(proto2.ParseFromString(model_bytes2));
  auto fbs_bytes_or2 = ::sentencepiece::lite::ToFlatbuffer(proto2);
  ASSERT_TRUE(fbs_bytes_or2.ok()) << fbs_bytes_or2.status();
  std::string model_buffer2 = std::move(*fbs_bytes_or2);
  SentencePieceLiteProcessor processor2(model_buffer2);
  ASSERT_EQ(processor2.status(), StatusCode::kOk);

  // Shared single TokenCache across different processor instances
  TokenCache shared_cache;
  const std::string text = "Hello world! Test multi-model cache isolation.";

  std::vector<int> p1_expected;
  std::vector<int> p2_expected;
  ASSERT_EQ(processor_->Encode(text, &p1_expected), StatusCode::kOk);
  ASSERT_EQ(processor2.Encode(text, &p2_expected), StatusCode::kOk);
  EXPECT_NE(p1_expected, p2_expected);

  std::vector<int> p1_cached;
  std::vector<int> p2_cached;
  EXPECT_EQ(CachedSentencePieceLite::Encode(*processor_, shared_cache, text,
                                            &p1_cached),
            StatusCode::kOk);
  EXPECT_EQ(CachedSentencePieceLite::Encode(processor2, shared_cache, text,
                                            &p2_cached),
            StatusCode::kOk);

  EXPECT_EQ(p1_cached, p1_expected);
  EXPECT_EQ(p2_cached, p2_expected);
}

TEST_F(CachedSentencePieceLiteTest, ThreadLocalUsage) {
  if (processor_ == nullptr || processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model file not available";
  }

  const std::string text =
      "Multi-threaded tokenization test with thread_local TokenCache.";
  std::vector<int> expected_ids;
  ASSERT_EQ(processor_->Encode(text, &expected_ids), StatusCode::kOk);

  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &text, &expected_ids]() {
      // thread_local TokenCache with custom Medium capacity (2 MB)
      static thread_local TokenCache tl_cache(TokenCache::Capacity::kMedium);
      for (int i = 0; i < kItersPerThread; ++i) {
        std::vector<int> ids;
        EXPECT_EQ(
            CachedSentencePieceLite::Encode(*processor_, tl_cache, text, &ids),
            StatusCode::kOk);
        EXPECT_EQ(ids, expected_ids);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(CachedSentencePieceLiteTest, MultiThreadMultiModelThreadLocalSharing) {
  if (processor_ == nullptr || processor_->status() != StatusCode::kOk) {
    GTEST_SKIP() << "Model file not available";
  }

  const std::string model_path2 =
      "test_data/"
      "wagahaiwa_nekodearu_2000_bpe_byte.model";
  const std::string model_bytes2 = ReadFile(model_path2);
  ::sentencepiece::ModelProto proto2;
  ASSERT_TRUE(proto2.ParseFromString(model_bytes2));
  auto fbs_bytes_or2 = ::sentencepiece::lite::ToFlatbuffer(proto2);
  ASSERT_TRUE(fbs_bytes_or2.ok()) << fbs_bytes_or2.status();
  std::string model_buffer2 = std::move(*fbs_bytes_or2);
  SentencePieceLiteProcessor processor2(model_buffer2);
  ASSERT_EQ(processor2.status(), StatusCode::kOk);

  const std::vector<std::string> test_texts = {
      "The quick brown fox jumps over the lazy dog.",
      "hello world! Multi-model thread-local cache sharing test.",
      "SentencePiece tokenization with high performance chunk caching.",
      "Repeat word Repeat word Repeat word Repeat word.",
      "short test phrase",
      "",
  };

  struct ExpectedOutput {
    std::vector<int> p1_ids;
    std::vector<int> p2_ids;
  };
  std::vector<ExpectedOutput> expected_outputs(test_texts.size());
  for (size_t i = 0; i < test_texts.size(); ++i) {
    ASSERT_EQ(processor_->Encode(test_texts[i], &expected_outputs[i].p1_ids),
              StatusCode::kOk);
    ASSERT_EQ(processor2.Encode(test_texts[i], &expected_outputs[i].p2_ids),
              StatusCode::kOk);
    if (!test_texts[i].empty()) {
      EXPECT_NE(expected_outputs[i].p1_ids, expected_outputs[i].p2_ids);
    }
  }

  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &processor2, &test_texts, &expected_outputs,
                          t]() {
      // Single shared thread_local TokenCache instance per worker thread.
      static thread_local TokenCache tl_cache(TokenCache::Capacity::kMedium);

      for (int iter = 0; iter < kItersPerThread; ++iter) {
        for (size_t i = 0; i < test_texts.size(); ++i) {
          const auto& text = test_texts[i];
          const auto& expected = expected_outputs[i];

          // Alternate order based on thread index and iteration to test
          // arbitrary interleaved access patterns.
          if ((t + iter) % 2 == 0) {
            std::vector<int> p1_ids;
            EXPECT_EQ(CachedSentencePieceLite::Encode(*processor_, tl_cache,
                                                      text, &p1_ids),
                      StatusCode::kOk);
            EXPECT_EQ(p1_ids, expected.p1_ids);

            std::vector<int> p2_ids;
            EXPECT_EQ(CachedSentencePieceLite::Encode(processor2, tl_cache,
                                                      text, &p2_ids),
                      StatusCode::kOk);
            EXPECT_EQ(p2_ids, expected.p2_ids);
          } else {
            std::vector<int> p2_ids;
            EXPECT_EQ(CachedSentencePieceLite::Encode(processor2, tl_cache,
                                                      text, &p2_ids),
                      StatusCode::kOk);
            EXPECT_EQ(p2_ids, expected.p2_ids);

            std::vector<int> p1_ids;
            EXPECT_EQ(CachedSentencePieceLite::Encode(*processor_, tl_cache,
                                                      text, &p1_ids),
                      StatusCode::kOk);
            EXPECT_EQ(p1_ids, expected.p1_ids);
          }

          // Also verify EncodeNormalized with interleaved models.
          std::string norm1, norm2;
          ASSERT_EQ(processor_->Normalize(text, &norm1), StatusCode::kOk);
          ASSERT_EQ(processor2.Normalize(text, &norm2), StatusCode::kOk);

          std::vector<int> p1_norm_ids;
          EXPECT_EQ(CachedSentencePieceLite::EncodeNormalized(
                        *processor_, tl_cache, norm1, &p1_norm_ids),
                    StatusCode::kOk);
          EXPECT_EQ(p1_norm_ids, expected.p1_ids);

          std::vector<int> p2_norm_ids;
          EXPECT_EQ(CachedSentencePieceLite::EncodeNormalized(
                        processor2, tl_cache, norm2, &p2_norm_ids),
                    StatusCode::kOk);
          EXPECT_EQ(p2_norm_ids, expected.p2_ids);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(CachedSentencePieceLiteTest, MakeChunkTagProperties) {
  // 1. Empty string returns 0.
  EXPECT_EQ(TokenCache::MakeChunkTag(""), 0);

  // 2. All 256 single byte values map to 256 unique tags (strict bijection).
  std::vector<uint16_t> single_byte_tags;
  single_byte_tags.reserve(256);
  for (int i = 0; i < 256; ++i) {
    const char c = static_cast<char>(i);
    single_byte_tags.push_back(
        TokenCache::MakeChunkTag(std::string_view(&c, 1)));
  }
  std::sort(single_byte_tags.begin(), single_byte_tags.end());
  auto it = std::unique(single_byte_tags.begin(), single_byte_tags.end());
  EXPECT_EQ(std::distance(single_byte_tags.begin(), it), 256)
      << "All 256 single-byte character tags must be strictly unique.";

  // 3. Different string lengths produce different tags even with same
  // characters.
  EXPECT_NE(TokenCache::MakeChunkTag("a"), TokenCache::MakeChunkTag("aa"));
  EXPECT_NE(TokenCache::MakeChunkTag("aa"), TokenCache::MakeChunkTag("aaa"));
  EXPECT_NE(TokenCache::MakeChunkTag("aaa"), TokenCache::MakeChunkTag("aaaa"));

  // 4. Boundary character differences produce distinct tags.
  EXPECT_NE(TokenCache::MakeChunkTag("cat"), TokenCache::MakeChunkTag("car"));
  EXPECT_NE(TokenCache::MakeChunkTag("cat"), TokenCache::MakeChunkTag("bat"));
}

TEST_F(CachedSentencePieceLiteTest, TagCollisionProtection) {
  TokenCache cache;
  constexpr uint64_t kSimulatedCollidingHash = 0xDEADBEEFCAFE1234ULL;
  const uint16_t tag1 = 0x1111;
  const uint16_t tag2 = 0x2222;
  const int32_t ids1[] = {10, 20};
  const int32_t ids2[] = {30, 40, 50};

  cache.Insert(kSimulatedCollidingHash, tag1, ids1, 2);

  // 1. Lookup with matching hash but wrong tag must be rejected.
  size_t count = 0;
  EXPECT_EQ(cache.Lookup(kSimulatedCollidingHash, tag2, &count), nullptr);
  EXPECT_EQ(cache.Lookup(kSimulatedCollidingHash, 0x9999, &count), nullptr);

  // 2. Lookup with matching hash and matching tag succeeds.
  const int32_t* result1 = cache.Lookup(kSimulatedCollidingHash, tag1, &count);
  ASSERT_NE(result1, nullptr);
  ASSERT_EQ(count, 2U);
  EXPECT_EQ(result1[0], 10);
  EXPECT_EQ(result1[1], 20);

  // 3. Inserting a second entry with identical hash but distinct tag occupies
  // the 2-way associative sibling slot rather than overwriting slot 0.
  cache.Insert(kSimulatedCollidingHash, tag2, ids2, 3);

  const int32_t* res1_again =
      cache.Lookup(kSimulatedCollidingHash, tag1, &count);
  ASSERT_NE(res1_again, nullptr);
  EXPECT_EQ(count, 2U);
  EXPECT_EQ(res1_again[0], 10);
  EXPECT_EQ(res1_again[1], 20);

  const int32_t* res2 = cache.Lookup(kSimulatedCollidingHash, tag2, &count);
  ASSERT_NE(res2, nullptr);
  EXPECT_EQ(count, 3U);
  EXPECT_EQ(res2[0], 30);
  EXPECT_EQ(res2[1], 40);
  EXPECT_EQ(res2[2], 50);
}

}  // namespace
}  // namespace sentencepiece::lite
