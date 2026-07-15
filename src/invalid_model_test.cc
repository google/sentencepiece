// Copyright 2026 Google Inc.
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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "testharness.h"
#include "third_party/absl/base/internal/endian.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/darts_clone/darts.h"
#include "unigram_model.h"
#include "util.h"

namespace sentencepiece {
namespace {

std::string GetTestDataPath(absl::string_view filename) {
  return util::JoinPath(::testing::SrcDir(), filename);
}

absl::Status LoadModelWithModifiedCharsmap(const ModelProto& base_model,
                                           const std::string& new_charsmap) {
  ModelProto model_proto = base_model;
  auto* spec = model_proto.mutable_normalizer_spec();
  spec->set_precompiled_charsmap(new_charsmap);
  SentencePieceProcessor sp;
  return sp.Load(model_proto);
}

// Test for GitHub issue #1269 (Out-of-bounds read in Darts trie validation)
TEST(SentencePieceProcessorTest, RejectCorruptedModel1269) {
  // 1. Read base model
  std::string model_path = GetTestDataPath("test_oss_model.model");
  std::ifstream ifs(model_path, std::ios::binary);
  ModelProto model_proto;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs));

  // 2. Corrupt precompiled_charsmap to trigger OOB read in darts
  auto* spec = model_proto.mutable_normalizer_spec();
  std::string charsmap = spec->precompiled_charsmap();
  ASSERT_GE(charsmap.size(), 8);

  // Overwrite trie root unit 0 with a crafted value:
  // bit 31 set (0x80000000) -> bypass validate() if it exists and uses label()
  // > 0xFF offset set to large value to steer traversal out of bounds.
  uint32_t mal = (0x80000000 | (0x00200000 << 10)) & 0xFFFFFFFF;
  absl::little_endian::Store32(const_cast<char*>(charsmap.data()) + 4, mal);
  spec->set_precompiled_charsmap(charsmap);

  // 3. Load the corrupted model
  SentencePieceProcessor sp;
  // This should FAIL now that we have validation.
  absl::Status status = sp.Load(model_proto);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "precompiled_charsmap is invalid.");
}

// Test for GitHub issue #1263 (Stack overflow on over-long piece string)
TEST(SentencePieceProcessorTest, ReproduceIssue1263) {
  ModelProto model_proto;
  // Add pieces
  auto* piece1 = model_proto.add_pieces();
  piece1->set_piece("<unk>");
  piece1->set_score(0.0);
  piece1->set_type(ModelProto::SentencePiece::UNKNOWN);

  auto* piece2 = model_proto.add_pieces();
  piece2->set_piece("<s>");
  piece2->set_score(0.0);
  piece2->set_type(ModelProto::SentencePiece::CONTROL);

  // Trigger piece: very long string (200KB)
  // This causes stack overflow during Load on unpatched versions.
  auto* piece3 = model_proto.add_pieces();
  piece3->set_piece(std::string(200000, 'a'));
  piece3->set_score(-1.0);
  piece3->set_type(ModelProto::SentencePiece::NORMAL);

  // TrainerSpec
  auto* trainer_spec = model_proto.mutable_trainer_spec();
  trainer_spec->set_model_type(TrainerSpec::UNIGRAM);

  // NormalizerSpec
  auto* normalizer_spec = model_proto.mutable_normalizer_spec();
  normalizer_spec->set_name("identity");

  SentencePieceProcessor sp;
  // This should fail safely now that we have the limit.
  absl::Status status = sp.Load(model_proto);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "piece is too long.");
}

TEST(SentencePieceProcessorTest, CharsmapValidationTests) {
  // Read base model once
  std::string model_path = GetTestDataPath("test_oss_model.model");
  std::ifstream ifs(model_path, std::ios::binary);
  ModelProto base_model;
  ASSERT_TRUE(base_model.ParseFromIstream(&ifs));

  const std::string& original_charsmap =
      base_model.normalizer_spec().precompiled_charsmap();
  ASSERT_GE(original_charsmap.size(), 4);

  // 1. Blob too small
  {
    absl::Status status = LoadModelWithModifiedCharsmap(base_model, "abc");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(status.message(), "Blob for normalization rule is broken.");
  }

  // 2. Trie size exceeds blob
  {
    std::string charsmap = original_charsmap;
    // Set size to a very large value (1MB)
    uint32_t bad_size = 1024 * 1024;
    absl::little_endian::Store32(const_cast<char*>(charsmap.data()), bad_size);
    absl::Status status = LoadModelWithModifiedCharsmap(base_model, charsmap);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(status.message(), "Trie data size exceeds the input blob size.");
  }

  // 3. Trie size less than 1024 (e.g. 512)
  {
    std::string charsmap = original_charsmap;
    uint32_t bad_size = 512;
    absl::little_endian::Store32(const_cast<char*>(charsmap.data()), bad_size);
    absl::Status status = LoadModelWithModifiedCharsmap(base_model, charsmap);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(status.message(), "Trie data size is not divisible by 1024.");
  }

  // 4. Trie size not multiple of 1024 (e.g. 1025)
  {
    std::string charsmap = original_charsmap;
    uint32_t bad_size = 1025;
    absl::little_endian::Store32(const_cast<char*>(charsmap.data()), bad_size);
    absl::Status status = LoadModelWithModifiedCharsmap(base_model, charsmap);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(status.message(), "Trie data size is not divisible by 1024.");
  }

  // 5. Normalized block not null terminated
  {
    std::string charsmap = original_charsmap;
    // Corrupt the last byte of the blob
    charsmap.back() = 'a';  // original should have been '\0'
    absl::Status status = LoadModelWithModifiedCharsmap(base_model, charsmap);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(status.message(), "normalized block must be null terminated.");
  }
}

TEST(SentencePieceProcessorTest, RejectModelWithOOBCharsmapValue) {
  std::string model_path = GetTestDataPath("test_oss_model.model");
  std::ifstream ifs(model_path, std::ios::binary);
  ModelProto model_proto;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs));

  auto* spec = model_proto.mutable_normalizer_spec();
  std::string charsmap = spec->precompiled_charsmap();
  ASSERT_GE(charsmap.size(), 8);

  uint32_t trie_blob_size = absl::little_endian::Load32(charsmap.data());

  size_t num_units = trie_blob_size / 4;
  bool found_leaf = false;
  for (size_t i = 0; i < num_units; ++i) {
    uint32_t unit = absl::little_endian::Load32(charsmap.data() + 4 + i * 4);
    if (unit & 0x80000000) {  // Leaf node
      // Corrupt it by setting value to maximum possible (0x7FFFFFFF)
      uint32_t corrupted_unit = 0xFFFFFFFF;
      absl::little_endian::Store32(
          const_cast<char*>(charsmap.data()) + 4 + i * 4, corrupted_unit);
      found_leaf = true;
      break;
    }
  }
  ASSERT_TRUE(found_leaf);
  spec->set_precompiled_charsmap(charsmap);

  SentencePieceProcessor sp;
  absl::Status status = sp.Load(model_proto);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "precompiled_charsmap is invalid.");
}

}  // namespace
}  // namespace sentencepiece
