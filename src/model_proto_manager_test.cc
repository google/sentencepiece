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
// limitations under the License.!

#include "model_proto_manager.h"

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "builder.h"
#include "gtest/gtest.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"

namespace sentencepiece {
namespace {

ModelProto MakeBaseModelProto(
    TrainerSpec::ModelType type = TrainerSpec::UNIGRAM) {
  ModelProto model_proto;
  auto* trainer_spec = model_proto.mutable_trainer_spec();
  trainer_spec->set_model_type(type);
  trainer_spec->set_unk_id(0);
  trainer_spec->set_bos_id(1);
  trainer_spec->set_eos_id(2);
  trainer_spec->set_pad_id(-1);

  auto* normalizer_spec = model_proto.mutable_normalizer_spec();
  normalizer_spec->set_name("identity");
  normalizer_spec->set_add_dummy_prefix(true);
  normalizer_spec->set_remove_extra_whitespaces(true);
  normalizer_spec->set_escape_whitespaces(true);

  auto add_piece = [&](absl::string_view piece, float score,
                       ModelProto::SentencePiece::Type piece_type) {
    auto* sp = model_proto.add_pieces();
    sp->set_piece(piece.data(), piece.size());
    if (piece_type == ModelProto::SentencePiece::NORMAL) {
      sp->set_score(score);
    } else {
      sp->set_type(piece_type);
      if (score != 0.0f) {
        sp->set_score(score);
      }
    }
  };

  add_piece("<unk>", 0.0f, ModelProto::SentencePiece::UNKNOWN);
  add_piece("<s>", 0.0f, ModelProto::SentencePiece::CONTROL);
  add_piece("</s>", 0.0f, ModelProto::SentencePiece::CONTROL);
  add_piece("\xe2\x96\x81", -1.0f, ModelProto::SentencePiece::NORMAL);
  add_piece("a", -2.0f, ModelProto::SentencePiece::NORMAL);
  add_piece("b", -3.0f, ModelProto::SentencePiece::NORMAL);
  if (type != TrainerSpec::CHAR) {
    add_piece("ab", -4.0f, ModelProto::SentencePiece::NORMAL);
  }
  add_piece("<user>", 0.0f, ModelProto::SentencePiece::USER_DEFINED);
  add_piece("<unused>", -5.0f, ModelProto::SentencePiece::UNUSED);

  trainer_spec->set_vocab_size(model_proto.pieces_size());
  return model_proto;
}

TEST(ModelProtoManagerTest, UninitializedStateTest) {
  ModelProtoManager manager;
  EXPECT_EQ(manager.lite_processor(), nullptr);
  EXPECT_EQ(manager.model_proto().pieces_size(), 0);
  EXPECT_EQ(&manager.model_proto(), &ModelProto::default_instance());
  EXPECT_EQ(manager.serialized_model_proto(), "");
}

TEST(ModelProtoManagerTest, LoadAndOnTheFlyReconstructionTest) {
  const ModelProto original = MakeBaseModelProto(TrainerSpec::UNIGRAM);
  const std::string expected_serialized = original.SerializeAsString();

  ModelProtoManager manager;
  auto input = std::make_unique<ModelProto>(original);
  ASSERT_TRUE(manager.Load(std::move(input)).ok());
  EXPECT_EQ(input, nullptr);

  // Verify trainer_spec, normalizer_spec, and cached min NORMAL score.
  EXPECT_EQ(manager.trainer_spec().SerializeAsString(),
            original.trainer_spec().SerializeAsString());
  EXPECT_EQ(manager.normalizer_spec().SerializeAsString(),
            original.normalizer_spec().SerializeAsString());
  EXPECT_FLOAT_EQ(manager.min_score(), -4.0f);

  // Verify lite_processor is initialized and populated.
  ASSERT_NE(manager.lite_processor(), nullptr);
  EXPECT_EQ(static_cast<int>(manager.lite_processor()->vocab_size()),
            original.pieces_size());
  for (int i = 0; i < original.pieces_size(); ++i) {
    EXPECT_EQ(manager.lite_processor()->IdToPiece(i),
              original.pieces(i).piece());
    EXPECT_FLOAT_EQ(manager.lite_processor()->GetScore(i),
                    original.pieces(i).score());
    EXPECT_EQ(manager.lite_processor()->piece_type(i),
              original.pieces(i).type());
  }

  // Verify on-the-fly generation of model_proto() and caching.
  const ModelProto& reconstructed = manager.model_proto();
  EXPECT_EQ(&reconstructed, &manager.model_proto());
  EXPECT_EQ(reconstructed.SerializeAsString(), expected_serialized);
  EXPECT_EQ(manager.serialized_model_proto(), expected_serialized);
}

TEST(ModelProtoManagerTest, PrecompiledCharsmapZeroCopyFromFlatbufferTest) {
  ModelProto original = MakeBaseModelProto(TrainerSpec::UNIGRAM);
  std::string charsmap;
  ASSERT_TRUE(
      normalizer::Builder::CompileCharsMap({{{'a'}, {'b'}}}, &charsmap).ok());
  ASSERT_FALSE(charsmap.empty());
  original.mutable_normalizer_spec()->set_precompiled_charsmap(charsmap);
  const std::string expected_serialized = original.SerializeAsString();

  ModelProtoManager manager;
  ASSERT_TRUE(manager.Load(std::make_unique<ModelProto>(original)).ok());

  // Verify precompiled_charsmap is stripped from normalizer_spec_ in memory.
  EXPECT_FALSE(manager.normalizer_spec().has_precompiled_charsmap());
  EXPECT_TRUE(manager.normalizer_spec().precompiled_charsmap().empty());

  // Verify precompiled_charsmap() references the FlatBuffer directly.
  EXPECT_EQ(manager.precompiled_charsmap(), charsmap);

  // Verify on-the-fly model_proto() restores precompiled_charsmap into
  // normalizer_spec.
  const ModelProto& reconstructed = manager.model_proto();
  EXPECT_TRUE(reconstructed.normalizer_spec().has_precompiled_charsmap());
  EXPECT_EQ(reconstructed.normalizer_spec().precompiled_charsmap(), charsmap);
  EXPECT_EQ(reconstructed.SerializeAsString(), expected_serialized);

  // Verify SwapPrecompiledCharsmapEndian round-trip (LE -> BE -> LE).
  std::string swapped = charsmap;
  lite::SwapPrecompiledCharsmapEndian(&swapped, /*from_little_endian=*/true);
  EXPECT_NE(swapped, charsmap);
  lite::SwapPrecompiledCharsmapEndian(&swapped, /*from_little_endian=*/false);
  EXPECT_EQ(swapped, charsmap);
}

TEST(ModelProtoManagerTest, ConcurrentModelProtoGenerationTest) {
  const ModelProto original = MakeBaseModelProto(TrainerSpec::BPE);
  const std::string expected_serialized = original.SerializeAsString();

  ModelProtoManager manager;
  ASSERT_TRUE(manager.Load(std::make_unique<ModelProto>(original)).ok());

  constexpr int kNumThreads = 16;
  std::vector<const ModelProto*> observed_ptrs(kNumThreads, nullptr);
  std::vector<std::string> observed_serialized(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      observed_ptrs[t] = &manager.model_proto();
      observed_serialized[t] = manager.serialized_model_proto();
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  for (int t = 0; t < kNumThreads; ++t) {
    EXPECT_EQ(observed_ptrs[t], observed_ptrs[0]);
    EXPECT_EQ(observed_serialized[t], expected_serialized);
  }
}

TEST(ModelProtoManagerTest, CharAndWordModelPreservationTest) {
  // CHAR model: trainer_spec().model_type() and fractional piece scores are
  // preserved across both ModelProto Load() and LoadFromSerializedFlatbuffer().
  {
    ModelProto char_proto = MakeBaseModelProto(TrainerSpec::CHAR);
    char_proto.mutable_pieces(4)->set_score(-1.5f);  // fractional float score
    ModelProtoManager manager;
    ASSERT_TRUE(manager.Load(std::make_unique<ModelProto>(char_proto)).ok());

    const ModelProto& reconstructed = manager.model_proto();
    EXPECT_EQ(reconstructed.trainer_spec().model_type(), TrainerSpec::CHAR);
    EXPECT_FLOAT_EQ(reconstructed.pieces(4).score(), -1.5f);
    EXPECT_EQ(reconstructed.SerializeAsString(),
              char_proto.SerializeAsString());

    ModelProtoManager fb_manager;
    ASSERT_TRUE(
        fb_manager.LoadFromSerializedFlatbuffer(manager.serialized_flatbuffer())
            .ok());
    EXPECT_EQ(fb_manager.trainer_spec().model_type(), TrainerSpec::CHAR);
    EXPECT_EQ(fb_manager.model_proto().SerializeAsString(),
              char_proto.SerializeAsString());
  }

  // WORD model: trainer_spec().model_type() and fractional piece scores are
  // preserved across both ModelProto Load() and LoadFromSerializedFlatbuffer().
  {
    ModelProto word_proto = MakeBaseModelProto(TrainerSpec::WORD);
    word_proto.mutable_pieces(6)->set_score(-1.5f);  // fractional float score
    ModelProtoManager manager;
    ASSERT_TRUE(manager.Load(std::make_unique<ModelProto>(word_proto)).ok());

    const ModelProto& reconstructed = manager.model_proto();
    EXPECT_EQ(reconstructed.trainer_spec().model_type(), TrainerSpec::WORD);
    EXPECT_FLOAT_EQ(reconstructed.pieces(6).score(), -1.5f);
    EXPECT_EQ(reconstructed.SerializeAsString(),
              word_proto.SerializeAsString());

    ModelProtoManager fb_manager;
    ASSERT_TRUE(
        fb_manager.LoadFromSerializedFlatbuffer(manager.serialized_flatbuffer())
            .ok());
    EXPECT_EQ(fb_manager.trainer_spec().model_type(), TrainerSpec::WORD);
    EXPECT_EQ(fb_manager.model_proto().SerializeAsString(),
              word_proto.SerializeAsString());
  }
}

TEST(ModelProtoManagerTest, ValidationFailureTest) {
  // Null ModelProto.
  {
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(nullptr).ok());
  }

  // Empty pieces.
  {
    ModelProto proto;
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(std::make_unique<ModelProto>(proto)).ok());
  }

  // Multi-character NORMAL piece in CHAR model.
  {
    ModelProto proto = MakeBaseModelProto(TrainerSpec::CHAR);
    auto* multi_char = proto.add_pieces();
    multi_char->set_piece("ab");
    multi_char->set_score(-1.0f);
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(std::make_unique<ModelProto>(proto)).ok());
  }

  // Duplicate pieces.
  {
    ModelProto proto = MakeBaseModelProto();
    auto* dup = proto.add_pieces();
    dup->set_piece("a");
    dup->set_score(-4.0f);
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(std::make_unique<ModelProto>(proto)).ok());
  }

  // Missing UNKNOWN piece.
  {
    ModelProto proto = MakeBaseModelProto();
    proto.mutable_pieces(0)->set_type(ModelProto::SentencePiece::NORMAL);
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(std::make_unique<ModelProto>(proto)).ok());
  }

  // BYTE piece when byte_fallback is false.
  {
    ModelProto proto = MakeBaseModelProto();
    proto.mutable_trainer_spec()->set_byte_fallback(false);
    auto* byte_sp = proto.add_pieces();
    byte_sp->set_piece("<0x00>");
    byte_sp->set_type(ModelProto::SentencePiece::BYTE);
    ModelProtoManager manager;
    EXPECT_FALSE(manager.Load(std::make_unique<ModelProto>(proto)).ok());
  }
}

TEST(ModelProtoManagerTest, LoadFromFlatbufferAndViewTest) {
  ModelProto original = MakeBaseModelProto(TrainerSpec::UNIGRAM);
  ModelProtoManager base_manager;
  ASSERT_TRUE(base_manager.Load(std::make_unique<ModelProto>(original)).ok());

  const absl::string_view fb_bytes = base_manager.serialized_flatbuffer();
  ASSERT_FALSE(fb_bytes.empty());

  // 1. LoadFromSerializedFlatbuffer (owned copy)
  {
    ModelProtoManager copy_manager;
    ASSERT_TRUE(copy_manager.LoadFromSerializedFlatbuffer(fb_bytes).ok());
    EXPECT_EQ(copy_manager.serialized_flatbuffer(), fb_bytes);
    EXPECT_NE(copy_manager.serialized_flatbuffer().data(), fb_bytes.data());
    EXPECT_FLOAT_EQ(copy_manager.min_score(), -4.0f);
    EXPECT_EQ(copy_manager.model_proto().SerializeAsString(),
              original.SerializeAsString());
  }

  // 2. LoadFromFlatbufferView (zero-copy view)
  {
    ModelProtoManager view_manager;
    ASSERT_TRUE(view_manager.LoadFromFlatbufferView(fb_bytes).ok());
    EXPECT_EQ(view_manager.serialized_flatbuffer().data(), fb_bytes.data());
    EXPECT_EQ(view_manager.serialized_flatbuffer().size(), fb_bytes.size());
    EXPECT_FLOAT_EQ(view_manager.min_score(), -4.0f);
    EXPECT_EQ(view_manager.model_proto().SerializeAsString(),
              original.SerializeAsString());
  }

  // 3. Reject invalid/empty FlatBuffer
  {
    ModelProtoManager bad_manager;
    EXPECT_FALSE(bad_manager.LoadFromSerializedFlatbuffer("").ok());
    EXPECT_FALSE(bad_manager.LoadFromFlatbufferView("").ok());
    EXPECT_FALSE(
        bad_manager.LoadFromSerializedFlatbuffer("invalid_flatbuffer").ok());
  }

  // 4. Reject CHAR FlatBuffer containing a multi-character NORMAL piece
  {
    ModelProto bad_char = MakeBaseModelProto(TrainerSpec::CHAR);
    auto* multi_char = bad_char.add_pieces();
    multi_char->set_piece("ab");
    multi_char->set_score(-1.0f);
    auto fb_or = lite::ToFlatbuffer(
        bad_char,
        {.skip_char_bigrams = false, .allow_unsupported_model_type = true});
    ASSERT_TRUE(fb_or.ok()) << fb_or.status();
    ModelProtoManager bad_char_manager;
    EXPECT_FALSE(bad_char_manager.LoadFromSerializedFlatbuffer(*fb_or).ok());
    EXPECT_FALSE(bad_char_manager.LoadFromFlatbufferView(*fb_or).ok());
  }
}

}  // namespace
}  // namespace sentencepiece
