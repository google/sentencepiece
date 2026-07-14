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

#include <thread>
#include <vector>

#include "builtin_upb/upb_wrapper.h"

#include "testharness.h"

namespace sentencepiece {
namespace {

// 1. Basic Serialization and Multi-write / Empty string support
TEST(UpbWrapperTest, BasicSerializationAndAccessors) {
  ModelProto model;

  EXPECT_EQ(model.pieces_size(), 0);
  EXPECT_FALSE(model.has_normalizer_spec());

  // Test primitive setters
  auto* trainer_spec = model.mutable_trainer_spec();
  trainer_spec->set_vocab_size(1000);
  trainer_spec->set_character_coverage(0.999f);
  trainer_spec->set_model_type(TrainerSpec::BPE);

  // Test nested string setters (NFKC / Normalizer)
  auto* normalizer_spec = model.mutable_normalizer_spec();
  normalizer_spec->set_name("nfkc");

  // Verify multiple overwrites on same fields
  trainer_spec->set_vocab_size(2000);
  normalizer_spec->set_name("identity");

  // Test empty string assignment (valid edge-case)
  normalizer_spec->set_precompiled_charsmap("");

  // Reflection checks
  EXPECT_EQ(model.trainer_spec().vocab_size(), 2000);
  EXPECT_NEAR(model.trainer_spec().character_coverage(), 0.999f, 1e-5);
  EXPECT_EQ(model.trainer_spec().model_type(), TrainerSpec::BPE);
  EXPECT_EQ(model.normalizer_spec().name(), "identity");
  EXPECT_EQ(model.normalizer_spec().precompiled_charsmap(), "");

  // Clear fields and verify defaults
  trainer_spec->clear_vocab_size();
  EXPECT_EQ(model.trainer_spec().vocab_size(), 8000);  // Default protobuf value

  // Serialization cycle
  std::string serialized = model.SerializeAsString();
  EXPECT_FALSE(serialized.empty());

  ModelProto model2;
  EXPECT_TRUE(model2.ParseFromString(serialized));
  EXPECT_EQ(model2.trainer_spec().vocab_size(), 8000);
  EXPECT_EQ(model2.normalizer_spec().name(), "identity");
  EXPECT_EQ(model2.normalizer_spec().precompiled_charsmap(), "");
}

// 2. Comprehensive Out-of-Bounds safety test across all wrapper classes
TEST(UpbWrapperTest, RepeatedAccessAndBoundsSafety) {
  ModelProto model;

  // Add pieces to ModelProto
  auto* p1 = model.add_pieces();
  p1->set_piece("hello");
  p1->set_score(1.5f);
  p1->set_type(ModelProto_SentencePiece::NORMAL);

  auto* p2 = model.add_pieces();
  p2->set_piece("world");
  p2->set_score(2.5f);

  EXPECT_EQ(model.pieces_size(), 2);

  // Normal access checks
  EXPECT_EQ(model.piece_at(0), "hello");
  EXPECT_NEAR(model.score_at(0), 1.5f, 1e-5);
  EXPECT_EQ(model.piece_at(1), "world");
  EXPECT_NEAR(model.score_at(1), 2.5f, 1e-5);

  // ModelProto Bounds Safety Checks
  EXPECT_EQ(model.piece_at(-1), "");
  EXPECT_EQ(model.piece_at(2), "");
  EXPECT_EQ(model.piece_at(100), "");
  EXPECT_NEAR(model.score_at(-1), 0.0f, 1e-5);
  EXPECT_NEAR(model.score_at(2), 0.0f, 1e-5);
  EXPECT_EQ(model.type_at(-1), 0);
  EXPECT_EQ(model.type_at(2), 0);

  // SentencePieceTextWrapper Bounds Safety Checks
  SentencePieceText spt;
  EXPECT_EQ(spt.pieces_size(), 0);
  // Access empty SentencePieceText should return default safe items or values
  EXPECT_EQ(spt.text(), "");
  EXPECT_NEAR(spt.score(), 0.0f, 1e-5);

  auto* sp_item = spt.add_pieces();
  sp_item->set_piece("token");
  sp_item->set_surface("surface");
  sp_item->set_id(42);

  EXPECT_EQ(spt.pieces_size(), 1);
  EXPECT_EQ(spt.piece_at(0), "token");
  EXPECT_EQ(spt.surface_at(0), "surface");
  EXPECT_EQ(spt.id_at(0), 42);

  // Test OOB for SentencePieceText
  EXPECT_EQ(spt.piece_at(-1), "");
  EXPECT_EQ(spt.piece_at(1), "");
  EXPECT_EQ(spt.surface_at(-1), "");
  EXPECT_EQ(spt.surface_at(1), "");
  EXPECT_EQ(spt.id_at(-1), 0);
  EXPECT_EQ(spt.id_at(1), 0);
  EXPECT_EQ(spt.begin_at(-1), 0);
  EXPECT_EQ(spt.begin_at(1), 0);
  EXPECT_EQ(spt.end_at(-1), 0);
  EXPECT_EQ(spt.end_at(1), 0);

  // NBestSentencePieceTextWrapper Bounds Safety Checks
  NBestSentencePieceText nbest;
  EXPECT_EQ(nbest.nbests_size(), 0);
  // Accessing OOB NBest items should return a static default empty wrapper
  // reference
  const auto& default_sub1 = nbest.nbests(-1);
  const auto& default_sub2 = nbest.nbests(0);
  EXPECT_EQ(default_sub1.pieces_size(), 0);
  EXPECT_EQ(default_sub2.pieces_size(), 0);

  auto* sub = nbest.add_nbests();
  auto* sub_sp = sub->add_pieces();
  sub_sp->set_piece("nbest_token");

  EXPECT_EQ(nbest.nbests_size(), 1);
  EXPECT_EQ(nbest.nbests(0).pieces_size(), 1);
  EXPECT_EQ(nbest.nbests(0).piece_at(0), "nbest_token");

  // Test OOB sub elements
  EXPECT_EQ(nbest.nbests(0).piece_at(-1), "");
  EXPECT_EQ(nbest.nbests(0).piece_at(1), "");
}

// 3. Child to Parent Synchronization & Cache Rebuilding Verification
TEST(UpbWrapperTest, ChildParentSyncAndRebuilding) {
  ModelProto model;

  EXPECT_FALSE(model.has_normalizer_spec());

  // 1. Test string field sync on normalizer_spec
  auto* normalizer = model.mutable_normalizer_spec();
  normalizer->set_name("custom_norm");
  EXPECT_TRUE(model.has_normalizer_spec());
  // Verify Eagerly rebuilt cache holds the correct updated value
  EXPECT_EQ(model.normalizer_spec().name(), "custom_norm");

  // 2. Test primitive field sync (e.g. add_dummy_prefix = false)
  // This was previously failing due to primitive accessor macros missing
  // on_change_ calls
  normalizer->set_add_dummy_prefix(false);
  EXPECT_FALSE(model.normalizer_spec().add_dummy_prefix());

  // 3. Test enum field sync on trainer_spec
  auto* trainer = model.mutable_trainer_spec();
  trainer->set_model_type(TrainerSpec::WORD);
  EXPECT_EQ(model.trainer_spec().model_type(), TrainerSpec::WORD);

  // 4. Test repeated string field sync on trainer_spec
  trainer->add_control_symbols("<special>");
  EXPECT_EQ(model.trainer_spec().control_symbols_size(), 1);
  EXPECT_EQ(model.trainer_spec().control_symbols(0), "<special>");

  // 5. Test Copy and Assignment cache consistency
  ModelProto model_copy(model);  // Copy constructor
  EXPECT_FALSE(model_copy.normalizer_spec().add_dummy_prefix());
  EXPECT_EQ(model_copy.trainer_spec().model_type(), TrainerSpec::WORD);

  ModelProto model_assigned;
  model_assigned = model;  // Copy assignment operator
  EXPECT_FALSE(model_assigned.normalizer_spec().add_dummy_prefix());
  EXPECT_EQ(model_assigned.trainer_spec().model_type(), TrainerSpec::WORD);
}

// 4. High-load concurrent read thread safety verification
TEST(UpbWrapperTest, ThreadSafetyHighLoad) {
  ModelProto model;

  // Add 5000 items to put pressure on memory layout
  for (int i = 0; i < 5000; ++i) {
    auto* p = model.add_pieces();
    p->set_piece("tok_" + std::to_string(i));
    p->set_score(static_cast<float>(i * 0.1));
    p->set_type(ModelProto_SentencePiece::NORMAL);
  }

  auto* norm = model.mutable_normalizer_spec();
  norm->set_name("ts_norm");
  norm->set_add_dummy_prefix(false);

  std::string serialized = model.SerializeAsString();

  auto const_model = std::make_shared<ModelProto>();
  EXPECT_TRUE(const_model->ParseFromString(serialized));

  // Spawn 20 threads performing high-frequency read access on const wrappers
  std::vector<std::thread> threads;
  for (int t = 0; t < 20; ++t) {
    threads.emplace_back([const_model, t]() {
      for (int i = 0; i < 500; ++i) {
        // Access global specs
        EXPECT_EQ(const_model->pieces_size(), 5000);
        EXPECT_EQ(const_model->normalizer_spec().name(), "ts_norm");
        EXPECT_FALSE(const_model->normalizer_spec().add_dummy_prefix());

        // Read specific index pieces dynamically
        int idx = (i * (t + 1)) % 5000;
        EXPECT_EQ(const_model->piece_at(idx), "tok_" + std::to_string(idx));
        EXPECT_NEAR(const_model->score_at(idx), static_cast<float>(idx * 0.1),
                    1e-5);
        EXPECT_EQ(const_model->type_at(idx), ModelProto_SentencePiece::NORMAL);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

// 5. Test Pre-initialization of spec caches and prevent UAF on modifications
TEST(UpbWrapperTest, MutableSpecPointerSyncAndLazyInit) {
  ModelProto model;

  // Verify pre-initialization on programmatically prepared models.
  // Prior to the fix, normalizer_spec() returned default_instance (add_dummy_prefix=true)
  // because the cache remained null when fields were mutated before any serialization.
  auto* norm_mutable = model.mutable_normalizer_spec();
  norm_mutable->set_add_dummy_prefix(false);
  norm_mutable->set_remove_extra_whitespaces(false);

  EXPECT_FALSE(model.normalizer_spec().add_dummy_prefix());
  EXPECT_FALSE(model.normalizer_spec().remove_extra_whitespaces());

  // Capture the pointer of normalizer_spec() cache wrapper.
  const NormalizerSpec* cached_spec_ptr = &model.normalizer_spec();

  // Perform modifications that trigger on_change() lambda.
  norm_mutable->set_escape_whitespaces(false);

  // Verify the address of normalizer_spec_cache_ remains identical
  // (We must not recreate the cached wrapper instance with std::make_unique,
  // otherwise existing raw pointers cached in sentencepiece::Normalizer become dangling).
  EXPECT_EQ(cached_spec_ptr, &model.normalizer_spec());

  // Verify values are correctly updated and safe to read via old pointer.
  EXPECT_FALSE(cached_spec_ptr->add_dummy_prefix());
  EXPECT_FALSE(cached_spec_ptr->remove_extra_whitespaces());
  EXPECT_FALSE(cached_spec_ptr->escape_whitespaces());
}

// 6. Verify that concurrent reads on specs are thread-safe and do not trigger data races.
TEST(UpbWrapperTest, ThreadSafetyLazyInitRace) {
  ModelProto model;

  // Manually construct NormalizerSpec. Cache is pre-allocated and won't be null.
  auto* norm_mutable = model.mutable_normalizer_spec();
  norm_mutable->set_add_dummy_prefix(false);

  // Spawn threads reading normalizer_spec() concurrently.
  // The cache is pre-allocated and getters are pure read-only operations, preventing data races.
  std::vector<std::thread> threads;
  for (int t = 0; t < 20; ++t) {
    threads.emplace_back([&model]() {
      for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(model.normalizer_spec().add_dummy_prefix());
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

// 7. Verify that OOB write on ModelProto pieces is safely ignored and doesn't crash.
TEST(UpbWrapperTest, OOBWritePoC) {
  ModelProto model;
  // Initialize with 0 pieces.
  ASSERT_EQ(model.pieces_size(), 0);

  // Get a wrapper pointing to index 99999 (which is OOB)
  auto piece = model.pieces(99999);

  std::cout << "DEBUG: Executing OOB write. Expecting NO crash (safe ignore)..." << std::endl;
  // With boundary checks, these should be safe no-ops and not crash.
  piece.set_score(1.0f);
  piece.set_type(ModelProto_SentencePiece::NORMAL);
  piece.set_piece("test");

  // We should reach here successfully.
  ASSERT_TRUE(true);
}

// 8. Verify that OOB write on SentencePieceText is safely ignored and doesn't crash.
TEST(UpbWrapperTest, SentencePieceTextOOBWritePoC) {
  SentencePieceText spt;
  // Initialize with 0 pieces.
  ASSERT_EQ(spt.pieces_size(), 0);

  std::cout << "DEBUG: Executing SentencePieceText OOB write. Expecting NO crash..." << std::endl;
  // With boundary checks, these should be safe no-ops or return nullptr.
  spt.set_id_at(99999, 123);
  spt.set_piece_at(99999, "test");
  spt.set_surface_at(99999, "test");
  spt.set_begin_at(99999, 0);
  spt.set_end_at(99999, 1);
  spt.SwapElementsData(99999, 0);

  auto* piece = spt.mutable_pieces(99999);
  ASSERT_EQ(piece, nullptr);

  // We should reach here successfully.
  ASSERT_TRUE(true);
}

// 9. Verify that OOB access on NBestSentencePieceText is safely ignored and doesn't crash.
TEST(UpbWrapperTest, NBestSentencePieceTextOOBWritePoC) {
  NBestSentencePieceText nbest;
  // Initialize with 0 nbests.
  ASSERT_EQ(nbest.nbests_size(), 0);

  std::cout << "DEBUG: Executing NBestSentencePieceText OOB write. Expecting NO crash..." << std::endl;
  // With boundary checks, this should return nullptr instead of crashing.
  auto* sub = nbest.mutable_nbests_at(99999);
  ASSERT_EQ(sub, nullptr);
}

// 10. Verify that OOB access on ModelProto mutable_pieces is safely ignored.
TEST(UpbWrapperTest, ModelProtoMutablePiecesOOBPoC) {
  ModelProto model;
  ASSERT_EQ(model.pieces_size(), 0);

  std::cout << "DEBUG: Executing ModelProto mutable_pieces OOB. Expecting nullptr..." << std::endl;
  // With boundary checks, this should return nullptr instead of crashing.
  auto* p = model.mutable_pieces(99999);
  ASSERT_EQ(p, nullptr);
}
}  // namespace
}  // namespace sentencepiece
