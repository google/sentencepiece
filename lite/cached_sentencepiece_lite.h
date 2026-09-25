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

#ifndef SENTENCEPIECE_LITE_CACHED_SENTENCEPIECE_LITE_H_
#define SENTENCEPIECE_LITE_CACHED_SENTENCEPIECE_LITE_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include "sentencepiece_lite.h"

namespace sentencepiece::lite {

// 2-Way Set-Associative Token Cache for SentencePieceLite.
//
// Features:
// - Zero Heap Allocations during Lookup and Insert.
// - 64B L1 Cache-Line alignment with 2-way short-circuit scalar search.
// - 1-Bit PLRU (Pseudo-LRU) replacement policy.
// - Single-block branchless (CMOV) Insert.
//
// Thread safety: Not thread-safe for concurrent writes. Use per-thread
// instances (e.g. thread_local TokenCache) or caller-managed thread-local
// scope.
class TokenCache {
 public:
  // Maximum number of subword tokens cached per pre-tokenized chunk.
  static constexpr size_t kMaxCachedTokenCount = 5;

  // Computes a 16-bit zero-overhead fingerprint (length + boundary signature)
  // from pre-tokenized chunk.
  static inline uint16_t MakeChunkTag(std::string_view chunk) {
    if (chunk.empty()) return 0;
    const uint8_t len = static_cast<uint8_t>(chunk.size());
    // Bijective mapping on single-char chunks: (c * 2) + c = 3 * c (mod 256).
    // Since gcd(3, 256) == 1, all 256 byte values map 1-to-1 uniquely.
    const uint8_t char_sig =
        static_cast<uint8_t>((static_cast<uint8_t>(chunk.front()) * 2) +
                             static_cast<uint8_t>(chunk.back()));
    return static_cast<uint16_t>((static_cast<uint16_t>(len) << 8) | char_sig);
  }

 private:
  // 32-Byte Aligned Cache Entry.
  struct alignas(32) TokenCacheEntry {
    uint64_t hash = 0;        // 64-bit Hash value
    uint8_t token_count = 0;  // token id count.
    uint8_t mru_bit = 0;      // 1 is MRU (Most Recently Used), 0 is LRU.
    uint16_t tag = 0;         // 16-bit Length & Boundary Fingerprint
    int32_t token_ids[kMaxCachedTokenCount] = {0, 0, 0, 0,
                                               0};  // Cached token ids.
  };
  static_assert(sizeof(TokenCacheEntry) == 32,
                "TokenCacheEntry must be exactly 32 bytes!");
  // 64-Byte Aligned Cache Pair (Fits exactly in 1 CPU L1 Cache Line).
  struct alignas(64) TokenCachePair {
    TokenCacheEntry entries[2];
  };
  static_assert(sizeof(TokenCachePair) == 64,
                "TokenCachePair must be exactly 64 bytes!");

 public:
  // Pre-defined power-of-2 cache capacities.
  //
  // Sizing Guide:
  // - kTiny (64 KB, 2,048 slots): Minimal memory footprint. Recommended for
  //   memory-constrained environments (e.g., mobile or embedded devices).
  // - kSmall (256 KB, 8,192 slots): Low memory footprint. Recommended for
  //   lightweight workers or tight per-thread memory budgets.
  // - kMedium (1 MB, 32,768 slots, Default): Balanced capacity and memory
  //   usage for general server-side serving pipelines.
  // - kLarge (8 MB, 262,144 slots): High capacity for large batch workloads or
  //   diverse multi-domain text.
  enum class Capacity : size_t {
    kTiny = 1024,     // 1,024 pairs = 2,048 slots (64 KB)
    kSmall = 4096,    // 4,096 pairs = 8,192 slots (256 KB)
    kMedium = 16384,  // 16,384 pairs = 32,768 slots (1 MB, Default)
    kLarge = 131072,  // 131,072 pairs = 262,144 slots (8 MB)
  };

  // Initializes TokenCache with default capacity (kMedium, 1 MB) and a dynamic
  // random seed.
  TokenCache();

  // Initializes TokenCache with an explicit capacity and optional seed (0 =
  // dynamic seed).
  explicit TokenCache(Capacity capacity, uint64_t seed = 0);

  // Initializes TokenCache with default capacity (kMedium, 1 MB) and an
  // explicit seed.
  explicit TokenCache(uint64_t seed);

  // Returns the 64-bit random seed associated with this cache instance.
  uint64_t seed() const { return seed_; }

  // Returns the total number of cache slots (2 * num_pairs).
  size_t capacity() const { return num_pairs_ * 2; }
  size_t num_pairs() const { return num_pairs_; }

  // Clear all entries cleanly.
  void Clear() {
    if (!pairs_) {
      pairs_ = std::make_unique<TokenCachePair[]>(num_pairs_);
    }
    std::memset(static_cast<void*>(pairs_.get()), 0,
                sizeof(TokenCachePair) * num_pairs_);
  }

  // Direct 2-Way Short-Circuit Scalar Cache Lookup with 1-Bit PLRU Update and
  // 16-bit Tag Validation.
  const int32_t* Lookup(uint64_t hash, uint16_t tag, size_t* out_count) {
    const size_t pair_idx = (hash & mask_);
    TokenCachePair& pair = pairs_[pair_idx];

    // Short-circuit: hash comparison failure skips tag and token_count checks.
    if (pair.entries[0].hash == hash && pair.entries[0].tag == tag &&
        pair.entries[0].token_count != 0) {
      *out_count = pair.entries[0].token_count;
      pair.entries[0].mru_bit = 1;
      pair.entries[1].mru_bit = 0;
      return pair.entries[0].token_ids;
    }
    if (pair.entries[1].hash == hash && pair.entries[1].tag == tag &&
        pair.entries[1].token_count != 0) {
      *out_count = pair.entries[1].token_count;
      pair.entries[1].mru_bit = 1;
      pair.entries[0].mru_bit = 0;
      return pair.entries[1].token_ids;
    }
    return nullptr;
  }

  // Insert or Update Cache Entry using Single-Block Branchless 1-Bit PLRU.
  void Insert(uint64_t hash, uint16_t tag, const int32_t* ids, size_t count) {
    if (count == 0 || count > kMaxCachedTokenCount) return;

    const size_t pair_idx = (hash & mask_);
    TokenCachePair& pair = pairs_[pair_idx];

    // Check if slot 0 or slot 1 is occupied (token_count != 0)
    const uint8_t occ0 = (pair.entries[0].token_count != 0 ? 1 : 0);
    const uint8_t occ1 = (pair.entries[1].token_count != 0 ? 1 : 0);

    // Check if occupied slot matches existing hash and tag
    const uint8_t match0 =
        (occ0 && pair.entries[0].hash == hash && pair.entries[0].tag == tag
             ? 1
             : 0);
    const uint8_t match1 =
        (occ1 && pair.entries[1].hash == hash && pair.entries[1].tag == tag
             ? 1
             : 0);

    // If both slots occupied, evict Least Recently Used (mru_bit == 0) slot.
    // If slot 0 is MRU (mru_bit == 1), evict slot 1. Else evict slot 0.
    const uint8_t plru_evict = (pair.entries[0].mru_bit == 1 ? 1 : 0);
    const uint8_t fallback_slot = (occ0 & occ1) ? plru_evict : occ0;

    // Single unified target_slot determination (Branchless CMOV):
    // Match 0 -> Slot 0, Match 1 -> Slot 1, Else -> Empty or PLRU Slot
    const uint8_t target_slot = match0 ? 0 : (match1 ? 1 : fallback_slot);

    TokenCacheEntry& target = pair.entries[target_slot];
    TokenCacheEntry& sibling = pair.entries[target_slot ^ 1];
    target.hash = hash;
    target.tag = tag;
    target.token_count = static_cast<uint8_t>(count);
    target.mru_bit = 1;
    sibling.mru_bit = 0;
    for (size_t i = 0; i < count; ++i) {
      target.token_ids[i] = ids[i];
    }
  }

 private:
  size_t num_pairs_ = static_cast<size_t>(Capacity::kMedium);
  size_t mask_ = static_cast<size_t>(Capacity::kMedium) - 1;
  uint64_t seed_ = 0;
  std::unique_ptr<TokenCachePair[]> pairs_;
};

// High-level Utility Class for Cached SentencePieceLite Tokenization.
//
// Cache Key Partitioning and Multi-Model Thread-Local Usage:
//   CachedSentencePieceLite mixes the memory address of the `processor`
//   instance into the effective hashing seed (`cache.seed() ^
//   (uintptr_t)&processor * kGoldenRatio`). This allows a single shared
//   `thread_local TokenCache` to safely tokenize requests across different
//   `SentencePieceLiteProcessor` instances without cross-model cache collisions
//   or output corruption (pseudo per-instance cache partitioning).
//
//   CAVEAT ON CACHE UTILIZATION:
//   While functionally safe and collision-free, if multiple distinct model
//   instances frequently interleave tokenization on the same shared
//   thread-local cache, cache lines will experience increased PLRU eviction,
//   reducing overall cache hit rate and utilization. For maximum throughput in
//   multi-model serving pipelines, dedicated per-processor cache instances are
//   recommended.
//
// Example usage:
//   // Initialize processor once (thread-safe, read-only across threads)
//   SentencePieceLiteProcessor processor(model_buffer);
//
//   // Worker function executed across worker threads
//   void TokenizeWorker(const SentencePieceLiteProcessor& processor,
//                       std::string_view text) {
//     // Thread-local cache instance
//     static thread_local TokenCache cache;
//
//     std::vector<int> ids;
//     StatusCode status =
//         CachedSentencePieceLite::Encode(processor, cache, text, &ids);
//     if (status == StatusCode::kOk) {
//       // Process token IDs...
//     }
//   }
class CachedSentencePieceLite {
 public:
  // Encodes raw `text` into token `ids` using `processor` and accelerating
  // repeated pre-tokenized chunks via `cache`. Automatically performs text
  // normalization.
  //
  // Returns StatusCode::kOk on success, or an error status code on failure.
  //
  // Thread safety: Thread-safe if each thread uses its own `TokenCache`
  // instance (e.g. thread_local TokenCache).
  static StatusCode Encode(const SentencePieceLiteProcessor& processor,
                           TokenCache& cache, std::string_view text,
                           std::vector<int>* ids);

  // Encodes pre-normalized `normalized_text` into token `ids` using `processor`
  // and accelerating repeated pre-tokenized chunks via `cache`.
  // Skips the text normalization step for maximum performance when the input is
  // already normalized.
  //
  // Returns StatusCode::kOk on success, or an error status code on failure.
  //
  // Thread safety: Thread-safe if each thread uses its own `TokenCache`
  // instance (e.g. thread_local TokenCache).
  static StatusCode EncodeNormalized(
      const SentencePieceLiteProcessor& processor, TokenCache& cache,
      std::string_view normalized_text, std::vector<int>* ids);
};

}  // namespace sentencepiece::lite

#endif  // SENTENCEPIECE_LITE_CACHED_SENTENCEPIECE_LITE_H_
