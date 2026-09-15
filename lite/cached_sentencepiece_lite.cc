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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "sentencepiece_lite.h"
#include "third_party/rapidhash/rapidhash.h"

namespace sentencepiece::lite {
namespace {

constexpr uint64_t kGoldenRatio64 = 0x9e3779b97f4a7c15ULL;
constexpr size_t kExpectedBytesPerToken = 2;

inline uint64_t PointerSeed(const void* ptr) {
  const uint64_t seed = reinterpret_cast<uintptr_t>(ptr) * kGoldenRatio64;
  return (seed == 0) ? kGoldenRatio64 : seed;
}

inline uint64_t FastHash64(std::string_view str, uint64_t seed) {
  return rapidhash_withSeed_unrolled(str.data(), str.size(), seed);
}

}  // namespace

TokenCache::TokenCache() : TokenCache(Capacity::kMedium, PointerSeed(this)) {}

TokenCache::TokenCache(Capacity capacity, uint64_t seed)
    : num_pairs_(static_cast<size_t>(capacity)),
      mask_(static_cast<size_t>(capacity) - 1),
      seed_((seed == 0) ? PointerSeed(this) : seed) {
  Clear();
}

TokenCache::TokenCache(uint64_t seed)
    : TokenCache(Capacity::kMedium, (seed == 0) ? PointerSeed(this) : seed) {}

StatusCode CachedSentencePieceLite::Encode(
    const SentencePieceLiteProcessor& processor, TokenCache& cache,
    std::string_view text, std::vector<int>* ids) {
  if (ids == nullptr) return StatusCode::kInvalidArgument;
  ids->clear();
  if (text.empty()) return StatusCode::kOk;

  std::string normalized_storage;
  StatusCode status = processor.Normalize(text, &normalized_storage);
  if (status != StatusCode::kOk) return status;

  return EncodeNormalized(processor, cache, normalized_storage, ids);
}

StatusCode CachedSentencePieceLite::EncodeNormalized(
    const SentencePieceLiteProcessor& processor, TokenCache& cache,
    std::string_view normalized_text, std::vector<int>* ids) {
  if (ids == nullptr) return StatusCode::kInvalidArgument;
  ids->clear();
  if (normalized_text.empty()) return StatusCode::kOk;

  const uint64_t effective_seed = cache.seed() ^ PointerSeed(&processor);

  ids->reserve(normalized_text.size() / kExpectedBytesPerToken + 1);
  std::vector<int> chunk_ids;
  return processor.PretokenizeAtSafeBoundaries(
      normalized_text, [&](std::string_view chunk) {
        if (chunk.empty()) return;
        const uint64_t hash = FastHash64(chunk, effective_seed);
        const uint16_t tag = TokenCache::MakeChunkTag(chunk);
        size_t count = 0;
        const int32_t* cached_ids = cache.Lookup(hash, tag, &count);
        if (cached_ids != nullptr) {
          ids->insert(ids->end(), cached_ids, cached_ids + count);
          return;
        }
        chunk_ids.clear();
        processor.EncodeNormalizedChunk(chunk, &chunk_ids);
        ids->insert(ids->end(), chunk_ids.begin(), chunk_ids.end());
        if (!chunk_ids.empty() &&
            chunk_ids.size() <= TokenCache::kMaxCachedTokenCount) {
          cache.Insert(hash, tag, chunk_ids.data(), chunk_ids.size());
        }
      });
}

}  // namespace sentencepiece::lite
