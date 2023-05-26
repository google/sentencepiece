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

#ifndef BPE_MODEL_TRAINER_H_
#define BPE_MODEL_TRAINER_H_

#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "absl/container/flat_hash_map.h"
#include "trainer_interface.h"

namespace sentencepiece {
namespace bpe {

// Trainer class for BPE model.
class Trainer : public TrainerInterface {
 public:
  Trainer(const TrainerSpec &trainer_spec,
          const NormalizerSpec &normalizer_spec,
          const NormalizerSpec &denormalizer_spec)
      : TrainerInterface::TrainerInterface(trainer_spec, normalizer_spec,
                                           denormalizer_spec) {}

  util::Status Train() override;

 private:
  // Symbol represents a character or symbol bigram.
  struct Symbol {
    uint32_t left;                   // left symbol in bigram
    uint32_t right;                  // right symbol in bigram
    uint64_t fp;                     // fingerprof this symbol.
    uint64_t freq;                   // frequency of this symbol.
    union {                          // all flattened character sequence
      char32 chars_embed[2];
      uint64_t chars_embed_pair;
      char32 *chars_ext;
    };

    // Position list. Sorted to preserve the order of occurrence.
    // See EncodePos/DecodePos.
    std::vector<uint64_t> positions;

    uint8_t chars_size;

    bool IsBigram() const noexcept { return left != ~0u && right != ~0u; }
    bool IsUnk() const noexcept { return fp == kUNKChar; }
    size_t CharsSize() const { return chars_size; }
    void AppendChar(char32 c);
    void AssignChars(const string_util::UnicodeText &text);
    void AppendCharsToText(string_util::UnicodeText *text) const;
    std::string ToString() const;
    Symbol(uint32_t left_ = ~0u,
           uint32_t right_ = ~0u,
           uint64_t fp_ = 0,
           uint64_t freq_ = 0)
      : left(left_),
        right(right_),
        fp(fp_),
        freq(freq_),
        chars_size(0) {}
    ~Symbol();
  };

  // char (*__sizeof)[sizeof(Symbol)] = 1;

  struct Position {
    uint32_t sid;    // sentence id
    uint32_t left;   // left symbol index
    uint32_t right;  // right symbol index
  };

  // Encodes sid, left and right bigram index into uint64_t.
  // Encoded value keeps the order of sid, left and right.
  static uint64_t EncodePos(uint32_t sid, uint32_t l, uint32_t r) {
    CHECK(sid != ~0u);
    CHECK(l != ~0u);
    CHECK(r != ~0u);
    CHECK_LE(l, std::numeric_limits<uint16_t>::max());
    CHECK_LE(r, std::numeric_limits<uint16_t>::max());
    const uint64_t n = (static_cast<uint64_t>(sid) << 32) |
                       (static_cast<uint64_t>(l) << 16) |
                       r;
    return n;
  }

  // Decodes sid, left and right bigram index from uint64_t.
  static Position DecodePos(uint64_t n) {
    Position p;
    p.sid = n >> 32;
    p.left = (n >> 16) & 0xffff;
    p.right = n & 0xffff;
    return p;
  }

  // Gets unary (character) symbol from the char code |c|.
  // The return value is cached.
  uint32_t GetCharSymbol(char32 c, bool require_cache);

  // Gets symbol pair from left/right symbols. The return value is cached.
  uint32_t GetPairSymbol(uint32_t left, uint32_t right);

  // Computes the frequency of |symbol| and update symbol->freq field.
  void ComputeFreq(Symbol *symbol) const;

  // Returns the valid index before symbols_[sid][index].
  uint32_t GetNextIndex(uint32_t sid, uint32_t index) const;

  // Returns the valid index after symbols_[sid][index].
  uint32_t GetPrevIndex(uint32_t sid, uint32_t index) const;

  // Makes a new bigram from [symbols_[sid][left], symbols_[sid][right]] and
  // Adds it to symbols_cache_ and active_symbols_.
  void AddNewPair(uint32_t sid, uint32_t left, uint32_t right);

  void SortSymbolPositions();

  // Resets the frequency of bigram [symbols_[sid][left] symbols_[sid][right]],
  // if this bigram is not |best|.
  void ResetFreq(uint32_t sid, uint32_t left, uint32_t right, uint32_t best);

  // Updates |active_symbols_| by copying the top 5% frequent symbols in
  // symbols_cache_.
  void UpdateActiveSymbols(ThreadPool *pool);

  util::Status LoadSentencesFromCache(filesystem::ReadableFile *cache_file);
  util::Status StoreSentencesToCache();

  // All unique symbols. Key is a fingerprint of Symbol.
  absl::flat_hash_map<uint64_t, uint32_t> symbols_cache_;

  // Set of symbols from which we find the best symbol in each iteration.
  absl::flat_hash_set<uint32_t> active_symbols_;

  // Stores symbols allocated in heap so that we can delete them at once.
  std::deque<Symbol> allocated_;

  // Sentences. symbols_[sid][index] stores a symbol in sentence_[sid][index].
  std::deque<std::vector<uint32_t>> symbols_;

  // Token frequencies.
  std::vector<int64> freqs_;
};
}  // namespace bpe
}  // namespace sentencepiece
#endif  // BPE_MODEL_TRAINER_H_
