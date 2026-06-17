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
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "trainer_interface.h"

namespace sentencepiece::bpe {

// Trainer class for BPE model.
class Trainer : public TrainerInterface {
 public:
  Trainer(const TrainerSpec& trainer_spec,
          const NormalizerSpec& normalizer_spec,
          const NormalizerSpec& denormalizer_spec)
      : TrainerInterface::TrainerInterface(trainer_spec, normalizer_spec,
                                           denormalizer_spec) {}

  absl::Status Train() override;

#ifdef SPM_NLCODEC_BPE
  // Fast BPE training using nlcodec's max-heap + linked-list algorithm.
  // Based on nlcodec by Thamme Gowda (https://github.com/isi-nlp/nlcodec)
  // "Many-to-English Machine Translation Tools, Data, and Pretrained Models"
  // Gowda et al., ACL 2021. https://arxiv.org/abs/2104.00290v2
  absl::Status TrainFast();
#endif  // SPM_NLCODEC_BPE

 private:
  // Symbol represents a character or symbol bigram.
  struct Symbol {
    const Symbol* left = nullptr;    // left symbol in bigram
    const Symbol* right = nullptr;   // right symbol in bigram
    string_util::UnicodeText chars;  // all flattend chracter sequence
    uint64_t fp = 0;                 // fingerprint of this symbol.
    uint64_t freq = 0;               // frequency of this symbol.
    bool is_unk = false;             // true if this symbol is unknown.
    bool active = true;              // true if this symbol is active.
    bool pending = false;            // true if this symbol is pending push.
    bool needs_recomputation =
        true;  // true if this symbol needs recomputation.

    // Position list. Use set so that we can keep the order of occurrence.
    // See EncodePos/DecodePos.
    absl::btree_set<uint64_t> positions;

    [[nodiscard]] bool IsBigram() const {
      return left != nullptr && right != nullptr;
    }
    [[nodiscard]] std::string ToString() const;
    Symbol() = default;
  };

  struct Position {
    int sid;    // sentence id
    int left;   // left symbol index
    int right;  // right symbol index
  };

  // Encodes sid, left and right bigram index into uint64_t.
  // Encoded value keeps the order of sid, left and right.
  static uint64_t EncodePos(int sid, int l, int r) {
    CHECK_GE(l, 0);
    CHECK_GE(r, 0);
    CHECK_LE(l, std::numeric_limits<uint16_t>::max());
    CHECK_LE(r, std::numeric_limits<uint16_t>::max());
    const uint64_t n = (static_cast<uint64_t>(sid) << 32) |
                       (static_cast<uint64_t>(l) << 16) | r;
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
  Symbol* GetCharSymbol(char32_t c);

  // Gets symbol pair from left/right symbols. The return value is cached.
  Symbol* GetPairSymbol(const Symbol* left, const Symbol* right);

  // Computes the frequency of |symbol| and update symbol->freq field.
  void ComputeFreq(Symbol* symbol) const;

  // Returns the valid index before symbols_[sid][index].
  int GetNextIndex(int sid, int index) const;

  // Returns the valid index after symbols_[sid][index].
  int GetPrevIndex(int sid, int index) const;

  // Makes a new bigram from [symbols_[sid][left], symbols_[sid][right]] and
  // Adds it to symbols_cache_ and active_symbols_.
  void AddNewPair(int sid, int left, int right);

  // Resets the fequency of bigram [symbols_[sid][left] symbols_[sid][right]],
  // if this bigram is not |best|.
  void ResetFreq(int sid, int left, int right, const Symbol* best);

  absl::Status AcceptSymbol(Symbol* symbol);

  // All unique symbols. Key is a fingerprint of Symbol.
  absl::flat_hash_map<uint64_t, Symbol*> symbols_cache_;

  struct QueueEntry {
    uint64_t freq;
    Symbol* symbol;
  };

  struct QueueEntryComparator {
    bool operator()(const QueueEntry& e1, const QueueEntry& e2) const {
      if (e1.freq != e2.freq) {
        return e1.freq < e2.freq;
      }
      if (e1.symbol->chars.size() != e2.symbol->chars.size()) {
        return e1.symbol->chars.size() > e2.symbol->chars.size();
      }
      return e1.symbol->chars > e2.symbol->chars;
    }
  };

  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryComparator>
      pq_;
  std::vector<Symbol*> pending_queue_;

  // Stores symbols allocated in heap so that they are automatically deleted.
  std::vector<std::unique_ptr<Symbol>> allocated_;

  // Sentences. symbols_[sid][index] stores a symbol in sentence_[sid][index].
  std::vector<std::vector<Symbol*>> symbols_;
};
}  // namespace sentencepiece::bpe

#endif  // BPE_MODEL_TRAINER_H_
