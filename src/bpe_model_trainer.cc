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

#include "bpe_model_trainer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "ret_check.h"
#include "util.h"

namespace sentencepiece::bpe {
namespace {

// Computes UTF-8 byte length for a Unicode codepoint without string
// allocations.
constexpr int UTF8ByteLength(char32_t c) {
  if (c <= 0x7F) return 1;
  if (c <= 0x7FF) return 2;
  if (c <= 0xFFFF) return 3;
  return 4;
}

// 1D Binary Indexed Tree (Fenwick Tree) supporting prefix sums and
// binary-lifting search in O(log N) time.
class FenwickTree {
 public:
  explicit FenwickTree(int n) : n_(n), tree_(n + 1, 0) {}

  // Adds `val` to the element at 0-based index `idx` in O(log N).
  void Add(int idx, int64_t val) {
    for (int i = idx + 1; i <= n_; i += i & -i) {
      tree_[i] += val;
    }
  }

  // Computes the prefix sum for elements in range [0, idx] in O(log N).
  // Returns 0 if idx < 0. Clamps idx to (n_ - 1) if idx >= n_.
  int64_t Query(int idx) const {
    if (idx < 0) return 0;
    idx = std::min(idx, n_ - 1);
    int64_t sum = 0;
    for (int i = idx + 1; i > 0; i -= i & -i) {
      sum += tree_[i];
    }
    return sum;
  }

  // Finds the smallest 0-based index `idx` such that Query(idx) >= `target`
  // using binary lifting in O(log N). Returns -1 if target <= 0.
  int FindByPrefixCount(int target) const {
    if (target <= 0) return -1;
    int idx = 0;
    int step = 1;
    while (step <= n_) step <<= 1;
    while (step > 0) {
      if (idx + step <= n_ && tree_[idx + step] < target) {
        idx += step;
        target -= tree_[idx];
      }
      step >>= 1;
    }
    return std::min(idx, n_ - 1);
  }

 private:
  int n_;
  std::vector<int64_t> tree_;
};

}  // namespace

Trainer::Symbol* Trainer::GetCharSymbol(char32_t c) {
  const auto req_it = required_chars_.find(c);
  const uint64_t freq = (req_it != required_chars_.end()) ? req_it->second : 1;
  CHECK_GT(freq, uint64_t{0});
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  auto s = std::make_unique<Symbol>();
  s->is_unk = (kUNKChar == c);
  s->fp = c;
  s->piece = string_util::UnicodeCharToUTF8(c);
  s->freq = freq;
  symbols_cache_.emplace(s->fp, s.get());
  Symbol* s_ptr = s.get();
  allocated_.push_back(std::move(s));
  return s_ptr;
}

Trainer::Symbol* Trainer::GetPairSymbol(const Symbol* left,
                                        const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }

  const uint64_t fp = absl::HashOf(left->fp, right->fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second;
  }

  CHECK(!left->piece.empty());
  CHECK(!right->piece.empty());
  std::string new_piece = absl::StrCat(left->piece, right->piece);

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(string_util::UTF8ToUnicodeText(new_piece))) {
    return nullptr;
  }

  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->piece = std::move(new_piece);
  s->char_len = left->char_len + right->char_len;
  symbols_cache_.emplace(s->fp, s.get());
  Symbol* s_ptr = s.get();
  allocated_.push_back(std::move(s));
  return s_ptr;
}

void Trainer::ComputeFreq(Symbol* symbol) const {
  if (!symbol->needs_recomputation) {
    return;
  }
  symbol->freq = 0;
  for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
    const Position pos = DecodePos(*it);
    // symbols_[sid][left] and symbols_[sid]right] must store
    // the same symbols in symbol->left and symbols->right.
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      it = symbol->positions.erase(it);
    } else {
      symbol->freq += sentences_[pos.sid].second;
      ++it;
    }
  }
  symbol->needs_recomputation = false;
}

int Trainer::GetNextIndex(int sid, int index) const {
  for (size_t i = index + 1; i < symbols_[sid].size(); ++i) {
    if (symbols_[sid][i] == nullptr) {
      continue;
    }
    return i;
  }
  return -1;
}

int Trainer::GetPrevIndex(int sid, int index) const {
  for (int i = index - 1; i >= 0; --i) {
    if (symbols_[sid][i] == nullptr) {
      continue;
    }
    return i;
  }
  return -1;
}

void Trainer::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    symbol->positions.insert(EncodePos(sid, left, right));
    if (!symbol->pending) {
      symbol->pending = true;
      pending_queue_.push_back(symbol);
    }
  }
}

void Trainer::ResetFreq(int sid, int left, int right, const Symbol* best) {
  if (left == -1 || right == -1) {
    return;
  }
  auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr && symbol != best) {
    symbol->needs_recomputation = true;
  }
}

absl::Status Trainer::AcceptSymbol(Symbol* symbol) {
  // Add new bigrams which are created after symbol replacement.
  // We do not need to scan all characters, but scan the neighbors in
  // best_symbol.
  for (const uint64_t& encoded_pos : symbol->positions) {
    const Position pos = DecodePos(encoded_pos);

    if (symbols_[pos.sid][pos.left] == nullptr) {
      // left index might be NULL (set in the previous iteration)
      // when left_symbol == right_symbol.
      continue;
    }
    RET_CHECK(symbols_[pos.sid][pos.right]);

    // We have three bigrams [prev, left], [left, right], [right, next],
    // which are affected with this symbol replacement.
    const int next = GetNextIndex(pos.sid, pos.right);
    const int prev = GetPrevIndex(pos.sid, pos.left);

    // Resets the frequencies of bigrams [prev, left] and [right, next].
    ResetFreq(pos.sid, prev, pos.left, symbol);
    ResetFreq(pos.sid, pos.right, next, symbol);

    // Merges two symbols.
    symbols_[pos.sid][pos.left] = symbol;
    symbols_[pos.sid][pos.right] = nullptr;

    // Makes new symbol bigrams [prev, left] and [left, next].
    AddNewPair(pos.sid, prev, pos.left);
    AddNewPair(pos.sid, pos.left, next);
  }

  // Removes best_symbol so it is not selected again.
  symbols_cache_.erase(symbol->fp);
  symbol->active = false;

  return absl::OkStatus();
}

absl::Status Trainer::Train() {
  ABSL_RETURN_IF_ERROR(status());

  RET_CHECK(normalizer_spec_.escape_whitespaces());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();

  // Load all sentences
  ABSL_RETURN_IF_ERROR(LoadSentences());
  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  const bool auto_vocab = trainer_spec_.auto_character_coverage();

  // Initializes symbols_. symbols_[sid][i] stores an unary symbol.
  symbols_.resize(sentences_.size());
  for (size_t i = 0; i < sentences_.size(); ++i) {
    for (const char32_t c :
         string_util::UTF8ToUnicodeText(sentences_[i].first)) {
      symbols_[i].push_back(GetCharSymbol(c));
    }
  }

  // Makes all bigram symbols.
  for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    for (size_t i = 1; i < symbols_[sid].size(); ++i) {
      AddNewPair(sid, i - 1, i);
    }
  }

  for (Symbol* symbol : pending_queue_) {
    symbol->pending = false;
    ComputeFreq(symbol);
    pq_.push({symbol->freq, symbol});
  }
  pending_queue_.clear();

  RET_CHECK_GE(trainer_spec_.vocab_size(),
               static_cast<int>(meta_pieces_.size()));
  const size_t target_final_pieces_size =
      trainer_spec_.vocab_size() - meta_pieces_.size();
  RET_CHECK_GE(target_final_pieces_size, required_chars_.size());
  const size_t merge_vocab_size =
      target_final_pieces_size - required_chars_.size();

  // In standard BPE, single characters are pre-determined by character_coverage
  // and take up slots beforehand, so we only need to perform |merge_vocab_size|
  // merges. In auto_character_coverage, single characters are not fixed in
  // advance. Global Search will optimize the trade-off between subwords and
  // single characters, allowing up to |target_final_pieces_size| subwords in
  // the extreme case where all budget is allocated to merges.
  const size_t max_merges =
      auto_vocab ? target_final_pieces_size : merge_vocab_size;

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<std::string> dup;
  std::vector<MergeCandidate> merge_candidates;

  // Main loop.
  while (merge_candidates.size() < max_merges) {
    Symbol* best_symbol = nullptr;
    while (!pq_.empty()) {
      QueueEntry entry = pq_.top();
      Symbol* symbol = entry.symbol;
      if (!symbol->active) {
        pq_.pop();
        continue;
      }
      if (entry.freq != symbol->freq) {
        pq_.pop();
        continue;
      }
      if (symbol->needs_recomputation) {
        pq_.pop();
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
        continue;
      }
      best_symbol = symbol;
      pq_.pop();
      break;
    }

    if (best_symbol == nullptr) {
      LOG(WARNING) << "No valid symbol found";
      break;
    }

    if (!dup.insert(best_symbol->piece).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol->fp);
      best_symbol->active = false;
      continue;
    }

    merge_candidates.push_back(
        {best_symbol->piece, static_cast<int64_t>(best_symbol->freq)});

    if (merge_candidates.size() % 100 == 0) {
      LOG(INFO) << "Merged: freq=" << best_symbol->freq
                << " merges=" << merge_candidates.size()
                << " piece=" << best_symbol->piece;
    }

    ABSL_RETURN_IF_ERROR(AcceptSymbol(best_symbol));

    for (Symbol* symbol : pending_queue_) {
      symbol->pending = false;
      if (symbol->active) {
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
      }
    }
    pending_queue_.clear();
  }  // end of main loop

  LOG(INFO) << "Completed " << merge_candidates.size() << " merge steps.";

  if (auto_vocab) {
    final_pieces_ =
        PrunePiecesWithGlobalSearch(merge_candidates, target_final_pieces_size);
  } else {
    for (const auto& candidate : merge_candidates) {
      final_pieces_.emplace_back(candidate.piece,
                                 -static_cast<float>(final_pieces_.size()));
    }
    // Adds required_chars_
    for (const auto& w : Sorted(required_chars_)) {
      const Symbol* symbol = GetCharSymbol(w.first);
      final_pieces_.emplace_back(symbol->piece,
                                 -static_cast<float>(final_pieces_.size()));
    }
  }

  allocated_.clear();
  symbols_cache_.clear();

  return Save();
}

std::vector<std::pair<std::string, float>> Trainer::PrunePiecesWithGlobalSearch(
    absl::Span<const MergeCandidate> merge_candidates,
    size_t target_final_pieces_size) {
  // 1. Count total occurrences of each Unicode character across the corpus.
  absl::flat_hash_map<char32_t, int64_t> char_counts;
  for (const auto& [sentence, weight] : sentences_) {
    for (const char32_t c : string_util::UTF8ToUnicodeText(sentence)) {
      char_counts[c] += weight;
    }
  }

  // 2. Decompose all candidate subwords into Unicode characters.
  const size_t num_candidates = merge_candidates.size();
  std::vector<absl::InlinedVector<char32_t, 8>> candidate_chars(num_candidates);
  for (size_t i = 0; i < num_candidates; ++i) {
    for (const char32_t c :
         string_util::UTF8ToUnicodeText(merge_candidates[i].piece)) {
      candidate_chars[i].push_back(c);
    }
  }

  // 3. Collect multibyte character candidates and their token reduction gain.
  // When a multi-byte character is in the vocabulary, it consumes 1 token
  // instead of `byte_len` fallback byte tokens, yielding (byte_len - 1) saving.
  struct SingleCharCandidate {
    char32_t c = 0;
    int64_t gain = 0;
    int64_t count = 0;
  };
  std::vector<SingleCharCandidate> single_char_candidates;
  for (const auto& [c, count] : char_counts) {
    if (required_chars_.contains(c)) {
      continue;  // Already in required_chars_ (if specified).
    }
    const int byte_len = UTF8ByteLength(c);
    if (byte_len >= 2) {
      const int64_t gain = count * (byte_len - 1);
      if (gain > 0) {
        single_char_candidates.push_back({c, gain, count});
      }
    }
  }

  // Sort character candidates by:
  // 1. Token reduction gain descending
  // 2. Frequency count descending
  // 3. Unicode codepoint ascending (lexicographical tie-breaker)
  absl::c_sort(single_char_candidates, [](const SingleCharCandidate& lhs,
                                          const SingleCharCandidate& rhs) {
    return std::tie(rhs.gain, rhs.count, lhs.c) <
           std::tie(lhs.gain, lhs.count, rhs.c);
  });

  const size_t num_char_candidates = single_char_candidates.size();
  absl::flat_hash_map<char32_t, int> char_to_rank;
  char_to_rank.reserve(num_char_candidates);
  for (size_t i = 0; i < num_char_candidates; ++i) {
    char_to_rank[single_char_candidates[i].c] = static_cast<int>(i);
  }

  LOG(INFO) << "Multi-byte character candidates: " << num_char_candidates;

  // 4. Prefix sums for merges: merge_prefix_gain[k] = sum of scores for merges
  // [0, k)
  std::vector<int64_t> merge_prefix_gain(num_candidates + 1, 0);
  for (size_t i = 0; i < num_candidates; ++i) {
    merge_prefix_gain[i + 1] = merge_prefix_gain[i] + merge_candidates[i].score;
  }

  // 5. Incremental Global Search over k in [0, num_candidates] with Character
  // Closure Constraint: Any character appearing in the selected subwords [0, k)
  // MUST be included in the vocabulary (closed under subword composition).
  //
  // As k increases (forward scan), newly required characters transition from
  // the standalone pool to the subword-required pool. Because each character
  // transitions at most once, maintaining available standalone characters via
  // Fenwick Trees takes O(C log C) amortized time across all k.
  const size_t budget = target_final_pieces_size - required_chars_.size();

  // available_chars[x] indicates whether single_char_candidates[x] would fall
  // back to raw bytes unless explicitly added to the vocabulary:
  // - 1: Not covered by the selected subwords [0, k); will become byte fallback
  //      unless selected into the remaining standalone vocabulary slots.
  // - 0: Already included in the vocabulary as a constituent character of the
  //      selected subwords [0, k); guaranteed not to fall back to bytes.
  // Query(x) returns the count of such byte-fallback candidates in [0..x].
  FenwickTree available_chars(num_char_candidates);

  // available_char_gains[x] tracks the token reduction gain (from avoiding
  // byte fallback) if single_char_candidates[x] is added to the vocabulary:
  // - candidate[x].gain if available_chars[x] == 1
  // - 0 if available_chars[x] == 0 (gain already accounted for in
  // subword_char_gain) Query(x) returns the sum of gains of the available
  // candidates in [0..x].
  FenwickTree available_char_gains(num_char_candidates);
  for (size_t i = 0; i < num_char_candidates; ++i) {
    available_chars.Add(i, 1);
    available_char_gains.Add(i, single_char_candidates[i].gain);
  }

  // OptimalCut records the best subword prefix and single character allocation.
  struct OptimalCut {
    size_t num_subwords = 0;  // Number of prefix subwords selected (k).
    size_t num_single_chars =
        0;  // Number of top single multibyte characters selected.
    int64_t total_gain = std::numeric_limits<int64_t>::lowest();
  };
  OptimalCut best_cut;

  absl::flat_hash_set<char32_t> subword_chars;
  int64_t subword_char_gain = 0;
  // Number of unique non-required characters required by subwords [0, k).
  size_t subword_char_count = 0;

  for (size_t k = 0; k <= num_candidates; ++k) {
    // Check feasibility: selecting k subwords and their required constituent
    // characters must not exceed the vocabulary budget.
    if (k + subword_char_count <= budget) {
      const size_t remaining_slots = budget - k - subword_char_count;
      const size_t num_available_chars =
          static_cast<size_t>(available_chars.Query(num_char_candidates - 1));
      const size_t num_single_chars =
          std::min(remaining_slots, num_available_chars);
      int64_t single_char_gain = 0;
      if (num_single_chars > 0) {
        const int idx = available_chars.FindByPrefixCount(
            static_cast<int>(num_single_chars));
        single_char_gain = available_char_gains.Query(idx);
      }
      const int64_t total_gain =
          merge_prefix_gain[k] + subword_char_gain + single_char_gain;
      // Prefer larger k on tie (prefer subwords)
      if (total_gain >= best_cut.total_gain) {
        best_cut.total_gain = total_gain;
        best_cut.num_subwords = k;
        best_cut.num_single_chars = num_single_chars;
      }
    }

    // Incorporate characters of merge_candidates[k] for the next step k + 1.
    // Guarded by k < num_candidates because the loop reaches k ==
    // num_candidates to evaluate the full-prefix boundary case.
    if (k < num_candidates) {
      for (const char32_t c : candidate_chars[k]) {
        if (required_chars_.contains(c) || subword_chars.contains(c)) {
          continue;
        }
        subword_chars.insert(c);
        subword_char_count++;
        if (const auto it = char_to_rank.find(c); it != char_to_rank.end()) {
          const int r = it->second;
          available_chars.Add(r, -1);
          available_char_gains.Add(r, -single_char_candidates[r].gain);
          subword_char_gain += single_char_candidates[r].gain;
        }
      }
    }
  }

  // Reconstruct exact character set for best_cut:
  // (1) Subword characters: chars appearing in merge_candidates[0 ...
  // best_cut.num_subwords - 1]
  absl::btree_set<char32_t> selected_subword_chars;
  for (size_t i = 0; i < best_cut.num_subwords; ++i) {
    for (const char32_t c : candidate_chars[i]) {
      if (!required_chars_.contains(c)) {
        selected_subword_chars.insert(c);
      }
    }
  }

  // (2) Top single characters (best_cut.num_single_chars)
  std::vector<char32_t> selected_single_chars;
  for (size_t i = 0; i < num_char_candidates &&
                     selected_single_chars.size() < best_cut.num_single_chars;
       ++i) {
    if (!selected_subword_chars.contains(single_char_candidates[i].c)) {
      selected_single_chars.push_back(single_char_candidates[i].c);
    }
  }

  LOG(INFO) << "Global search pruning:";
  LOG(INFO) << " - Selected subwords: " << best_cut.num_subwords
            << " (truncated " << (num_candidates - best_cut.num_subwords)
            << " tail merges)";
  LOG(INFO) << " - Subword-covered characters: "
            << selected_subword_chars.size();
  LOG(INFO) << " - Standalone multibyte characters: "
            << selected_single_chars.size();
  LOG(INFO) << " - Total gain: " << best_cut.total_gain;

  // Populate final_pieces:
  std::vector<std::pair<std::string, float>> final_pieces;
  final_pieces.reserve(target_final_pieces_size);

  auto append_piece = [&](absl::string_view piece) {
    final_pieces.emplace_back(std::string(piece),
                              -static_cast<float>(final_pieces.size()));
  };

  // (a) Selected subwords in strict merge order [0, best_cut.num_subwords)
  for (size_t i = 0; i < best_cut.num_subwords; ++i) {
    append_piece(merge_candidates[i].piece);
  }
  // (b) Subword-covered multibyte characters (sorted for determinism)
  for (const char32_t c : selected_subword_chars) {
    append_piece(string_util::UnicodeCharToUTF8(c));
  }
  // (c) Standalone multibyte characters
  for (const char32_t c : selected_single_chars) {
    append_piece(string_util::UnicodeCharToUTF8(c));
  }
  // (d) Base required_chars_ (if specified)
  for (const auto& w : Sorted(required_chars_)) {
    const Symbol* symbol = GetCharSymbol(w.first);
    append_piece(symbol->piece);
  }

  return final_pieces;
}

}  // namespace sentencepiece::bpe
