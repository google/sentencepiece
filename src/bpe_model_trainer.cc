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
#include <string>
#include <unordered_set>
#include <vector>

#include "pretokenizer_for_training.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/hash/hash.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/string_view.h"
#include "util.h"

#ifdef SPM_NLCODEC_BPE
#include "contrib/nlcodec/bpe_model_trainer_nlcodec.h"
ABSL_DECLARE_FLAG(bool, nlcodec_bpe);
#endif  // SPM_NLCODEC_BPE

namespace sentencepiece::bpe {

std::string Trainer::Symbol::ToString() const {
  return string_util::UnicodeTextToUTF8(chars);
}

Trainer::Symbol* Trainer::GetCharSymbol(char32_t c) {
  const uint64_t freq = port::FindWithDefault(required_chars_, c, 1);
  CHECK_GT(freq, 0);
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  auto s = std::make_unique<Symbol>();
  s->is_unk = (kUNKChar == c);
  s->fp = c;
  s->chars.push_back(c);
  s->freq = freq;
  port::InsertOrDie(&symbols_cache_, s->fp, s.get());
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

  CHECK(!left->chars.empty());
  CHECK(!right->chars.empty());
  string_util::UnicodeText ut;
  for (const char32_t c : left->chars) {
    ut.push_back(c);
  }
  for (const char32_t c : right->chars) {
    ut.push_back(c);
  }

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(ut)) {
    return nullptr;
  }

  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->chars = ut;
  port::InsertOrDie(&symbols_cache_, s->fp, s.get());
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
  RETURN_IF_ERROR(status());

#ifdef SPM_NLCODEC_BPE
  if (absl::GetFlag(FLAGS_nlcodec_bpe)) {
    return TrainFast();
  }
#endif  // SPM_NLCODEC_BPE

  RET_CHECK(normalizer_spec_.escape_whitespaces());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();

  // Load all sentences
  RETURN_IF_ERROR(LoadSentences());

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  // Pretokenizer applied only in training time.
  // Pretokenizer is used as a constraint of piece extractions.
  const auto* pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();

  if ((pretokenizer != nullptr) ||
      !trainer_spec_.pretokenization_delimiter().empty()) {
    absl::string_view delimiter = trainer_spec_.pretokenization_delimiter();
    LOG(INFO) << "Preprocessing with pretokenizer...";
    for (auto& w : sentences_) {
      if (pretokenizer != nullptr) {
        w.first = absl::StrJoin(pretokenizer->PreTokenize(w.first),
                                TrainerInterface::kUPPBoundaryStr);
      } else if (!delimiter.empty()) {
        w.first = absl::StrReplaceAll(
            w.first, {{delimiter, TrainerInterface::kUPPBoundaryStr}});
      }
    }
  }

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

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  RET_CHECK_GE(vocab_size, 0);

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<std::string> dup;

  // Main loop.
  RET_CHECK(final_pieces_.empty());
  while (final_pieces_.size() < static_cast<size_t>(vocab_size)) {
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

    if (!dup.insert(best_symbol->ToString()).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol->fp);
      best_symbol->active = false;
      continue;
    }

    // Stores the best_symbol in the final output.
    final_pieces_.emplace_back(best_symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));

    if (final_pieces_.size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << best_symbol->freq
                << " size=" << final_pieces_.size()
                << " all=" << symbols_cache_.size() << " active=" << pq_.size()
                << " piece=" << best_symbol->ToString();
    }

    RETURN_IF_ERROR(AcceptSymbol(best_symbol));

    for (Symbol* symbol : pending_queue_) {
      symbol->pending = false;
      if (symbol->active) {
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
      }
    }
    pending_queue_.clear();
  }  // end of main loop

  // Adds required_chars_
  for (const auto& w : Sorted(required_chars_)) {
    const Symbol* symbol = GetCharSymbol(w.first);
    final_pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));
  }

  allocated_.clear();
  symbols_cache_.clear();

  return Save();
}

#ifdef SPM_NLCODEC_BPE
absl::Status Trainer::TrainFast() {
  RET_CHECK(normalizer_spec_.escape_whitespaces());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  RETURN_IF_ERROR(LoadSentences());

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  RET_CHECK_GE(vocab_size, 0);
  RET_CHECK(final_pieces_.empty());

  RETURN_IF_ERROR(
      nlcodec::RunFastBPEMerges(sentences_, vocab_size, &final_pieces_,
                                [this](const string_util::UnicodeText& ut) {
                                  return IsValidSentencePiece(ut);
                                }));

  // Add required_chars_
  for (const auto& w : Sorted(required_chars_)) {
    const Symbol* symbol = GetCharSymbol(w.first);
    final_pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));
  }

  allocated_.clear();
  symbols_cache_.clear();

  return Save();
}
#endif  // SPM_NLCODEC_BPE
}  // namespace sentencepiece::bpe
