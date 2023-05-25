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

#include <malloc.h>
#ifdef TCMALLOC
#include <gperftools/malloc_extension.h>
#endif

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "pretokenizer_for_training.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "util.h"

namespace sentencepiece {
namespace bpe {

std::string Trainer::Symbol::ToString() const {
  string_util::UnicodeText ut;
  AppendCharsToText(&ut);
  return string_util::UnicodeTextToUTF8(ut);
}

Trainer::Symbol::~Symbol() {
  if (chars_size > 2) {
    delete[] chars_ext;
  }
}

void Trainer::Symbol::AppendChar(char32 c) {
  char32 *ext;
  switch (chars_size++) {
    case 0:
      chars_embed[0] = c;
      break;
    case 1:
      chars_embed[1] = c;
      break;
    case 2:
      ext = new char32[3];
      *reinterpret_cast<uint64_t *>(ext) = chars_embed_pair;
      ext[2] = c;
      chars_ext = ext;
      break;
    default:
      ext = new char32[chars_size];
      memcpy(ext, chars_ext, sizeof(char32) * (chars_size - 1));
      ext[chars_size - 1] = c;
      delete[] chars_ext;
      chars_ext = ext;
      break;
  }
}

void Trainer::Symbol::AssignChars(const string_util::UnicodeText &text) {
  if (chars_size > 2) {
    delete[] chars_ext;
  }
  chars_size = text.size();
  switch (chars_size) {
    case 0:
      break;
    case 1:
      chars_embed[0] = text[0];
      break;
    case 2:
      chars_embed_pair = *reinterpret_cast<const uint64_t *>(text.data());
      break;
    default:
      chars_ext = new char32[chars_size];
      memcpy(chars_ext, text.data(), sizeof(char32) * chars_size);
      break;
  }
}

void Trainer::Symbol::AppendCharsToText(string_util::UnicodeText *text) const {
  switch (chars_size) {
    case 0:
      break;
    case 1:
      text->push_back(chars_embed[0]);
      break;
    case 2:
      text->push_back(chars_embed[0]);
      text->push_back(chars_embed[1]);
      break;
    default:
      text->insert(text->end(), chars_ext, chars_ext + chars_size);
      break;
  }
}

Trainer::Symbol *Trainer::GetCharSymbol(char32 c) {
  const uint64 freq = port::FindWithDefault(required_chars_, c, 1);
  CHECK_GT(freq, 0);
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  auto &s = allocated_.emplace_back(nullptr, nullptr, c, freq);
  s.AppendChar(c);
  port::InsertOrDie(&symbols_cache_, s.fp, &s);
  return &s;
}

Trainer::Symbol *Trainer::GetPairSymbol(const Symbol *left,
                                        const Symbol *right) {
  if (left == nullptr || right == nullptr || left->IsUnk() || right->IsUnk()) {
    return nullptr;
  }

  const uint64 fp = port::FingerprintCat(left->fp, right->fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second;
  }

  CHECK(left->CharsSize() > 0);
  CHECK(right->CharsSize() > 0);
  string_util::UnicodeText ut;
  left->AppendCharsToText(&ut);
  right->AppendCharsToText(&ut);

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(ut)) {
    return nullptr;
  }

  auto &s = allocated_.emplace_back(left, right, fp);
  s.AssignChars(ut);
  port::InsertOrDie(&symbols_cache_, s.fp, &s);
  return &s;
}

void Trainer::ComputeFreq(Symbol *symbol) const {
  if (symbol->freq > 0) {  // if freq == 0, re-computation is required.
    return;
  }
  CHECK_EQ(0, symbol->freq);
  std::vector<uint64_t> erased;
  for (uint64_t i = 0; i < symbol->positions.size(); i++) {
    const Position pos = DecodePos(symbol->positions[i]);
    // symbols_[sid][left] and symbols_[sid][right] must store
    // the same symbols in symbol->left and symbols->right.
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      erased.push_back(i);
    } else {
      symbol->freq += freqs_[pos.sid];
    }
  }
  if (!erased.empty()) {
    size_t new_positions_size = symbol->positions.size() - erased.size();
    std::vector<uint64_t> new_positions(new_positions_size);
    memcpy(new_positions.data(), symbol->positions.data(), erased[0] * sizeof(uint64_t));
    uint64_t offset = 0;
    for (uint64_t i = 0; i < erased.size(); i++) {
      uint64_t s = erased[i];
      uint64_t f = i < (erased.size() - 1) ? erased[i + 1] : symbol->positions.size();
      memcpy(new_positions.data() + s - offset++,
             symbol->positions.data() + s + 1,
             (f - s - 1) * sizeof(uint64_t));
    }
    symbol->positions = std::move(new_positions);
  }
}

int Trainer::GetNextIndex(int sid, int index) const {
  auto &sentence = symbols_[sid];
  for (size_t i = index + 1; i < sentence.size(); ++i) {
    if (sentence[i] == nullptr) continue;
    return i;
  }
  return -1;
}

int Trainer::GetPrevIndex(int sid, int index) const {
  auto &sentence = symbols_[sid];
  for (int i = index - 1; i >= 0; --i) {
    if (sentence[i] == nullptr) continue;
    return i;
  }
  return -1;
}

void Trainer::AddNewPair(int sid, int left, int right, bool sort) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    active_symbols_.insert(symbol);
    uint64_t pos = EncodePos(sid, left, right);
    if (sort) {
      auto it = std::lower_bound(symbol->positions.begin(), symbol->positions.end(), pos);
      if (it != symbol->positions.end()) {
        if (*it == pos) {
          return;
        }
        auto offset = it - symbol->positions.begin();
        if (symbol->positions.capacity() >= symbol->positions.size() + 1) {
          symbol->positions.emplace_back();
          memmove(symbol->positions.data() + offset + 1,
                  symbol->positions.data() + offset,
                  (symbol->positions.size() - offset - 1) * sizeof(uint64_t));
        } else {
          std::vector<uint64_t> new_positions(symbol->positions.size() + 1);
          memcpy(new_positions.data(), symbol->positions.data(), offset * sizeof(uint64_t));
          memcpy(new_positions.data() + offset + 1,
                 symbol->positions.data() + offset,
                 (symbol->positions.size() - offset - 1) * sizeof(uint64_t));
          symbol->positions = std::move(new_positions);
        }
        symbol->positions[offset] = pos;
      } else {
        symbol->positions.push_back(pos);
      }
    } else {
      symbol->positions.push_back(pos);
    }
  }
}

void Trainer::SortSymbolPositions() {
  auto pool = absl::make_unique<ThreadPool>(trainer_spec_.num_threads());
  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      for (size_t i = n; i < allocated_.size();
        i += trainer_spec_.num_threads()) {
        auto &symbol = allocated_[i];
        symbol.positions.shrink_to_fit();
        std::sort(symbol.positions.begin(), symbol.positions.end());
      }
    });
  }
}

void Trainer::ResetFreq(int sid, int left, int right, const Symbol *best) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr && symbol != best) {
    symbol->freq = 0;
  }
}

void Trainer::UpdateActiveSymbols(ThreadPool *pool) {
  std::vector<Symbol *> symbols;
  symbols.reserve(symbols_cache_.size());
  for (auto &it : symbols_cache_) {
    Symbol *symbol = it.second;
    if (symbol->IsBigram()) {
      symbols.push_back(symbol);
    }
  }
   pool->Loop(0, symbols.size(), [this, &symbols](const int beg, const int end) {
     for (int i = beg; i < end; i++) {
       ComputeFreq(symbols[i]);
     }
   });
   pool->Wait();

  // At least kMinActiveSymbolsSize symbols must be in |active_symbols_|.
  constexpr int kMinActiveSymbolsSize = 1000;

  // Keeps top 5% frequent symbols.
  constexpr float kTopFrequentRatio = 0.05;
  const int size =
      std::min<int>(std::max<int>(kMinActiveSymbolsSize,
                                  symbols_cache_.size() * kTopFrequentRatio),
                    symbols.size());

  std::partial_sort(symbols.begin(), symbols.begin() + size, symbols.end(),
                    [](Symbol *s1, Symbol *s2) { return s1->freq > s2->freq; });
  LOG(INFO) << "Updating active symbols. max_freq=" << symbols[0]->freq
            << " min_freq=" << symbols[size - 1]->freq;

  active_symbols_.clear();
  active_symbols_.insert(symbols.begin(), symbols.begin() + size);
}

util::Status Trainer::LoadSentencesFromCache(filesystem::ReadableFile *cache_file) {
  LOG(INFO) << "Loading cached sentences from "
            << trainer_spec_.cache_sentence_frequencies_file();
  std::string freq(sizeof(Sentence::second_type), 0),
              size(sizeof(size_t), 0),
              rcp(sizeof(char32) + sizeof(int64), 0);
  absl::string_view token;
  size_t required_chars_size;
  cache_file->ReadBuffer(&size);
  required_chars_size = *reinterpret_cast<size_t *>(size.data());
  for (size_t i = 0; i < required_chars_size; i++) {
    if (!cache_file->ReadBuffer(&rcp)) {
      return cache_file->status();;
    }
    required_chars_[*reinterpret_cast<char32 *>(rcp.data())] =
        *reinterpret_cast<int64 *>(rcp.data() + 4);
  }
  while (cache_file->ReadLine(&token) && cache_file->ReadBuffer(&freq)) {
    sentences_.emplace_back(
        token,
        *reinterpret_cast<const Sentence::second_type *>(freq.data())
    );
  }
  LOG(INFO) << "Loaded " << sentences_.size() << " cached sentences";
  return cache_file->status();
}

util::Status Trainer::StoreSentencesToCache() {
  auto writer = filesystem::NewWritableFile(
      trainer_spec_.cache_sentence_frequencies_file(), true);
  if (writer->status() == util::OkStatus()) {
    LOG(INFO) << "Storing " << sentences_.size() << " sentences to cache "
              << trainer_spec_.cache_sentence_frequencies_file();
    std::unique_ptr<char[]> buffer;
    size_t allocated = required_chars_.size();
    writer->Write(absl::string_view(
        reinterpret_cast<char *>(&allocated), sizeof(allocated)));
    for (auto &c : required_chars_) {
      char buf[4 + sizeof(Sentence::second_type)];
      *reinterpret_cast<char32 *>(buf) = c.first;
      *reinterpret_cast<Sentence::second_type *>(buf + 4) = c.second;
      if (!writer->Write(absl::string_view(buf, sizeof(buf)))) {
        return writer->status();
      }
    }
    allocated = 0;
    for (auto &s : sentences_) {
      size_t size = s.first.size() + 1 + sizeof(Sentence::second_type);
      if (allocated < size) {
        buffer = std::make_unique<char[]>(size);
        allocated = size;
      }
      memcpy(buffer.get(), s.first.data(), s.first.size());
      buffer[s.first.size()] = 0;
      memcpy(buffer.get() + s.first.size() + 1,
             &s.second,
             sizeof(Sentence::second_type));
      if (!writer->Write(absl::string_view(buffer.get(), size))) {
        return writer->status();
      }
    }
    return util::OkStatus();
  } else {
    return writer->status();
  }
}

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());
  CHECK_EQ_OR_RETURN(TrainerSpec::BPE, trainer_spec_.model_type());

  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  active_symbols_.clear();
  std::unique_ptr<filesystem::ReadableFile> cache_file;

  if (!trainer_spec_.cache_sentence_frequencies_file().empty()) {
    cache_file = filesystem::NewReadableFile(
        trainer_spec_.cache_sentence_frequencies_file(), true, 0);
    if (cache_file->status() == util::OkStatus()) {
      RETURN_IF_ERROR(LoadSentencesFromCache(cache_file.get()));
    }
  }

  if (sentences_.empty()) {
    // Load all sentences
    RETURN_IF_ERROR(LoadSentences(false));

    if (trainer_spec_.split_by_whitespace()) {
      SplitSentencesByWhitespace();
    }

    if (!trainer_spec_.cache_sentence_frequencies_file().empty()) {
      RETURN_IF_ERROR(StoreSentencesToCache());
      return util::OkStatus();
    }
  }

  // Pretokenizer applied only in training time.
  // Pretokenizer is used as a constraint of piece extractions.
  const auto *pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();

  if (pretokenizer || !trainer_spec_.pretokenization_delimiter().empty()) {
    absl::string_view delimiter = trainer_spec_.pretokenization_delimiter();
    LOG(INFO) << "Preprocessing with pretokenizer...";
    for (auto &w : sentences_) {
      if (pretokenizer) {
        w.first = bank_->View(absl::StrJoin(pretokenizer->PreTokenize(w.first),
                                            TrainerInterface::kUPPBoundaryStr));
      } else if (!delimiter.empty()) {
        w.first = bank_->View(absl::StrReplaceAll(
            w.first, {{delimiter, TrainerInterface::kUPPBoundaryStr}}));
      }
    }
  }

  LOG(INFO) << "Initializing symbols...";
  // Initializes symbols_. symbols_[sid][i] stores an unary symbol.
  symbols_.reserve(sentences_.size());
  freqs_.reserve(sentences_.size());
  size_t overflows = 0;
  constexpr size_t uint16_max = static_cast<size_t>(std::numeric_limits<uint16_t>::max());
  for (auto &sentence : sentences_) {
    auto &&text = string_util::UTF8ToUnicodeText(sentence.first);
    if (symbols_.size() == symbols_.capacity()) {
      symbols_.reserve(sentences_.size() + overflows);
      freqs_.reserve(sentences_.size() + overflows);
    }
    auto &symbols_sentence = symbols_.emplace_back();
    symbols_sentence.reserve(std::min(text.size(), uint16_max));
    for (const char32 c : text) {
      if (symbols_sentence.size() == uint16_max) {
        // this sentence is too long, must split to be able to call EncodePos
        overflows++;
        symbols_sentence.shrink_to_fit();
        symbols_sentence = symbols_.emplace_back();
        // we can overflow several times, but it doesn't hurt to overallocate
        symbols_sentence.reserve(std::min(text.size() - uint16_max, uint16_max));
      }
      symbols_sentence.push_back(GetCharSymbol(c));
    }
    freqs_.push_back(sentence.second);
  }

  cache_file.reset();
  sentences_.clear();
  sentences_.shrink_to_fit();
  #ifdef TCMALLOC
  MallocExtension::instance()->ReleaseFreeMemory();
  #endif
  size_t unisize = allocated_.size();
  LOG(INFO) << "Allocated " << unisize << " chars with " << overflows << " overflows";

  // Makes all bigram symbols.
  for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    if (sid % 10000000 == 0 && sid > 0) {
      LOG(INFO) << "Generated pairs from " << sid << " symbols";
    }
    for (size_t i = 1; i < symbols_[sid].size(); ++i) {
      AddNewPair(sid, i - 1, i, false);
    }
  }

  LOG(INFO) << "Allocated " << allocated_.size() - unisize << " pairs";
  malloc_stats();

  LOG(INFO) << "Sorting positions...";
  SortSymbolPositions();
  #ifdef TCMALLOC
  MallocExtension::instance()->ReleaseFreeMemory();
  #endif
  malloc_stats();

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  CHECK_GE_OR_RETURN(vocab_size, 0);

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<absl::string_view> dup;

  auto pool = absl::make_unique<ThreadPool>(trainer_spec_.num_threads());
  std::mutex sync;

  // Main loop.
  CHECK_OR_RETURN(final_pieces_.empty());
  LOG(INFO) << "Will do " << vocab_size << " iterations";
  while (final_pieces_.size() < static_cast<size_t>(vocab_size)) {
    constexpr int kUpdateActiveSymbolsInterval = 100;
    if (final_pieces_.size() % kUpdateActiveSymbolsInterval == 0) {
      UpdateActiveSymbols(pool.get());
    }

    // Scanning active symbols, finds the best_symbol with highest freq.
    Symbol *best_symbol = nullptr;
    {
      for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
        pool->Schedule([&, n]() {
          Symbol *my_best_symbol = nullptr;
          size_t pos = 0;
          for (auto &it : active_symbols_) {
            if ((n + pos++) % trainer_spec_.num_threads()) {
              continue;
            }
            Symbol *symbol = it;
            ComputeFreq(symbol);
            // If the frequency is the same, take shorter symbol.
            // if the length is the same, use lexicographical comparison
            if (my_best_symbol == nullptr ||
                (symbol->freq > my_best_symbol->freq ||
                 (symbol->freq == my_best_symbol->freq &&
                  (symbol->CharsSize() < my_best_symbol->CharsSize() ||
                   (symbol->CharsSize() == my_best_symbol->CharsSize() &&
                    symbol->ToString() < my_best_symbol->ToString()))))) {
              my_best_symbol = symbol;
            }
          }
          std::lock_guard lock(sync);
          if (best_symbol == nullptr ||
              (my_best_symbol->freq > best_symbol->freq ||
               (my_best_symbol->freq == best_symbol->freq &&
                (my_best_symbol->CharsSize() < best_symbol->CharsSize() ||
                 (my_best_symbol->CharsSize() == best_symbol->CharsSize() &&
                  my_best_symbol->ToString() < best_symbol->ToString()))))) {
            best_symbol = my_best_symbol;
          }
        });
      }
      pool->Wait();
    }

    if (best_symbol == nullptr) {
      LOG(WARNING) << "No valid symbol found";
      break;
    }

    if (!dup.emplace(best_symbol->ToString()).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol->fp);
      active_symbols_.erase(best_symbol);
      continue;
    }

    // Stores the best_symbol in the final output.
    final_pieces_.emplace_back(best_symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));

    if (final_pieces_.size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << best_symbol->freq
                << " size=" << final_pieces_.size()
                << " all=" << symbols_cache_.size()
                << " active=" << active_symbols_.size()
                << " piece=" << best_symbol->ToString()
                << " (" << best_symbol->CharsSize()
                << ", " << best_symbol->positions.size() << ")";
    }

    // Add new bigrams which are created after symbol replacement.
    // We do not need to scan all characters, but scan the neighbors in
    // best_symbol.
    for (uint64_t encoded_pos : best_symbol->positions) {
      const Position pos = DecodePos(encoded_pos);

      if (symbols_[pos.sid][pos.left] == nullptr) {
        // left index might be NULL (set in the previous iteration)
        // when left_symbol == right_symbol.
        continue;
      }

      CHECK_OR_RETURN(symbols_[pos.sid][pos.right]);

      // We have three bigrams [prev, left], [left, right], [right, next],
      // which are affected with this symbol replacement.
      const int next = GetNextIndex(pos.sid, pos.right);
      const int prev = GetPrevIndex(pos.sid, pos.left);

      // Resets the frequencies of bigrams [prev, left] and [right, next].
      ResetFreq(pos.sid, prev, pos.left, best_symbol);
      ResetFreq(pos.sid, pos.right, next, best_symbol);

      // Merges two symbols.
      symbols_[pos.sid][pos.left] = best_symbol;
      symbols_[pos.sid][pos.right] = nullptr;

      // Makes new symbol bigrams [prev, left] and [left, next].
      AddNewPair(pos.sid, prev, pos.left, true);
      AddNewPair(pos.sid, pos.left, next, true);
    }

    // Removes best_symbol so it is not selected again.
    symbols_cache_.erase(best_symbol->fp);
    active_symbols_.erase(best_symbol);
  }  // end of main loop

  // Adds required_chars_
  for (const auto &w : Sorted(required_chars_, trainer_spec_.num_threads())) {
    const Symbol *symbol = GetCharSymbol(w.first);
    final_pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));
  }

  allocated_.clear();
  allocated_.shrink_to_fit();
  freqs_.clear();
  freqs_.shrink_to_fit();

  return Save();
}
}  // namespace bpe
}  // namespace sentencepiece
