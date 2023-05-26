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

uint32_t Trainer::GetCharSymbol(char32 c, bool require_cache) {
  const uint64 freq = port::FindWithDefault(required_chars_, c, 1);
  CHECK_GT(freq, 0);
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  CHECK(!require_cache);
  uint32_t index = allocated_.size();
  CHECK_LT(index, ~0u);
  auto &s = allocated_.emplace_back(~0u, ~0u, c, freq);
  s.AppendChar(c);
  port::InsertOrDie(&symbols_cache_, s.fp, index);
  return index;
}

uint32_t Trainer::GetPairSymbol(uint32_t left, uint32_t right) {
  if (left == ~0u || right == ~0u) {
    return ~0u;
  }
  auto &left_symbol = allocated_[left];
  auto &right_symbol = allocated_[right];
  if (left_symbol.IsUnk() || right_symbol.IsUnk()) {
    return ~0u;
  }

  const uint64 fp = port::FingerprintCat(left_symbol.fp, right_symbol.fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second;
  }

  CHECK(left_symbol.CharsSize() > 0);
  CHECK(right_symbol.CharsSize() > 0);
  string_util::UnicodeText ut;
  left_symbol.AppendCharsToText(&ut);
  right_symbol.AppendCharsToText(&ut);

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(ut)) {
    return ~0u;
  }

  uint32_t index = allocated_.size();
  CHECK_LT(index, ~0u);
  auto &s = allocated_.emplace_back(left, right, fp);
  s.AssignChars(ut);
  port::InsertOrDie(&symbols_cache_, s.fp, index);
  return index;
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

uint32_t Trainer::GetNextIndex(uint32_t sid, uint32_t index) const {
  auto &sentence = symbols_[sid];
  for (uint32_t i = index + 1; i < sentence.size(); ++i) {
    if (sentence[i] == ~0u) continue;
    return i;
  }
  return ~0u;
}

uint32_t Trainer::GetPrevIndex(uint32_t sid, uint32_t index) const {
  auto &sentence = symbols_[sid];
  for (int64 i = static_cast<int64>(index) - 1; i >= 0; --i) {
    if (sentence[i] == ~0u) continue;
    return i;
  }
  return ~0u;
}

void Trainer::AddNewPair(uint32_t sid, uint32_t left, uint32_t right) {
  if (left == ~0u || right == ~0u) return;
  auto symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol == ~0u) {
    return;
  }
  active_symbols_.insert(symbol);
  uint64_t pos = EncodePos(sid, left, right);
  auto &symbol_ref = allocated_[symbol];
  auto it = std::lower_bound(symbol_ref.positions.begin(),
                             symbol_ref.positions.end(),
                             pos);
  if (it != symbol_ref.positions.end()) {
    if (*it == pos) {
      return;
    }
    auto offset = it - symbol_ref.positions.begin();
    if (symbol_ref.positions.capacity() >= symbol_ref.positions.size() + 1) {
      symbol_ref.positions.emplace_back();
      memmove(symbol_ref.positions.data() + offset + 1,
              symbol_ref.positions.data() + offset,
              (symbol_ref.positions.size() - offset - 1) * sizeof(uint64_t));
    } else {
      std::vector<uint64_t> new_positions(symbol_ref.positions.size() + 1);
      memcpy(new_positions.data(),
             symbol_ref.positions.data(),
             offset * sizeof(uint64_t));
      memcpy(new_positions.data() + offset + 1,
             symbol_ref.positions.data() + offset,
             (symbol_ref.positions.size() - offset - 1) * sizeof(uint64_t));
      symbol_ref.positions = std::move(new_positions);
    }
    symbol_ref.positions[offset] = pos;
  } else {
    symbol_ref.positions.push_back(pos);
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

void Trainer::ResetFreq(uint32_t sid, uint32_t left, uint32_t right, uint32_t best) {
  if (left == ~0u || right == ~0u) return;
  auto symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != ~0u && symbol != best) {
    allocated_[symbol].freq = 0;
  }
}

void Trainer::UpdateActiveSymbols(ThreadPool *pool) {
  std::vector<uint32_t> symbols;
  symbols.reserve(symbols_cache_.size());
  for (auto &it : symbols_cache_) {
    if (allocated_[it.second].IsBigram()) {
      symbols.push_back(it.second);
    }
  }
  pool->Loop(0, symbols.size(),
  [this, &symbols](const uint32_t beg, const uint32_t end) {
    for (uint32_t i = beg; i < end; i++) {
      ComputeFreq(&allocated_[symbols[i]]);
    }
  });
  pool->Wait();

  // At least kMinActiveSymbolsSize symbols must be in |active_symbols_|.
  constexpr int kMinActiveSymbolsSize = 1000;

  // Keeps top 5% frequent symbols.
  constexpr float kTopFrequentRatio = 0.05;
  const uint32_t size = std::min<uint32_t>(
      std::max<int>(kMinActiveSymbolsSize,
                    symbols_cache_.size() * kTopFrequentRatio),
      symbols.size());

  std::partial_sort(symbols.begin(), symbols.begin() + size, symbols.end(),
                    [this](uint32_t s1, uint32_t s2) {
                      return allocated_[s1].freq > allocated_[s2].freq;
                    });
  LOG(INFO) << "Updating active symbols. max_freq=" << allocated_[symbols[0]].freq
            << " min_freq=" << allocated_[symbols[size - 1]].freq;

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
    if (sentences_.size() % 10000000 == 0) {
      LOG(INFO) << "Loaded " << sentences_.size() << " sentences";
    }
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
  for (auto &p : required_chars_) {
    GetCharSymbol(p.first, false);
  }
  GetCharSymbol(kUNKChar, false);
  auto pool = absl::make_unique<ThreadPool>(trainer_spec_.num_threads());
  std::mutex sync;
  // Initializes symbols_. symbols_[sid][i] stores an unary symbol.
  ssize_t max_chunk_size = 60000000;
  std::atomic<int64> overflows = 0;
  symbols_.resize(sentences_.size());
  freqs_.resize(sentences_.size());
  LOG(INFO) << "Extracting single chars...";
  constexpr size_t uint16_max = static_cast<size_t>(std::numeric_limits<uint16_t>::max());
  for (ssize_t i = sentences_.size(); i >= 0; i -= max_chunk_size) {
    auto chunk_size = std::min(i, max_chunk_size);
    pool->Loop(i - chunk_size, i,
    [this, &overflows, &sync, uint16_max](const ssize_t beg, const ssize_t end) {
      for (ssize_t j = beg; j < end; j++) {
        auto &sentence = sentences_[j];
        auto &&text = string_util::UTF8ToUnicodeText(sentence.first);
        auto &symbols_sentence = symbols_[j];
        freqs_[j] = sentence.second;
        symbols_sentence.reserve(std::min(text.size(), uint16_max));
        for (const char32 c : text) {
          if (symbols_sentence.size() == uint16_max) {
            // this sentence is too long, must split to be able to call EncodePos
            overflows++;
            break;
          }
          symbols_sentence.push_back(GetCharSymbol(c, true));
        }
      }
    });
    pool->Wait();
    LOG(INFO) << "Materialized " << (symbols_.size() - i + chunk_size) << " / "
              << symbols_.size() << " symbols";
    sentences_.resize(i - chunk_size);
    sentences_.shrink_to_fit();
    #ifdef TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
    #endif
  }

  cache_file.reset();
  #ifdef TCMALLOC
  MallocExtension::instance()->ReleaseFreeMemory();
  #endif
  malloc_stats();

  uint32_t unisize = allocated_.size();
  LOG(INFO) << "Allocated " << unisize << " chars with "
            << overflows.load() << " overflows";

  // Makes all bigram symbols.
  for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    if (sid % 10000000 == 0 && sid > 0) {
      LOG(INFO) << "Generated pairs from " << sid << " symbols";
    }
    auto &sentence = symbols_[sid];
    for (size_t i = 1; i < sentence.size(); ++i) {
      uint32_t left = i - 1, right = i;
      auto symbol = GetPairSymbol(sentence[left], sentence[right]);
      if (symbol != ~0u) {
        allocated_[symbol].positions.push_back(EncodePos(sid, left, right));
      }
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

  for (uint32_t i = unisize; i < allocated_.size(); i++) {
    active_symbols_.insert(i);
  }

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  CHECK_GE_OR_RETURN(vocab_size, 0);

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<absl::string_view> dup;

  // Main loop.
  CHECK_OR_RETURN(final_pieces_.empty());
  LOG(INFO) << "Will do " << vocab_size << " iterations";
  while (final_pieces_.size() < static_cast<size_t>(vocab_size)) {
    constexpr int kUpdateActiveSymbolsInterval = 100;
    if (final_pieces_.size() % kUpdateActiveSymbolsInterval == 0) {
      UpdateActiveSymbols(pool.get());
    }

    // Scanning active symbols, finds the best_symbol with highest freq.
    uint32_t best_symbol = ~0u;
    {
      for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
        pool->Schedule([&, n]() {
          uint32_t my_best_symbol = ~0u;
          size_t pos = 0;
          for (auto &it : active_symbols_) {
            if ((n + pos++) % trainer_spec_.num_threads()) {
              continue;
            }
            auto &symbol = allocated_[it];
            ComputeFreq(&symbol);
            // If the frequency is the same, take shorter symbol.
            // if the length is the same, use lexicographical comparison
            bool update = my_best_symbol == ~0u;
            if (!update) {
              auto &my_best_symbol_ref = allocated_[my_best_symbol];
              update = (symbol.freq > my_best_symbol_ref.freq ||
                 (symbol.freq == my_best_symbol_ref.freq &&
                  (symbol.CharsSize() < my_best_symbol_ref.CharsSize() ||
                   (symbol.CharsSize() == my_best_symbol_ref.CharsSize() &&
                    symbol.ToString() < my_best_symbol_ref.ToString()))));
            }
            if (update) {
              my_best_symbol = it;
            }
          }
          std::lock_guard lock(sync);
          bool update = best_symbol == ~0u;
          if (!update) {
            auto &best_symbol_ref = allocated_[best_symbol];
            auto &my_best_symbol_ref = allocated_[my_best_symbol];
            update = (my_best_symbol_ref.freq > best_symbol_ref.freq ||
               (my_best_symbol_ref.freq == best_symbol_ref.freq &&
                (my_best_symbol_ref.CharsSize() < best_symbol_ref.CharsSize() ||
                 (my_best_symbol_ref.CharsSize() == best_symbol_ref.CharsSize() &&
                  my_best_symbol_ref.ToString() < best_symbol_ref.ToString()))));
          }
          if (update) {
            best_symbol = my_best_symbol;
          }
        });
      }
      pool->Wait();
    }

    if (best_symbol == ~0u) {
      LOG(WARNING) << "No valid symbol found";
      break;
    }
    auto &best_symbol_ref = allocated_[best_symbol];

    if (!dup.emplace(best_symbol_ref.ToString()).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol_ref.fp);
      active_symbols_.erase(best_symbol);
      continue;
    }

    // Stores the best_symbol in the final output.
    final_pieces_.emplace_back(best_symbol_ref.ToString(),
                               -static_cast<float>(final_pieces_.size()));

    if (final_pieces_.size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << best_symbol_ref.freq
                << " size=" << final_pieces_.size()
                << " all=" << symbols_cache_.size()
                << " active=" << active_symbols_.size()
                << " piece=" << best_symbol_ref.ToString()
                << " (" << best_symbol_ref.CharsSize()
                << ", " << best_symbol_ref.positions.size() << ")";
    }

    // Add new bigrams which are created after symbol replacement.
    // We do not need to scan all characters, but scan the neighbors in
    // best_symbol.
    for (uint64_t encoded_pos : best_symbol_ref.positions) {
      const Position pos = DecodePos(encoded_pos);

      if (symbols_[pos.sid][pos.left] == ~0u) {
        // left index might be NULL (set in the previous iteration)
        // when left_symbol == right_symbol.
        continue;
      }

      CHECK_OR_RETURN(symbols_[pos.sid][pos.right] != ~0u);

      // We have three bigrams [prev, left], [left, right], [right, next],
      // which are affected with this symbol replacement.
      const uint32_t next = GetNextIndex(pos.sid, pos.right);
      const uint32_t prev = GetPrevIndex(pos.sid, pos.left);

      // Resets the frequencies of bigrams [prev, left] and [right, next].
      ResetFreq(pos.sid, prev, pos.left, best_symbol);
      ResetFreq(pos.sid, pos.right, next, best_symbol);

      // Merges two symbols.
      symbols_[pos.sid][pos.left] = best_symbol;
      symbols_[pos.sid][pos.right] = ~0u;

      // Makes new symbol bigrams [prev, left] and [left, next].
      AddNewPair(pos.sid, prev, pos.left);
      AddNewPair(pos.sid, pos.left, next);
    }

    // Removes best_symbol so it is not selected again.
    symbols_cache_.erase(best_symbol_ref.fp);
    active_symbols_.erase(best_symbol);
  }  // end of main loop

  // Adds required_chars_
  for (const auto &w : Sorted(required_chars_, trainer_spec_.num_threads())) {
    const Symbol &symbol = allocated_[GetCharSymbol(w.first, false)];
    final_pieces_.emplace_back(symbol.ToString(),
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
