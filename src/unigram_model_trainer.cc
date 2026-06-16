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

#include "unigram_model_trainer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "normalizer.h"
#include "pretokenizer_for_training.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/esaxx/esa.hxx"  // Suffix array library.
#include "trainer_interface.h"
#include "unicode_script.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

constexpr char32_t kSentenceBoundary = 0x0000;

double Digamma(double x) {
  double result = 0.0;
  for (; x < 7; ++x) result -= 1 / x;
  x -= 1.0 / 2.0;
  const double xx = 1.0 / x;
  const double xx2 = xx * xx;
  const double xx4 = xx2 * xx2;
  result += std::log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
            (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
  return result;
}

template <typename IT>
void ToLogProb(IT begin, IT end) {
  float sum = 0.0;
  for (auto it = begin; it != end; ++it) {
    sum += it->second;
  }
  float logsum = std::log(static_cast<double>(sum));
  for (auto it = begin; it != end; ++it) {
    it->second = std::log(static_cast<double>(it->second)) - logsum;
  }
}

class BoundedPriorityQueue {
 public:
  explicit BoundedPriorityQueue(size_t size) : max_capacity_(size) {}

  void Push(const std::string& key, int64_t score) {
    auto& current_score = data_[key];  // initializes with 0 if not exists.
    if (score > current_score) {
      current_score = score;
    }
    // Run GC when the size exceeds 8 * max_capacity_
    if (data_.size() > 8 * max_capacity_) {
      Gc();
    }
  }

  std::vector<std::pair<std::string, int64_t>> Get() {
    std::vector<std::pair<std::string, int64_t>> results;
    results.reserve(data_.size());
    for (auto& it : data_) results.emplace_back(std::move(it));
    data_.clear();

    Sort(results);

    if (results.size() > max_capacity_) {
      results.resize(max_capacity_);
    }

    return results;
  }

 private:
  void Gc() {
    LOG(INFO) << "Running GC to shrink the candidate pieces";
    std::vector<std::pair<std::string, int64_t>> tmp;
    tmp.reserve(data_.size());
    for (auto& it : data_) tmp.emplace_back(std::move(it));
    Sort(tmp);
    data_.clear();

    const size_t keep = std::min(tmp.size(), max_capacity_);
    for (size_t i = 0; i < keep; ++i) {
      data_.emplace(std::move(tmp[i].first), tmp[i].second);
    }
  }

  void Sort(std::vector<std::pair<std::string, int64_t>>& agenda) {
    std::sort(
        agenda.begin(), agenda.end(), [](const auto& lhs, const auto& rhs) {
          // Sort by score, length, and dictionary order.
          return std::forward_as_tuple(rhs.second, rhs.first.size(),
                                       lhs.first) <
                 std::forward_as_tuple(lhs.second, lhs.first.size(), rhs.first);
        });
  }

  size_t max_capacity_;
  absl::flat_hash_map<std::string, int64_t> data_;
};
}  // namespace

TrainerModel::TrainerModel(const TrainerSpec& trainer_spec,
                           const NormalizerSpec& normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {}

TrainerModel::~TrainerModel() {}

const TrainerModel::SentencePieces& TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

absl::Status TrainerModel::SetSentencePieces(SentencePieces&& sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);

  min_score_ = FLT_MAX;
  model_proto_data_.Clear();
  model_proto_ = &model_proto_data_;
  std::vector<std::pair<absl::string_view, int>> pieces;

  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const absl::string_view w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;         // score.
    RET_CHECK(!std::isnan(score));
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
    auto* piece = model_proto_data_.add_pieces();
    piece->set_piece(w.data(), w.size());
    piece->set_score(score);
  }

  BuildTrie(&pieces);
  return status();
}

TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() {
  return trainer_spec_.train_extremely_large_corpus()
             ? MakeSeedSentencePiecesInternal<int64_t>()
             : MakeSeedSentencePiecesInternal<int32_t>();
}

// Returns seed sentencepieces for EM training.
template <typename node_int_type>
TrainerModel::SentencePieces Trainer::MakeSeedSentencePiecesInternal() {
  if (sentences_.empty() || required_chars_.empty()) {
    return {};
  }

  // Pretokenizer applied only in training time.
  // Pretokenizer is used as a constraint of piece extractions.
  const auto* pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();

  auto pretokenize_or_rewrite = [&](std::pair<std::string, int64_t>* w) {
    if (pretokenizer) {
      std::vector<char32_t> chars;
      for (const auto& w : pretokenizer->PreTokenize(w->first)) {
        for (const auto& c : string_util::UTF8ToUnicodeText(w)) {
          chars.push_back(c);
        }
        chars.push_back(kSentenceBoundary);
      }
      return chars;
    } else if (!trainer_spec_.pretokenization_delimiter().empty()) {
      // When delimiter is specified, tokenize the input with the delimiter.
      // For EM training, we assume that the delimiter doesn't exist and
      // rewrite the original sentence.
      std::vector<char32_t> chars;
      absl::string_view delimiter = trainer_spec_.pretokenization_delimiter();
      for (const auto& w : absl::StrSplit(w->first, delimiter)) {
        for (const auto& c : string_util::UTF8ToUnicodeText(w)) {
          chars.push_back(c);
        }
        chars.push_back(kSentenceBoundary);
      }
      // Removes the delimiter.
      w->first = absl::StrReplaceAll(w->first, {{delimiter, ""}});
      return chars;
    }
    return string_util::UTF8ToUnicodeText(w->first);
  };

  // Merges all sentences into one array with 0x0000 delimiter.
  std::vector<char32_t> array;
  absl::flat_hash_map<std::string, int64_t> all_chars;

  const bool is_tsv = trainer_spec_.input_format() == "tsv";

  for (auto& w : sentences_) {
    const auto ut = pretokenize_or_rewrite(&w);
    for (const auto& c : ut) {
      array.push_back(c);
      if (c != kUNKChar && c != kSentenceBoundary) {
        all_chars[string_util::UnicodeCharToUTF8(c)] += w.second;
      }
    }
    array.push_back(kSentenceBoundary);  // sentence boundary marker.

    // Naive workaround to over-sample the input.
    // In TSV mode, the frequency field is not used to extract the seed piece.
    // we can at least extract all pieces by copying the input because
    // the occurrence gets at least larger than or equals to 2.
    if (is_tsv) {
      for (const auto& c : ut) array.push_back(c);
      array.push_back(kSentenceBoundary);
    }
  }

  // all_chars must be included in the seed sentencepieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  for (const auto& it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(it);
  }

  if (!trainer_spec_.seed_sentencepieces_file().empty()) {
    auto seed_sentencepieces_file = sentencepiece::filesystem::NewReadableFile(
        trainer_spec_.seed_sentencepieces_file());
    std::string line;
    int64_t freq = 1;
    int skipped_sentencepieces = 0;
    while (seed_sentencepieces_file->ReadLine(&line)) {
      const std::vector<std::string> fields = absl::StrSplit(line, '\t');
      if (fields.size() < 2) {
        LOG(ERROR) << "Format error: must be <piece> <tab> <freq>";
        return {};
      }
      const auto& seed_sentencepiece = fields[0];
      if (!absl::SimpleAtoi(fields[1], &freq)) {
        LOG(ERROR) << "Could not parse the frequency; line: " << line;
        return {};
      }
      const UnicodeText uw = string_util::UTF8ToUnicodeText(seed_sentencepiece);
      if (!IsValidSentencePiece(uw)) {
        ++skipped_sentencepieces;
        continue;
      }
      // Initialise score of a piece by character coverage.
      seed_sentencepieces.emplace_back(seed_sentencepiece, freq * uw.size());
      if (seed_sentencepieces.size() % 1000000 == 0) {
        LOG(INFO) << "loaded " << seed_sentencepieces.size()
                  << " seed sentencepieces";
      }
    }

    LOG(INFO) << "skipped " << skipped_sentencepieces << " seed sentencepieces";

    // Take highest scoring pieces as initial vocab.
    seed_sentencepieces = Sorted(seed_sentencepieces);
    seed_sentencepieces.resize(std::min<size_t>(
        trainer_spec_.seed_sentencepiece_size(), seed_sentencepieces.size()));

    LOG(INFO) << "Initialized " << seed_sentencepieces.size()
              << " seed sentencepieces from file.";
  } else {
    CHECK_LE(array.size(),
             static_cast<size_t>(std::numeric_limits<node_int_type>::max()))
        << "Input corpus too large, try with train_extremely_large_corpus=true";
    const node_int_type n = array.size();

    std::vector<node_int_type> SA(n);  // suffix array
    std::vector<node_int_type> L(n);   // left boundaries of internal node
    std::vector<node_int_type> R(n);   // right boundaries of internal node
    std::vector<node_int_type> D(n);   // depths of internal node

    // Makes a suffix array to extract all sub strings occurring
    // more than 2 times in the sentence.
    constexpr node_int_type kAlphabetSize = 0x110000;  // All UCS4 range.
    node_int_type node_num = 0;
    LOG(INFO) << "Making suffix array...";
    CHECK_EQ(0, esaxx(array.begin(), SA.begin(), L.begin(), R.begin(),
                      D.begin(), n, kAlphabetSize, node_num));

    LOG(INFO) << "Extracting frequent sub strings... node_num=" << node_num;

    BoundedPriorityQueue queue(
        static_cast<size_t>(trainer_spec_.seed_sentencepiece_size()));

    // split candidate frequent piece into the actual piece.
    auto split_into_pieces =
        [&](absl::string_view w) -> std::vector<absl::string_view> {
      if (trainer_spec_.split_by_whitespace()) {
        return SplitIntoWords(w, trainer_spec_.treat_whitespace_as_suffix(),
                              trainer_spec_.allow_whitespace_only_pieces());
      }
      return {w};
    };

    for (node_int_type i = 0; i < node_num; ++i) {
      const node_int_type offset = SA[L[i]];
      const node_int_type len = D[i];
      if (len <= 1 || offset >= array.size() || offset + len >= array.size()) {
        continue;
      }
      const char32_t* begin = &array[offset];
      const char32_t* end = &array[offset + len];
      const uint64_t freq = R[i] - L[i];

      // Split by kSentenceBoundary, as some frequent phrases may cross
      // the sentence boundary.
      while (begin < end) {
        const char32_t* delim = std::find(begin, end, kSentenceBoundary);
        const UnicodeText uw(begin, delim);
        begin = delim + 1;
        if (uw.size() <= 1) continue;

        const std::string w = string_util::UnicodeTextToUTF8(uw);
        for (absl::string_view piece : split_into_pieces(w)) {
          const UnicodeText pw = string_util::UTF8ToUnicodeText(piece);
          if (pw.size() <= 1 || !IsValidSentencePiece(pw)) {
            continue;
          }
          const uint64_t score = freq * pw.size();
          queue.Push(std::string(piece), score);
        }
      }
    }

    for (auto& [w, score] : queue.Get()) {
      CHECK(!port::ContainsKey(all_chars, w));
      seed_sentencepieces.emplace_back(std::move(w), score);
    }
  }

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces";

  return seed_sentencepieces;
}

std::vector<float> Trainer::RunEStep(const TrainerModel& model, float* obj,
                                     int64_t* num_tokens) const {
  std::vector<std::vector<float>> expected(trainer_spec_.num_threads());
  std::vector<float> objs(trainer_spec_.num_threads(), 0.0);
  std::vector<int64_t> ntokens(trainer_spec_.num_threads(), 0.0);

  auto pool = std::make_unique<ThreadPool>(trainer_spec_.num_threads());

  int64_t all_sentence_freq = 0;
  for (const auto& w : sentences_) {
    all_sentence_freq += w.second;
  }

  // Executes E step in parallel
  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      Lattice lattice;
      expected[n].resize(model.GetPieceSize(), 0.0);
      for (size_t i = n; i < sentences_.size();
           i += trainer_spec_.num_threads()) {
        const std::string& w = sentences_[i].first;
        const int64_t freq = sentences_[i].second;
        lattice.SetSentence(w);
        model.PopulateNodes(&lattice);
        const float Z = lattice.PopulateMarginal(freq, &expected[n]);
        ntokens[n] += lattice.Viterbi().first.size() * freq;
        CHECK(!std::isnan(Z))
            << "likelihood is NAN. Input sentence may be too long";
        objs[n] -= Z / all_sentence_freq;
      }
    });
  }
  pool.reset(nullptr);

  // Merges expectations
  for (int n = 1; n < trainer_spec_.num_threads(); ++n) {
    objs[0] += objs[n];
    ntokens[0] += ntokens[n];
    for (size_t k = 0; k < expected[0].size(); ++k) {
      expected[0][k] += expected[n][k];
    }
  }

  *obj = objs[0];
  *num_tokens = ntokens[0];
  CHECK(!std::isnan(*obj));

  return expected[0];
}

TrainerModel::SentencePieces Trainer::RunMStep(
    const TrainerModel& model, const std::vector<float>& expected) const {
  const auto& sentencepieces = model.GetSentencePieces();
  CHECK_EQ(sentencepieces.size(), expected.size());
  TrainerModel::SentencePieces new_sentencepieces;

  float sum = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float freq = expected[i];

    // Filter infrequent sentencepieces here.
    constexpr float kExpectedFrequencyThreshold = 0.5;
    if (freq < kExpectedFrequencyThreshold) {
      continue;
    }

    new_sentencepieces.emplace_back(sentencepieces[i].first, freq);
    sum += freq;
  }

  // Here we do not use the original EM, but use the
  // Bayesianified/DPified EM algorithm.
  // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
  // This modification will act as a sparse prior.
  const float logsum = Digamma(sum);
  for (auto& w : new_sentencepieces) {
    w.second = Digamma(w.second) - logsum;
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::PruneSentencePieces(
    const TrainerModel& model) const {
  const auto& sentencepieces = model.GetSentencePieces();

  Lattice lattice;
  std::vector<bool> always_keep(sentencepieces.size(), true);
  std::vector<std::vector<int>> alternatives(sentencepieces.size());

  // First, segments the current sentencepieces to know
  // how each sentencepiece is resegmented if this sentencepiece is removed
  // from the vocabulary.
  // To do so, we take the second best segmentation of sentencepiece[i].
  // alternatives[i] stores the sequence of second best sentencepieces.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    const auto& w = sentencepieces[i];
    lattice.SetSentence(w.first);
    model.PopulateNodes(&lattice);
    const auto nbests = lattice.NBest(2, false, 0.0);
    if (nbests.size() == 1) {
      // No second-best result is found. always keep this sentencepiece.
      always_keep[i] = true;
      continue;
    } else if (nbests[0].first.size() >= 2) {
      // Can safely remove this sentencepiece if its Viterbi path is split.
      always_keep[i] = false;
    } else if (nbests[0].first.size() == 1) {
      always_keep[i] = true;
      for (const auto* node : nbests[1].first) {
        alternatives[i].push_back(node->id);
      }
    }
  }

  // Second, segment all sentences to compute token frequencies
  // with a unigram language model using the Viterbi path.
  std::vector<float> freq(sentencepieces.size(), 0.0);
  {
    std::vector<std::vector<float>> freqs(trainer_spec_.num_threads());

    auto pool = std::make_unique<ThreadPool>(trainer_spec_.num_threads());
    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      freqs[n].resize(sentencepieces.size(), 0.0);

      pool->Schedule([&, n]() {
        Lattice lattice;
        for (size_t i = n; i < sentences_.size();
             i += trainer_spec_.num_threads()) {
          const auto& w = sentences_[i];
          lattice.SetSentence(w.first);
          model.PopulateNodes(&lattice);
          for (const auto* node : lattice.Viterbi().first) {
            if (node->id >= 0) {
              freqs[n][node->id] += w.second;
            }
          }
        }
      });
    }
    pool.reset(nullptr);

    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      for (size_t i = 0; i < sentencepieces.size(); ++i) {
        freq[i] += freqs[n][i];
      }
    }
  }

  const float sum = std::accumulate(freq.begin(), freq.end(), 0.0);
  const float logsum = std::log(static_cast<double>(sum));
  std::vector<std::pair<int, float>> candidates;
  TrainerModel::SentencePieces new_sentencepieces;

  // Finally, computes how likely the LM likelihood is reduced if
  // the sentencepiece[i] is removed from the vocabulary.
  // Since the exact computation of loss is difficult, we compute the
  // loss approximately by assuming that all sentencepiece[i] in the sentences
  // are replaced with alternatives[i] when sentencepiece[i] is removed.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    if (freq[i] == 0 || !always_keep[i]) {
      // not found in Viterbi path. Can remove this entry safely.
      continue;
    } else if (alternatives[i].empty()) {
      // no alternatives. Keeps this entry.
      new_sentencepieces.push_back(sentencepieces[i]);
    } else {
      // The logprob with the sentencepiece[i].
      const float logprob_sp = std::log(static_cast<double>(freq[i])) - logsum;

      // After removing the sentencepiece[i], its frequency freq[i] is
      // re-assigned to alternatives.
      // new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
      //         = current_sum + freq[i] * (alternatives[i] - 1)
      const float logsum_alt = std::log(
          static_cast<double>(sum + freq[i] * (alternatives[i].size() - 1)));

      // The frequencies of altenatives are increased by freq[i].
      float logprob_alt = 0.0;
      for (const int n : alternatives[i]) {
        logprob_alt +=
            (std::log(static_cast<double>(freq[n] + freq[i])) - logsum_alt);
      }

      // loss: the diff of likelihood after removing the sentencepieces[i].
      float F = freq[i] / sum;  // normalized token frequency
      const float loss = F * (logprob_sp - logprob_alt);
      candidates.emplace_back(i, loss);
    }
  }

  const int pruned_size =
      std::max<int>(desired_vocab_size_,
                    trainer_spec_.shrinking_factor() * sentencepieces.size());

  // Keeps trainer_spec_.shrinking_factor * sentencepieces.size() pieces.
  // shrinking_factor is 0.75 by default.
  for (const auto& w : Sorted(candidates)) {
    if (new_sentencepieces.size() == static_cast<size_t>(pruned_size)) {
      break;
    }
    new_sentencepieces.emplace_back(sentencepieces[w.first]);
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel& model) const {
  const auto& sentencepieces = model.GetSentencePieces();
  absl::flat_hash_map<std::string, float> final_sentencepieces;
  absl::flat_hash_map<std::string, float> sp(sentencepieces.begin(),
                                             sentencepieces.end());

  // required_chars_ must be included in the final sentencepieces.
  float min_score_penalty = 0.0;
  constexpr float kMinScorePenaltyDelta = 0.0001;
  for (const auto& w : Sorted(required_chars_)) {
    const std::string s = string_util::UnicodeCharToUTF8(w.first);
    if (port::ContainsKey(sp, s)) {
      final_sentencepieces[s] = sp[s];
    } else {
      // Add penalty to avoid required pieces from having the same score.
      // Since the required_chars_ is sorted, frequent pieces have
      // less penalties.
      final_sentencepieces[s] = model.min_score() + min_score_penalty;
      min_score_penalty += kMinScorePenaltyDelta;
    }
  }

  const int vocab_size_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  CHECK_GT(vocab_size_size, 0);

  // Then keeps sentencepieces with higher scores.
  for (const auto& w : Sorted(sentencepieces)) {
    if (port::ContainsKey(final_sentencepieces, w.first)) {
      continue;
    }
    if (static_cast<size_t>(vocab_size_size) == final_sentencepieces.size()) {
      break;
    }
    final_sentencepieces[w.first] = w.second;
  }

  return Sorted(final_sentencepieces);
}

absl::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  RET_CHECK_EQ(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  RET_CHECK(normalizer_spec_.escape_whitespaces());

  TrainerModel model(trainer_spec_, normalizer_spec_);

  RETURN_IF_ERROR(model.status());
  RETURN_IF_ERROR(LoadSentences());
  RET_CHECK(!required_chars_.empty());

  auto seed_sentencepieces = MakeSeedSentencePieces();
  RET_CHECK(!seed_sentencepieces.empty());

  RETURN_IF_ERROR(model.SetSentencePieces(std::move(seed_sentencepieces)));

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

  desired_vocab_size_ = static_cast<size_t>(trainer_spec_.vocab_size() * 1.1);

  while (true) {
    // Sub-EM iteration.
    for (int iter = 0; iter < trainer_spec_.num_sub_iterations(); ++iter) {
      // Executes E step
      float objective = 0.0;
      int64_t num_tokens = 0;
      const auto expected = RunEStep(model, &objective, &num_tokens);

      // Executes M step.
      auto new_sentencepieces = RunMStep(model, expected);
      RETURN_IF_ERROR(model.SetSentencePieces(std::move(new_sentencepieces)));

      LOG(INFO) << "EM sub_iter=" << iter << " size=" << model.GetPieceSize()
                << " obj=" << objective << " num_tokens=" << num_tokens
                << " num_tokens/piece="
                << 1.0 * num_tokens / model.GetPieceSize();
    }  // end of Sub EM iteration

    // Stops the iteration when the size of sentences reaches to the
    // desired symbol size.
    if (model.GetPieceSize() <= desired_vocab_size_) {
      break;
    }

    // Prunes pieces.
    auto new_sentencepieces = PruneSentencePieces(model);
    RETURN_IF_ERROR(model.SetSentencePieces(std::move(new_sentencepieces)));
  }  // end of EM iteration

  // Finally, adjusts the size of sentencepices to be |vocab_size|.
  final_pieces_ = FinalizeSentencePieces(model);

  return Save();
}
}  // namespace unigram
}  // namespace sentencepiece
