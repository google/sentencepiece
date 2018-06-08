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
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "normalizer.h"
#include "third_party/esaxx/esa.hxx"  // Suffix array library.
#include "unicode_script.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

double Digamma(double x) {
  double result = 0.0;
  for (; x < 7; ++x) result -= 1 / x;
  x -= 1.0 / 2.0;
  const double xx = 1.0 / x;
  const double xx2 = xx * xx;
  const double xx4 = xx2 * xx2;
  result += log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
            (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
  return result;
}

template <typename IT>
void ToLogProb(IT begin, IT end) {
  float sum = 0.0;
  for (auto it = begin; it != end; ++it) {
    sum += it->second;
  }
  float logsum = log(sum);
  for (auto it = begin; it != end; ++it) {
    it->second = log(it->second) - logsum;
  }
}

class ThreadPool {
 public:
  ThreadPool() {}
  virtual ~ThreadPool() {
    for (auto &task : tasks_) {
      task.join();
    }
  }

  void Schedule(std::function<void()> closure) { tasks_.emplace_back(closure); }

 private:
  std::vector<std::thread> tasks_;
};
}  // namespace

TrainerModel::TrainerModel(const TrainerSpec &trainer_spec,
                           const NormalizerSpec &normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {}
TrainerModel::~TrainerModel() {}

const TrainerModel::SentencePieces &TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

void TrainerModel::SetSentencePieces(SentencePieces &&sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);
  CHECK(!sentencepieces_.empty());

  min_score_ = FLT_MAX;
  std::vector<std::pair<StringPiece, int>> pieces;
  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const StringPiece w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;   // score.
    CHECK(!std::isnan(score));
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
  }

  BuildTrie(&pieces);
  CHECK_OK(status());
}

// Returns seed sentencepieces for EM training.
TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() const {
  CHECK(!sentences_.empty());
  CHECK(!required_chars_.empty());

  // Merges all sentences into one array with 0x0000 delimiter.
  std::vector<char32> array;
  std::unordered_map<std::string, int64> all_chars;
  constexpr char32 kSentenceBoundary = 0x0000;

  const size_t mining_size =
      std::min<size_t>(sentences_.size(), trainer_spec_.mining_sentence_size());
  LOG(INFO) << "Using " << mining_size
            << " sentences for making seed sentencepieces";

  std::vector<std::string> mining_sentences(mining_size);
  for (size_t i = 0; i < mining_size; ++i) {
    mining_sentences[i] = sentences_[i].first;
  }

  for (const auto &w : mining_sentences) {
    for (const auto &c : string_util::UTF8ToUnicodeText(w)) {
      array.push_back(c);
      if (c != kUNKChar && c != kSentenceBoundary) {
        ++all_chars[string_util::UnicodeCharToUTF8(c)];
      }
    }
    array.push_back(kSentenceBoundary);  // sentence boundary marker.
  }

  const int n = array.size();
  std::vector<int> SA(n);  // suffix array
  std::vector<int> L(n);   // left boundaries of internal node
  std::vector<int> R(n);   // right boundaries of internal node
  std::vector<int> D(n);   // depths of internal node

  // Makes a suffix array to extract all sub strings occurring
  // more than 2 times in the sentence.
  constexpr int kAlphabetSize = 0x110000;  // All UCS4 range.
  int node_num = 0;
  LOG(INFO) << "Making suffix array...";
  CHECK_EQ(0, esaxx(array.begin(), SA.begin(), L.begin(), R.begin(), D.begin(),
                    n, kAlphabetSize, node_num));

  LOG(INFO) << "Extracting frequent sub strings...";
  std::vector<std::pair<int, int>> substr_index;
  for (int i = 0; i < node_num; ++i) {
    const int offset = SA[L[i]];
    const int len = D[i];
    if (len <= 1) {
      continue;
    }
    const char32 *begin = &array[0] + offset;
    const char32 *end = &array[0] + offset + len;
    // Skips if a substring contains a sentence boundary.
    if (std::find(begin, end, kSentenceBoundary) != end) {
      continue;
    }
    const UnicodeText uw(begin, end);
    if (!IsValidSentencePiece(uw)) {
      continue;
    }

    // character-wise coverage is the default score.
    const int freq = R[i] - L[i];
    const int score = freq * len;
    substr_index.emplace_back(i, score);
  }

  // all_chars must be included in the seed sentencepieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  for (const auto &it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(it);
  }

  // Sort by the coverage of sub strings.
  for (const auto &p : Sorted(substr_index)) {
    const int offset = SA[L[p.first]];
    const int len = D[p.first];
    CHECK_GT(len, 0);
    const char32 *begin = &array[offset];
    const char32 *end = &array[offset + len];
    const UnicodeText uw(begin, end);
    CHECK(IsValidSentencePiece(uw));  // just in case.
    const std::string w = string_util::UnicodeTextToUTF8(uw);
    if (seed_sentencepieces.size() ==
        static_cast<size_t>(trainer_spec_.seed_sentencepiece_size())) {
      break;
    }
    CHECK(!port::ContainsKey(all_chars, w));
    seed_sentencepieces.emplace_back(w, p.second);
  }

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces";

  return seed_sentencepieces;
}

std::vector<float> Trainer::RunEStep(const TrainerModel &model, float *obj,
                                     int64 *num_tokens) const {
  std::vector<std::vector<float>> expected(trainer_spec_.num_threads());
  std::vector<float> objs(trainer_spec_.num_threads(), 0.0);
  std::vector<int64> ntokens(trainer_spec_.num_threads(), 0.0);

  auto pool = port::MakeUnique<ThreadPool>();

  int64 all_sentence_freq = 0;
  for (const auto &w : sentences_) {
    all_sentence_freq += w.second;
  }

  // Executes E step in parallel
  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      Lattice lattice;
      expected[n].resize(model.GetPieceSize(), 0.0);
      for (size_t i = n; i < sentences_.size();
           i += trainer_spec_.num_threads()) {
        const std::string &w = sentences_[i].first;
        const int64 freq = sentences_[i].second;
        lattice.SetSentence(w);
        model.PopulateNodes(&lattice);
        const float Z = lattice.PopulateMarginal(freq, &expected[n]);
        ntokens[n] += lattice.Viterbi().size();
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
    const TrainerModel &model, const std::vector<float> &expected) const {
  const auto &sentencepieces = model.GetSentencePieces();
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
  for (auto &w : new_sentencepieces) {
    w.second = Digamma(w.second) - logsum;
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::PruneSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();

  Lattice lattice;
  std::vector<bool> always_keep(sentencepieces.size(), true);
  std::vector<std::vector<int>> alternatives(sentencepieces.size());

  // First, segments the current sentencepieces to know
  // how each sentencepiece is resegmented if this sentencepiece is removed
  // from the vocabulary.
  // To do so, we take the second best segmentation of sentencepiece[i].
  // alternatives[i] stores the sequence of second best sentencepieces.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    const auto &w = sentencepieces[i];
    lattice.SetSentence(w.first);
    model.PopulateNodes(&lattice);
    const auto nbests = lattice.NBest(2);
    if (nbests.size() == 1) {
      // No second-best result is found. always keep this sentencepiece.
      always_keep[i] = true;
      continue;
    } else if (nbests[0].size() >= 2) {
      // Can safely remove this sentencepiece if its Viterbi path is split.
      always_keep[i] = false;
    } else if (nbests[0].size() == 1) {
      always_keep[i] = true;
      for (const auto *node : nbests[1]) {
        alternatives[i].push_back(node->id);
      }
    }
  }

  // Second, segments all sentences to compute likelihood
  // with a unigram language model. inverted[i] stores
  // the set of sentence index where the sentencepieces[i] appears.
  float vsum = 0.0;
  std::vector<float> freq(sentencepieces.size(), 0.0);
  std::vector<std::vector<int>> inverted(sentencepieces.size());
  {
    std::vector<float> vsums(trainer_spec_.num_threads(), 0.0);
    std::vector<std::vector<float>> freqs(trainer_spec_.num_threads());
    std::vector<std::vector<std::vector<int>>> inverteds(
        trainer_spec_.num_threads());

    auto pool = port::MakeUnique<ThreadPool>();
    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      freqs[n].resize(sentencepieces.size(), 0.0);
      inverteds[n].resize(sentencepieces.size());

      pool->Schedule([&, n]() {
        Lattice lattice;
        for (size_t i = n; i < sentences_.size();
             i += trainer_spec_.num_threads()) {
          const auto &w = sentences_[i];
          lattice.SetSentence(w.first);
          model.PopulateNodes(&lattice);
          vsums[n] += w.second;
          for (const auto *node : lattice.Viterbi()) {
            if (node->id >= 0) {
              freqs[n][node->id] += w.second;
              inverteds[n][node->id].push_back(i);
            }
          }
        }
      });
    }
    pool.reset(nullptr);

    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      vsum += vsums[n];
      for (size_t i = 0; i < sentencepieces.size(); ++i) {
        freq[i] += freqs[n][i];
        std::copy(inverteds[n][i].begin(), inverteds[n][i].end(),
                  std::back_inserter(inverted[i]));
      }
    }
  }

  const float sum = std::accumulate(freq.begin(), freq.end(), 0.0);
  const float logsum = log(sum);
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
      float F = 0.0;  // the frequency of sentencepieces[i].
      for (const int n : inverted[i]) {
        F += sentences_[n].second;
      }
      F /= vsum;  // normalizes by all sentence frequency.

      // The logprob with the sentencepiece[i].
      const float logprob_sp = log(freq[i]) - logsum;

      // After removing the sentencepiece[i], its frequency freq[i] is
      // re-assigned to alternatives.
      // new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
      //         = current_sum + freq[i] (alternatives - 1)
      const float logsum_alt = log(sum + freq[i] * (alternatives.size() - 1));

      // The frequencies of altenatives are increased by freq[i].
      float logprob_alt = 0.0;
      for (const int n : alternatives[i]) {
        logprob_alt += (log(freq[n] + freq[i]) - logsum_alt);
      }

      // loss: the diff of likelihood after removing the sentencepieces[i].
      const float loss = F * (logprob_sp - logprob_alt);
      candidates.emplace_back(i, loss);
    }
  }

  const int pruned_size =
      std::max<int>(desired_vocab_size_,
                    trainer_spec_.shrinking_factor() * sentencepieces.size());

  // Keeps trainer_spec_.shrinking_factor * sentencepieces.size() pieces.
  // shrinking_factor is 0.75 by default.
  for (const auto &w : Sorted(candidates)) {
    if (new_sentencepieces.size() == static_cast<size_t>(pruned_size)) {
      break;
    }
    new_sentencepieces.emplace_back(sentencepieces[w.first]);
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();
  std::unordered_map<std::string, float> final_sentencepieces;
  std::unordered_map<std::string, float> sp(sentencepieces.begin(),
                                            sentencepieces.end());

  // required_chars_ must be included in the final sentencepieces.
  float min_score_penalty = 0.0;
  constexpr float kMinScorePenaltyDelta = 0.0001;
  for (const auto &w : Sorted(required_chars_)) {
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
  for (const auto &w : Sorted(sentencepieces)) {
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

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  LOG(INFO) << "Starts training with : \n" << trainer_spec_.Utf8DebugString();

  CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());

  TrainerModel model(trainer_spec_, normalizer_spec_);

  RETURN_IF_ERROR(model.status());
  RETURN_IF_ERROR(LoadSentences());

  auto seed_sentencepieces = MakeSeedSentencePieces();
  model.SetSentencePieces(std::move(seed_sentencepieces));

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  } else {
    const int training_size = std::min<size_t>(
        sentences_.size(), trainer_spec_.training_sentence_size());
    sentences_.resize(training_size);
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

  desired_vocab_size_ = static_cast<size_t>(trainer_spec_.vocab_size() * 1.1);

  while (true) {
    // Sub-EM iteration.
    for (int iter = 0; iter < trainer_spec_.num_sub_iterations(); ++iter) {
      // Executes E step
      float objective = 0.0;
      int64 num_tokens = 0;
      const auto expected = RunEStep(model, &objective, &num_tokens);

      // Executes M step.
      auto new_sentencepieces = RunMStep(model, expected);
      model.SetSentencePieces(std::move(new_sentencepieces));

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
    model.SetSentencePieces(std::move(new_sentencepieces));
  }  // end of EM iteration

  // Finally, adjusts the size of sentencepices to be |vocab_size|.
  final_pieces_ = FinalizeSentencePieces(model);

  return Save();
}
}  // namespace unigram
}  // namespace sentencepiece
