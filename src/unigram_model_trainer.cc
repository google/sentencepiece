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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "filesystem.h"
#include "libsais.h"
#include "ret_check.h"
#include "sentencepiece_trainer.h"
#include "trainer_interface.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

template <typename IT>
void ToLogProb(IT begin, IT end) {
  double sum = 0.0;
  for (auto it = begin; it != end; ++it) {
    sum += it->second;
  }
  const double logsum = std::log(sum);
  for (auto it = begin; it != end; ++it) {
    it->second =
        static_cast<float>(std::log(static_cast<double>(it->second)) - logsum);
  }
}

class BoundedPriorityQueue {
 public:
  explicit BoundedPriorityQueue(size_t size)
      : max_capacity_(size), gc_keep_capacity_(8 * size) {}

  void Add(absl::string_view key_view, uint64_t freq_delta) {
    auto it = data_.find(key_view);
    if (it != data_.end()) {
      it->second += freq_delta;
    } else {
      data_.emplace(std::string(key_view), freq_delta);
    }
  }

  void MaybeGc() {
    if (data_.size() > 2 * gc_keep_capacity_) {
      Gc();
    }
  }

  const absl::flat_hash_map<std::string, uint64_t>& data() const {
    return data_;
  }

  std::vector<std::pair<std::string, float>> Get() {
    struct ItemScore {
      std::string* key;
      float score;
    };

    std::vector<ItemScore> items;
    items.reserve(data_.size());
    const float power = absl::GetFlag(FLAGS_seed_piece_length_power);

    for (auto& [key, freq] : data_) {
      if (freq < 2) continue;
      const float score = CalculatePieceScore(key, freq, power);
      items.push_back({const_cast<std::string*>(&key), score});
    }

    std::sort(
        items.begin(), items.end(),
        [](const ItemScore& lhs, const ItemScore& rhs) {
          return std::forward_as_tuple(rhs.score, rhs.key->size(), *lhs.key) <
                 std::forward_as_tuple(lhs.score, lhs.key->size(), *rhs.key);
        });

    std::vector<std::pair<std::string, float>> results;
    const size_t keep = std::min(items.size(), max_capacity_);
    results.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
      results.emplace_back(std::move(*items[i].key), items[i].score);
    }
    data_.clear();
    return results;
  }

 private:
  static float CalculateScore(uint64_t freq, float len, float power) {
    if (power == 1.0f) {
      return static_cast<float>(freq) * len;
    } else if (power == 0.5f) {
      return static_cast<float>(freq) * std::sqrt(len);
    } else if (power == 0.0f) {
      return static_cast<float>(freq);
    }
    return static_cast<float>(freq) * std::pow(len, power);
  }

  static float CalculatePieceScore(absl::string_view key, uint64_t freq,
                                   float power) {
    return CalculateScore(freq, static_cast<float>(string_util::UTF8Len(key)),
                          power);
  }

  void Gc() {
    LOG(INFO) << "Running Score-Based QuickSelect GC (current_size="
              << data_.size() << ", keeping top " << gc_keep_capacity_
              << " highest-scoring items)...";
    using MapPair = std::pair<const std::string, uint64_t>;
    struct GcItem {
      const MapPair* kv;
      float score;
    };
    const float power = absl::GetFlag(FLAGS_seed_piece_length_power);
    std::vector<GcItem> tmp;
    tmp.reserve(data_.size());
    for (const auto& kv : data_) {
      tmp.push_back({&kv, CalculatePieceScore(kv.first, kv.second, power)});
    }

    const size_t keep = std::min(tmp.size(), gc_keep_capacity_);
    std::nth_element(
        tmp.begin(), tmp.begin() + keep, tmp.end(),
        [](const GcItem& lhs, const GcItem& rhs) {
          return std::forward_as_tuple(rhs.score, rhs.kv->first.size(),
                                       lhs.kv->first) <
                 std::forward_as_tuple(lhs.score, lhs.kv->first.size(),
                                       rhs.kv->first);
        });

    absl::flat_hash_map<std::string, uint64_t> new_data;
    new_data.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
      auto& mutable_kv = const_cast<MapPair&>(*tmp[i].kv);
      new_data.emplace(std::move(mutable_kv.first), mutable_kv.second);
    }
    data_ = std::move(new_data);
  }

  size_t max_capacity_;
  size_t gc_keep_capacity_;
  absl::flat_hash_map<std::string, uint64_t> data_;
};

// Size of nodes pre-allocated in Lattice.
constexpr size_t kPreallocateLatticeNodeSize = 1024;
constexpr float kUnkPenalty = 10.0;

inline float LogSumExp(float x, float y, bool init_mode) {
  if (init_mode) {
    return y;
  }
  const float vmin = std::min(x, y);
  const float vmax = std::max(x, y);
  constexpr float kMinusLogEpsilon = 50;
  if (vmax > vmin + kMinusLogEpsilon) {
    return vmax;
  } else {
    return vmax + std::log1p(std::exp(static_cast<double>(vmin - vmax)));
  }
}

}  // namespace

Lattice::Lattice() {
  node_blocks_.push_back(std::make_unique<Node[]>(kPreallocateLatticeNodeSize));
}
Lattice::~Lattice() = default;

const std::vector<Lattice::Node*>& Lattice::begin_nodes(int pos) const {
  return begin_nodes_[pos];
}

int Lattice::size() const { return std::max<int>(0, surface_.size() - 1); }

int Lattice::utf8_size() const { return sentence_.size(); }

const char* Lattice::sentence() const { return sentence_.data(); }

const char* Lattice::surface(int pos) const { return surface_[pos]; }

Lattice::Node* Lattice::NewNode() {
  const size_t block_idx = node_size_ / kPreallocateLatticeNodeSize;
  const size_t elem_idx = node_size_ % kPreallocateLatticeNodeSize;
  if (block_idx == node_blocks_.size()) {
    node_blocks_.push_back(
        std::make_unique<Node[]>(kPreallocateLatticeNodeSize));
  }
  Node* node = &node_blocks_[block_idx][elem_idx];
  *node = Node{};
  node->node_id = node_size_++;
  return node;
}

void Lattice::Clear() {
  begin_nodes_.clear();
  end_nodes_.clear();
  sentence_ = absl::string_view("");
  surface_.clear();
  node_size_ = 0;
}

void Lattice::SetSentence(absl::string_view sentence) {
  Clear();

  sentence_ = sentence;
  surface_.reserve(sentence.size() + 1);

  while (!sentence.empty()) {
    const int mblen = std::min<int>(string_util::OneCharLen(sentence.data()),
                                    sentence.size());
    surface_.push_back(sentence.data());
    sentence.remove_prefix(mblen);
  }
  surface_.push_back(sentence.data());

  const size_t len = size();
  begin_nodes_.resize(len + 1);
  end_nodes_.resize(len + 1);

  constexpr size_t kReservedNodeSize = 16;
  for (size_t i = 0; i <= len; ++i) {
    begin_nodes_[i].reserve(kReservedNodeSize);
    end_nodes_[i].reserve(kReservedNodeSize);
  }

  Node* bos = NewNode();
  bos->id = -1;
  bos->pos = 0;
  end_nodes_[0].push_back(bos);

  Node* eos = NewNode();
  eos->id = -1;
  eos->pos = len;
  begin_nodes_[len].push_back(eos);
}

Lattice::Node* Lattice::Insert(int pos, int length) {
  const int len = size();
  if (pos < 0 || length <= 0 || pos + length > len) {
    LOG(ERROR) << "Invalid node position or length: pos=" << pos
               << " length=" << length << " size=" << len;
    return nullptr;
  }
  Node* node = NewNode();
  node->pos = pos;
  node->length = length;
  begin_nodes_[pos].push_back(node);
  end_nodes_[pos + node->length].push_back(node);

  return node;
}

Lattice::LatticePathWithScore Lattice::Viterbi() {
  const size_t len = size();
  if (len == 0) return {};

  for (size_t pos = 0; pos <= len; ++pos) {
    for (Node* rnode : begin_nodes_[pos]) {
      rnode->prev = nullptr;
      float best_score = 0.0;
      Node* best_node = nullptr;
      for (Node* lnode : end_nodes_[pos]) {
        const float score = lnode->backtrace_score + rnode->score;
        if (best_node == nullptr || score > best_score) {
          best_node = lnode;
          best_score = score;
        }
      }
      if (best_node == nullptr) {
        LOG(ERROR) << "Failed to find the best path in Viterbi.";
        return {};
      }
      rnode->prev = best_node;
      rnode->backtrace_score = best_score;
    }
  }

  std::vector<Node*> results;
  if (begin_nodes_[len].empty() || begin_nodes_[len][0] == nullptr) {
    LOG(ERROR) << "EOS node is missing in Viterbi.";
    return {};
  }
  float score = begin_nodes_[len][0]->backtrace_score;
  for (Node* node = begin_nodes_[len][0]->prev;
       node != nullptr && node->prev != nullptr; node = node->prev) {
    results.push_back(node);
    if (results.size() > len) {
      LOG(ERROR) << "Detected cycle or infinite loop in Viterbi backtrace.";
      return {};
    }
  }

  std::reverse(results.begin(), results.end());

  return {results, score};
}

std::vector<float> Lattice::ForwardAlgorithm() const {
  const size_t len = size();
  std::vector<float> alpha(node_size_, 0.0);

  for (size_t pos = 0; pos <= len; ++pos) {
    for (Node* rnode : begin_nodes_[pos]) {
      for (Node* lnode : end_nodes_[pos]) {
        alpha[rnode->node_id] = LogSumExp(alpha[rnode->node_id],
                                          lnode->score + alpha[lnode->node_id],
                                          lnode == end_nodes_[pos][0]);
      }
    }
  }

  return alpha;
}

std::vector<float> Lattice::BackwardAlgorithm() const {
  const size_t len = size();
  std::vector<float> beta(node_size_, 0.0);

  for (int pos = len; pos >= 0; --pos) {
    for (Node* lnode : end_nodes_[pos]) {
      for (Node* rnode : begin_nodes_[pos]) {
        beta[lnode->node_id] =
            LogSumExp(beta[lnode->node_id], rnode->score + beta[rnode->node_id],
                      rnode == begin_nodes_[pos][0]);
      }
    }
  }

  return beta;
}

float Lattice::PopulateMarginal(float freq,
                                std::vector<float>* expected) const {
  if (expected == nullptr) return 0.0;

  const size_t len = size();
  const auto alpha = ForwardAlgorithm();
  const auto beta = BackwardAlgorithm();

  const float Z = alpha[begin_nodes_[len][0]->node_id];
  for (size_t pos = 0; pos < len; ++pos) {
    for (Node* node : begin_nodes_[pos]) {
      if (node->id >= 0) {
        (*expected)[node->id] +=
            freq *
            std::exp(static_cast<double>(alpha[node->node_id] + node->score +
                                         beta[node->node_id] - Z));
      }
    }
  }

  return freq * Z;
}

TrainerModel::~TrainerModel() = default;

const TrainerModel::SentencePieces& TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

void TrainerModel::BuildTrie(
    std::vector<std::pair<absl::string_view, int>>* pieces) {
  if (!status().ok()) return;

  if (pieces->empty()) {
    status_ = absl::InternalError("no pieces are loaded.");
    return;
  }

  std::sort(pieces->begin(), pieces->end());

  std::vector<const char*> key(pieces->size());
  std::vector<size_t> length(pieces->size());
  std::vector<int> value(pieces->size());
  for (size_t i = 0; i < pieces->size(); ++i) {
    key[i] = (*pieces)[i].first.data();
    length[i] = (*pieces)[i].first.size();
    value[i] = (*pieces)[i].second;
  }

  trie_ = std::make_unique<Darts::DoubleArray>();
  if (trie_->build(key.size(), const_cast<char**>(&key[0]),
                   const_cast<size_t*>(&length[0]), &value[0]) != 0) {
    status_ = absl::InternalError("cannot build double-array.");
    return;
  }

  const int kMaxTrieResultsSize = 1024;
  std::vector<Darts::DoubleArray::result_pair_type> results(
      kMaxTrieResultsSize);
  trie_results_size_ = 0;
  for (const auto& p : *pieces) {
    const int num_nodes = trie_->commonPrefixSearch(
        p.first.data(), results.data(), results.size(), p.first.size());
    trie_results_size_ = std::max(trie_results_size_, num_nodes);
  }

  if (trie_results_size_ == 0)
    status_ = absl::InternalError("no entry is found in the trie.");
}

void TrainerModel::PopulateNodes(Lattice* lattice) const {
  auto get_chars_length = [&lattice](int begin_pos, const char* end) {
    int pos = begin_pos;
    while (lattice->surface(pos) < end) ++pos;
    return pos - begin_pos;
  };

  const float unk_score = min_score() - kUnkPenalty;

  const size_t len = lattice->size();
  const char* end = lattice->sentence() + lattice->utf8_size();

  absl::FixedArray<Darts::DoubleArray::result_pair_type, 64> trie_results(
      trie_results_size_ + 1);

  for (size_t begin_pos = 0; begin_pos < len; ++begin_pos) {
    const char* begin = lattice->surface(begin_pos);

    const size_t num_nodes =
        std::min<size_t>(trie_->commonPrefixSearch(
                             begin, trie_results.data(), trie_results.size(),
                             static_cast<int>(end - begin)),
                         trie_results.size());

    bool has_single_node = false;

    for (size_t k = 0; k < num_nodes; ++k) {
      const int id = trie_results[k].value;
      const int length =
          get_chars_length(begin_pos, begin + trie_results[k].length);
      if (length <= 0) continue;
      Lattice::Node* node = lattice->Insert(begin_pos, length);
      if (!node) continue;
      node->id = id;
      node->score = sentencepieces_[id].second;
      if (!has_single_node && node->length == 1) {
        has_single_node = true;
      }
    }

    if (!has_single_node) {
      Lattice::Node* node = lattice->Insert(begin_pos, 1);
      if (node) {
        node->id = 0;
        node->score = unk_score;
      }
    }
  }
}

absl::Status TrainerModel::SetSentencePieces(SentencePieces&& sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);

  min_score_ = FLT_MAX;
  std::vector<std::pair<absl::string_view, int>> pieces;
  pieces.reserve(sentencepieces_.size());

  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const absl::string_view w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;         // score.
    RET_CHECK(std::isfinite(score));
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
  }

  BuildTrie(&pieces);
  return status();
}

TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() {
  if (sentences_.empty() ||
      (!trainer_spec_.auto_character_coverage() && required_chars_.empty())) {
    return {};
  }

  absl::flat_hash_map<std::string, int64_t> all_chars;
  for (const auto& w : sentences_) {
    for (const char32_t c : string_util::UTF8ToUnicodeText(w.first)) {
      if (c != kUNKChar &&
          c != TrainerInterface::kPretokenizationBoundaryChar) {
        all_chars[string_util::UnicodeCharToUTF8(c)] += w.second;
      }
    }
  }

  if (!trainer_spec_.seed_sentencepieces_file().empty()) {
    return LoadSeedSentencePiecesFromFile(all_chars);
  } else {
    return MakeSeedSentencePiecesFromCorpus(all_chars);
  }
}

TrainerModel::SentencePieces Trainer::LoadSeedSentencePiecesFromFile(
    const absl::flat_hash_map<std::string, int64_t>& all_chars) {
  // all_chars must be included in the seed sentencepieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  absl::flat_hash_set<std::string> seen_pieces;
  for (const auto& it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(it);
    seen_pieces.insert(it.first);
  }

  auto seed_sentencepieces_file = sentencepiece::filesystem::NewReadableFile(
      trainer_spec_.seed_sentencepieces_file());
  const size_t effective_seed_size =
      trainer_spec_.seed_sentencepiece_size() == 0
          ? std::max<size_t>(
                100000, static_cast<size_t>(trainer_spec_.vocab_size()) * 8)
          : static_cast<size_t>(trainer_spec_.seed_sentencepiece_size());
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
    if (!absl::SimpleAtoi(fields[1], &freq) || freq <= 0) {
      LOG(ERROR) << "Could not parse the frequency; line: " << line;
      return {};
    }
    const UnicodeText uw = string_util::UTF8ToUnicodeText(seed_sentencepiece);
    if (!IsValidSentencePiece(uw) ||
        !seen_pieces.insert(seed_sentencepiece).second) {
      ++skipped_sentencepieces;
      continue;
    }
    if (seed_sentencepieces.size() >= effective_seed_size) {
      LOG(WARNING) << "Exceeded seed_sentencepiece_size ("
                   << effective_seed_size
                   << "); ignoring remaining pieces in file.";
      break;
    }
    // Initialise score of a piece by character coverage.
    seed_sentencepieces.emplace_back(seed_sentencepiece, freq * uw.size());
  }

  LOG(INFO) << "skipped " << skipped_sentencepieces << " seed sentencepieces";

  // Take highest scoring pieces as initial vocab.
  seed_sentencepieces = Sorted(seed_sentencepieces);

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces from file.";

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  return seed_sentencepieces;
}

TrainerModel::SentencePieces Trainer::MakeSeedSentencePiecesFromCorpus(
    const absl::flat_hash_map<std::string, int64_t>& all_chars) {
  // all_chars must be included in the seed sentencepieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  for (const auto& it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(it);
  }

  size_t effective_seed_size =
      static_cast<size_t>(trainer_spec_.seed_sentencepiece_size());
  if (effective_seed_size == 0) {
    const size_t vocab_size = trainer_spec_.vocab_size();
    effective_seed_size = std::max<size_t>(100000, vocab_size * 8);
    LOG(INFO) << "Dynamically scaled seed_sentencepiece_size to "
              << effective_seed_size << " (8x target_vocab_size=" << vocab_size
              << ")";
  } else {
    LOG(INFO) << "Using explicit seed_sentencepiece_size="
              << effective_seed_size;
  }

  constexpr size_t kMaxChunkBytes = 500 * 1024 * 1024;  // 500 MB per chunk

#if defined(LIBSAIS_OPENMP)
  const int num_threads = trainer_spec_.num_threads();
  LOG(INFO) << "Extracting seed sentencepieces (libsais_omp, " << num_threads
            << " threads)";
#else
  LOG(INFO) << "Extracting seed sentencepieces (libsais)";
#endif

  BoundedPriorityQueue queue(effective_seed_size);

  size_t sentence_idx = 0;
  size_t chunk_id = 0;

  // Concatenate corpus sentences into contiguous 500 MB buffers.
  // Each sentence is terminated by a null byte ('\0') to act as a hard
  // boundary, ensuring suffix array intervals cannot match subwords across
  // distinct sentences or documents.
  while (sentence_idx < sentences_.size()) {
    std::string chunk_bytes;
    while (sentence_idx < sentences_.size() &&
           chunk_bytes.size() < kMaxChunkBytes) {
      const auto& w = sentences_[sentence_idx++];
      for (int64_t i = 0; i < w.second && chunk_bytes.size() < kMaxChunkBytes;
           ++i) {
        chunk_bytes.append(w.first);
        chunk_bytes.push_back('\0');  // Sentence boundary delimiter
      }
    }

    const int32_t n32 = static_cast<int32_t>(chunk_bytes.size());
    if (n32 == 0) break;

    chunk_id++;

    // Analytical dynamic minimum frequency floor derived from Zipf law:
    // min_freq = max(2, alpha * N / (K * ln(K)))
    // where:
    //   N = current chunk byte size (n32)
    //   K = target vocabulary size (vocab_size)
    //   alpha = min_freq_alpha flag (default 0.0 -> min_freq = 2)
    const float alpha = absl::GetFlag(FLAGS_min_freq_alpha);
    const float vocab_size =
        static_cast<float>(std::max<int>(1, trainer_spec_.vocab_size()));
    const float k_log_k = vocab_size * std::log(vocab_size);
    const uint64_t min_freq = std::max<uint64_t>(
        2ULL, static_cast<uint64_t>(alpha * static_cast<float>(n32) /
                                    std::max(1.0f, k_log_k)));

    LOG(INFO) << "[Chunk #" << chunk_id << "] Start processing " << n32
              << " bytes (base_min_freq=" << min_freq << ")...";

    const uint8_t* u_bytes =
        reinterpret_cast<const uint8_t*>(chunk_bytes.data());
    std::vector<int32_t> SA32(n32), LCP32(n32);

    // Construct Suffix Array (SA32) and Longest Common Prefix array (LCP32)
    // using induced-sorting algorithm (libsais) with OpenMP parallelization.
    // - SA32[i] stores the byte offset of the i-th lexicographically smallest
    // suffix.
    // - LCP32[i] stores the length of the longest common prefix between suffix
    // SA32[i-1] and SA32[i].
#if defined(LIBSAIS_OPENMP)
    {
      std::vector<int32_t> PLCP32(n32);
      libsais_omp(u_bytes, SA32.data(), n32, 0, NULL, num_threads);
      libsais_plcp_omp(u_bytes, SA32.data(), PLCP32.data(), n32, num_threads);
      libsais_lcp_omp(PLCP32.data(), SA32.data(), LCP32.data(), n32,
                      num_threads);
    }
#else
    {
      std::vector<int32_t> PLCP32(n32);
      libsais(u_bytes, SA32.data(), n32, 0, NULL);
      libsais_plcp(u_bytes, SA32.data(), PLCP32.data(), n32);
      libsais_lcp(PLCP32.data(), SA32.data(), LCP32.data(), n32);
    }
#endif

    // LCP Interval Tree Stack Traversal (Abouelhoda et al. linear-time
    // algorithm).
    //
    // Concept: Every internal node in the LCP interval tree corresponds to a
    // maximal recurring substring in the corpus.
    // A subword appearing at interval [l, r] with LCP depth `depth` occurs
    // exact `freq = r - l + 1` times in the text.
    struct LCPInterval {
      int32_t l;  // Leftmost index in the suffix array for this LCP interval
      int32_t depth;  // Length of the common prefix for all suffixes in [l, r]
    };

    std::vector<LCPInterval> stack;
    stack.push_back({0, 0});

    int64_t node_num = 0;
    int64_t valid_maximal_nodes = 0;

    for (int32_t i = 1; i <= n32; ++i) {
      int32_t lcp_val = (i < n32) ? LCP32[i] : 0;
      int32_t last_l = i - 1;

      // When LCP drops below the stack top's depth, all intervals deeper than
      // lcp_val have terminated. Pop and process them in post-order.
      while (lcp_val < stack.back().depth) {
        auto top = stack.back();
        stack.pop_back();
        last_l = top.l;

        int32_t l = top.l;
        int32_t r = i - 1;
        int32_t depth = top.depth;

        node_num++;

        auto process_interval = [&]() {
          const uint64_t freq = static_cast<uint64_t>(r - l + 1);
          // Skip single characters or pieces below minimum frequency threshold.
          // Candidate A: Length-clamped adaptive min_freq
          // Short subwords (depth <= 3 bytes) use min_freq = 2 to preserve
          // grammar/morphemes. Longer subwords (depth > 3 bytes) use
          // base_min_freq to prune long noise.
          const uint64_t eff_min_freq = (depth <= 3) ? 2ULL : min_freq;
          if (depth <= 1 || freq < eff_min_freq) return;

          const int32_t offset = SA32[l];
          // Skip invalid array range offsets.
          if (offset < 0 || offset + depth > n32) return;

          // Reject candidates that slice through multi-byte UTF-8 character
          // boundaries. In UTF-8, continuation bytes match binary pattern
          // 10xxxxxx (0x80..0xBF). A boundary is valid if neither the start
          // byte nor the byte immediately after the sequence is a UTF-8
          // continuation byte.
          const bool is_valid_utf8_boundary =
              ((u_bytes[offset] & 0xC0) != 0x80) &&
              (offset + depth == n32 ||
               (u_bytes[offset + depth] & 0xC0) != 0x80);
          if (!is_valid_utf8_boundary) return;

          absl::string_view piece(chunk_bytes.data() + offset, depth);
          // If piece ends with '\0' (sentence boundary), trim trailing '\0'
          while (!piece.empty() && piece.back() == '\0') {
            piece.remove_suffix(1);
          }
          // Skip candidate pieces crossing internal sentence boundary null
          // bytes ('\0').
          if (piece.find('\0') != absl::string_view::npos) return;

          // Fast zero-allocation character length validation using
          // register-based trail-byte checks.
          const size_t max_len =
              static_cast<size_t>(trainer_spec_.max_sentencepiece_length());
          const size_t utf8_len = string_util::UTF8Len(piece);
          if (utf8_len <= 1 || utf8_len > max_len) return;

          if (!IsValidSentencePiece(string_util::UTF8ToUnicodeText(piece))) {
            return;
          }

          valid_maximal_nodes++;
          queue.Add(piece, freq);
        };
        process_interval();

        if (lcp_val > stack.back().depth) {
          stack.push_back({top.l, lcp_val});
        }
      }

      if (lcp_val > stack.back().depth) {
        stack.push_back({last_l, lcp_val});
      }
    }

    queue.MaybeGc();

    LOG(INFO) << "[Chunk #" << chunk_id << "] Finished: Traversed " << node_num
              << " LCP nodes -> total unique candidates in heap="
              << queue.data().size();
  }

  LOG(INFO) << "Finished processing all " << chunk_id
            << " chunks. Filtering and sorting top candidates into seed "
               "pieces...";

  for (auto& [w, score] : queue.Get()) {
    CHECK(!all_chars.contains(w));
    seed_sentencepieces.emplace_back(std::move(w), score);
  }

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  LOG(INFO) << "Initialized " << seed_sentencepieces.size() << " seed pieces";

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

TrainerModel::SentencePieces Trainer::PruneUnreachableSentencePieces(
    const TrainerModel& model) const {
  const auto& sentencepieces = model.GetSentencePieces();
  TrainerModel::SentencePieces new_sentencepieces;

  Lattice lattice;
  int unreachable_count = 0;

  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    const auto& w = sentencepieces[i];
    lattice.SetSentence(w.first);
    model.PopulateNodes(&lattice);

    // Find the self-node covering the entire length of the sentencepiece.
    Lattice::Node* self_node = nullptr;
    for (Lattice::Node* node : lattice.begin_nodes(0)) {
      if (node->id == static_cast<int>(i) &&
          node->length == static_cast<uint32_t>(lattice.size())) {
        self_node = node;
        break;
      }
    }

    if (!self_node) continue;
    const float original_score = self_node->score;

    // Temporarily disable the self_node to force Viterbi to find an alternative
    // path.
    self_node->score = -std::numeric_limits<float>::infinity();

    const auto alt_path = lattice.Viterbi();

    // Restore original score.
    self_node->score = original_score;

    // If an alternative path exists and its score is strictly higher than
    // original_score, then the piece is shadowed by its decomposition and
    // unreachable in Viterbi.
    if (!alt_path.first.empty() && alt_path.second > original_score) {
      ++unreachable_count;
      LOG(INFO) << "Pruned unreachable piece: ID=" << i << " piece='" << w.first
                << "' (score=" << original_score
                << " vs alt_score=" << alt_path.second << ")";
      continue;
    }

    new_sentencepieces.push_back(w);
  }

  LOG(INFO) << "PruneUnreachableSentencePieces pruned " << unreachable_count
            << " unreachable pieces out of " << sentencepieces.size();

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel& model) const {
  const auto& sentencepieces = model.GetSentencePieces();
  absl::flat_hash_map<std::string, float> final_sentencepieces;
  absl::flat_hash_map<std::string, float> sp(sentencepieces.begin(),
                                             sentencepieces.end());

  // required_chars_ must be included in the final sentencepieces ONLY IF
  // auto_character_coverage is false.
  if (!trainer_spec_.auto_character_coverage()) {
    float min_score_penalty = 0.0;
    constexpr float kMinScorePenaltyDelta = 0.0001;
    for (const auto& w : Sorted(required_chars_)) {
      const std::string s = string_util::UnicodeCharToUTF8(w.first);
      if (const auto it = sp.find(s); it != sp.end()) {
        final_sentencepieces[s] = it->second;
      } else {
        // Add penalty to avoid required pieces from having the same score.
        // Since the required_chars_ is sorted, frequent pieces have
        // less penalties.
        final_sentencepieces[s] = model.min_score() + min_score_penalty;
        min_score_penalty += kMinScorePenaltyDelta;
      }
    }
  }

  const int vocab_size_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  if (vocab_size_size <= 0) return {};

  // Then keeps sentencepieces with higher scores.
  for (const auto& w : Sorted(sentencepieces)) {
    if (final_sentencepieces.contains(w.first)) {
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
  ABSL_RETURN_IF_ERROR(status());

  RET_CHECK_EQ(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  RET_CHECK(normalizer_spec_.escape_whitespaces());

  TrainerModel model;

  ABSL_RETURN_IF_ERROR(model.status());
  ABSL_RETURN_IF_ERROR(LoadSentences());
  // Note: Prerequisites for auto_character_coverage (e.g.,
  // --byte_fallback=true) are already validated in VerifySpec() during
  // TrainerInterface initialization / LoadSentences().
  if (!trainer_spec_.auto_character_coverage()) {
    RET_CHECK(!required_chars_.empty());
  }

  auto seed_sentencepieces = MakeSeedSentencePieces();
  RET_CHECK(!seed_sentencepieces.empty());

  ABSL_RETURN_IF_ERROR(model.SetSentencePieces(std::move(seed_sentencepieces)));

  // Seed pieces are extracted from whole sentences above. EM only needs
  // per-word frequencies when pieces cannot cross whitespace, so collapse
  // sentences into a deduplicated word-frequency list. This drastically
  // reduces the EM cost on whitespace-delimited languages.
  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

  desired_vocab_size_ = static_cast<size_t>(trainer_spec_.vocab_size() * 1.1);

  ABSL_RETURN_IF_ERROR(TrainSparsePruning(&model));

  auto reachable_pieces = PruneUnreachableSentencePieces(model);
  ABSL_RETURN_IF_ERROR(model.SetSentencePieces(std::move(reachable_pieces)));

  // Finally, adjusts the size of sentencepices to be |vocab_size|.
  final_pieces_ = FinalizeSentencePieces(model);

  return Save();
}

namespace {
int64_t ComputeTotalCorpusBytes(const TrainerInterface::Sentences& sentences) {
  int64_t total_corpus_bytes = 0;
  for (const auto& w : sentences) {
    total_corpus_bytes += w.first.size() * w.second;
  }
  return total_corpus_bytes;
}
}  // namespace

absl::Status Trainer::TrainSparsePruning(TrainerModel* model) {
  const int64_t total_corpus_bytes = ComputeTotalCorpusBytes(sentences_);
  auto calc_bytes_per_tok = [&](int64_t tokens) -> float {
    return (tokens > 0) ? static_cast<float>(total_corpus_bytes) / tokens
                        : 0.0f;
  };
  float lambda = 0.0f;
  LOG(INFO) << "Starting Sparse Pruning (target K=" << desired_vocab_size_
            << ")";
  int epoch = 0;
  const size_t target_size = desired_vocab_size_;
  std::vector<float> expected;
  while (true) {
    // Step 1: E-step & Log evaluation.
    float objective = 0.0;
    int64_t num_tokens = 0;
    expected = RunEStep(*model, &objective, &num_tokens);

    // Step 2: Primal-Dual feedback control (dynamic quantile annealing).
    // Anneals shadow price (lambda) at a 25% geometric decay rate per epoch
    // (kShrinkingFactor = 0.75f) to find market-clearing exchange rate for
    // budget K.
    if (expected.size() > target_size * 1.05) {
      constexpr float kShrinkingFactor = 0.75f;
      const size_t desired_active = std::max(
          target_size, static_cast<size_t>(expected.size() * kShrinkingFactor));
      // Copy expected to active_counts because std::nth_element is destructive,
      // preserving the indexed alignment of expected with model pieces for
      // Step 3.
      std::vector<float> active_counts = expected;
      std::nth_element(active_counts.begin(),
                       active_counts.begin() + desired_active,
                       active_counts.end(), std::greater<float>());
      lambda = std::max(lambda, active_counts[desired_active]);
    }

    LOG(INFO) << "Sparse epoch=" << epoch << " active=" << expected.size()
              << " lambda=" << lambda << " obj=" << objective
              << " |w|=" << num_tokens
              << " bytes/tok=" << calc_bytes_per_tok(num_tokens);

    // Step 3: M-step with proximal soft-thresholding (w - lambda).
    // Using expected[i] >= lambda guarantees retaining at least desired_active
    // (>= target_size) active pieces determined by nth_element in Step 2,
    // eliminating loop side-effects and vector index-order dependencies.
    TrainerModel::SentencePieces new_pieces;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] >= lambda) {
        const float c = std::max(1e-4f, expected[i] - lambda);
        new_pieces.emplace_back(model->GetSentencePieces()[i].first, c);
      }
    }
    ToLogProb(new_pieces.begin(), new_pieces.end());
    ABSL_RETURN_IF_ERROR(model->SetSentencePieces(std::move(new_pieces)));

    // Step 4: Convergence check (target reached or max epochs).
    if (expected.size() <= target_size || epoch >= 50) {
      break;
    }
    epoch++;
  }

  // Step 5: Post-Lasso debiased refit (unpenalized lambda=0 MLE).
  // Eliminates L1 shrinkage bias and Bayesian prior bias on surviving active
  // vocabulary V* via unpenalized MLE normalization (ToLogProb), restoring
  // exact Maximum Likelihood estimates without subtracting lambda or applying
  // Digamma.
  const bool debias = absl::GetFlag(FLAGS_post_l1_debias);
  if (debias) {
    LOG(INFO) << "Post-Lasso Refit...";
    float post_obj = 0.0;
    int64_t post_tokens = 0;
    const auto post_expected = RunEStep(*model, &post_obj, &post_tokens);
    TrainerModel::SentencePieces refit_pieces;
    refit_pieces.reserve(post_expected.size());
    for (size_t i = 0; i < post_expected.size(); ++i) {
      refit_pieces.emplace_back(model->GetSentencePieces()[i].first,
                                std::max(1e-4f, post_expected[i]));
    }
    ToLogProb(refit_pieces.begin(), refit_pieces.end());
    ABSL_RETURN_IF_ERROR(model->SetSentencePieces(std::move(refit_pieces)));
    LOG(INFO) << "Post-Lasso Refit Done: size=" << model->GetPieceSize()
              << " obj=" << post_obj << " |w|=" << post_tokens
              << " bytes/tok=" << calc_bytes_per_tok(post_tokens);
  }
  return absl::OkStatus();
}
}  // namespace unigram
}  // namespace sentencepiece
