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

#ifndef UNIGRAM_MODEL_TRAINER_H_
#define UNIGRAM_MODEL_TRAINER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "darts.h"
#include "sentencepiece_model.pb.h"
#include "trainer_interface.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {

using string_util::UnicodeText;

// Lattice represents a search space of sentence piece segmentation.
class Lattice {
 public:
  Lattice();
  ~Lattice();

  struct Node {
    uint32_t pos;           // Unicode position in the sentence.
    uint32_t length;        // Unicode length, not UTF8 byte.
    uint32_t node_id;       // unique id in the current lattice.
    int id;                 // vocab id. (maybe -1 for UNK)
    float score;            // logprob of this sentencepiece.
    float backtrace_score;  // backtrace info used in Viterbi.
    Node* prev;             // best previous node on Viterbi path.
  };

  // Returns nodes starting at |pos|.
  const std::vector<Node*>& begin_nodes(int pos) const;

  // Returns Unicode character length.
  int size() const;

  // Returns multi-byte (utf8) length.
  int utf8_size() const;

  // Returns the substring of sentence. sentence[pos:]
  const char* surface(int pos) const;

  // Returns immutable sentence. The same as surface(0)
  const char* sentence() const;

  // Clears the lattice.
  void Clear();

  // Sets new sentence.
  void SetSentence(absl::string_view sentence);

  // Inserts a new node at [pos, pos + length - 1].
  // After calling this method, The caller must set Node::score and Node::id.
  Node* Insert(int pos, int length);

  using LatticePathWithScore = std::pair<std::vector<Node*>, float>;

  // Returns Viterbi path. All nodes must be populated in advance.
  LatticePathWithScore Viterbi();

  // Populates marginal probability of every node in this lattice.
  // |freq| is the frequency of the sentence.
  // Returns the log-likelihood of this sentence.
  float PopulateMarginal(float freq, std::vector<float>* expected) const;

 private:
  Node* NewNode();
  std::vector<float> ForwardAlgorithm() const;
  std::vector<float> BackwardAlgorithm() const;

  absl::string_view sentence_;
  std::vector<const char*> surface_;
  std::vector<std::vector<Node*>> begin_nodes_;
  std::vector<std::vector<Node*>> end_nodes_;
  std::vector<std::unique_ptr<Node[]>> node_blocks_;
  size_t node_size_ = 0;
};

class TrainerModel {
 public:
  using SentencePieces = std::vector<std::pair<std::string, float>>;

  TrainerModel() = default;
  ~TrainerModel();

  absl::Status status() const { return status_; }

  int GetPieceSize() const { return static_cast<int>(sentencepieces_.size()); }

  // Returns the sentencepieces.
  // The meta symbols, e.g., </s> are NOT included.
  const SentencePieces& GetSentencePieces() const;

  // Sets sentencepieces. The sentencepieces are moved.
  // The meta symbols, e.g., </s> are NOT included.
  absl::Status SetSentencePieces(SentencePieces&& sentencepieces);

  // Returns the minimum score in sentence pieces.
  // min_score() - 10 is used for the cost of unknown sentence.
  float min_score() const { return min_score_; }

  // Populates all sentence pieces to the |lattice|.
  // After calling this function, lattice.Viterbi() returns the
  // best segmentation.
  void PopulateNodes(Lattice* lattice) const;

 private:
  void BuildTrie(std::vector<std::pair<absl::string_view, int>>* pieces);

  SentencePieces sentencepieces_;
  float min_score_ = 0.0;
  std::unique_ptr<Darts::DoubleArray> trie_;
  int trie_results_size_ = 0;
  absl::Status status_;
};

class Trainer : public TrainerInterface {
 public:
  Trainer(const TrainerSpec& trainer_spec,
          const NormalizerSpec& normalizer_spec)
      : TrainerInterface::TrainerInterface(trainer_spec, normalizer_spec) {}

  // Makes and returns initial seed sentencepieces for EM training.
  TrainerModel::SentencePieces MakeSeedSentencePieces();

  absl::Status Train() override;

 private:
  FRIEND_TEST(UnigramTrainerTest, PruneUnreachableSentencePiecesTest);

  // Loads and returns seed sentencepieces from external TSV file specified in
  // seed_sentencepieces_file.
  TrainerModel::SentencePieces LoadSeedSentencePiecesFromFile(
      const absl::flat_hash_map<std::string, int64_t>& all_chars);

  // Dynamically constructs and returns seed sentencepieces from the training
  // corpus using parallel Suffix Array (libsais_omp).
  TrainerModel::SentencePieces MakeSeedSentencePiecesFromCorpus(
      const absl::flat_hash_map<std::string, int64_t>& all_chars);

  // Executes the E step of EM and returns expected count.
  // The index of return array is the vocab id.
  // |objective| is a negative likelihood of the current model.
  // |num_token| is the number of total tokens to tokenize
  // training corpus.
  std::vector<float> RunEStep(const TrainerModel& model, float* objective,
                              int64_t* num_tokens) const;

  // Prunes unreachable sentencepieces whose score is strictly lower than
  // their alternative decomposition (Viterbi score inversion).
  TrainerModel::SentencePieces PruneUnreachableSentencePieces(
      const TrainerModel& model) const;

  // Runs continuous L1 relaxation proximal sparse pruning EM loop.
  absl::Status TrainSparsePruning(TrainerModel* model);

  // Makes the final sentence pieces by incorporating the required characters
  // and control/user defined symbols.
  TrainerModel::SentencePieces FinalizeSentencePieces(
      const TrainerModel& model) const;

  // When the size of SentencePieces becomes less than desired_vocab_size_,
  // break the main training loop. desired_vocab_size_ = 1.1 * vocab_size_
  // for now.
  int desired_vocab_size_;
};
}  // namespace unigram
}  // namespace sentencepiece
#endif  // UNIGRAM_MODEL_TRAINER_H_
