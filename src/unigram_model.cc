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

#include "unigram_model.h"

#include <cfloat>
#include <map>
#include <queue>
#include <string>
#include <vector>

#include "stringpiece.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {
constexpr size_t kNodeChunkSize = 512;
}

Lattice::Lattice() {}
Lattice::~Lattice() { Clear(); }

const std::vector<Lattice::Node *> &Lattice::begin_nodes(int pos) const {
  return begin_nodes_[pos];
}

const std::vector<Lattice::Node *> &Lattice::end_nodes(int pos) const {
  return end_nodes_[pos];
}

int Lattice::size() const {
  // -1 because surface_ may include the EOS.
  return std::max<int>(0, surface_.size() - 1);
}

int Lattice::utf8_size() const { return sentence_.size(); }

const char *Lattice::sentence() const { return sentence_.data(); }

const char *Lattice::surface(int pos) const { return surface_[pos]; }

Lattice::Node *Lattice::bos_node() const { return end_nodes_[0][0]; }

Lattice::Node *Lattice::eos_node() const { return begin_nodes_[size()][0]; }

Lattice::Node *Lattice::NewNode() {
  Node *node = new Node;
  memset(node, 0, sizeof(*node));
  node->node_id = all_nodes_.size();
  all_nodes_.push_back(node);
  return node;
}

void Lattice::Clear() {
  begin_nodes_.clear();
  end_nodes_.clear();
  sentence_.clear();
  surface_.clear();
  all_nodes_.clear();
  port::STLDeleteElements(&all_nodes_);
}

void Lattice::SetSentence(StringPiece sentence) {
  Clear();

  sentence_ = sentence;
  CHECK(!sentence_.empty());

  const char *begin = sentence_.data();
  const char *end = sentence_.data() + sentence_.size();
  while (begin < end) {
    const int mblen =
        std::min<int>(string_util::OneCharLen(begin), end - begin);
    surface_.push_back(begin);
    begin += mblen;
  }
  surface_.push_back(end);

  const int len = size();
  begin_nodes_.resize(len + 1);
  end_nodes_.resize(len + 1);

  for (int i = 0; i <= len; ++i) {
    begin_nodes_[i].reserve(16);
    end_nodes_[i].reserve(16);
  }

  Node *bos = NewNode();
  bos->id = -1;
  bos->pos = 0;
  end_nodes_[0].push_back(bos);

  Node *eos = NewNode();
  eos->id = -1;
  eos->pos = len;
  begin_nodes_[len].push_back(eos);
}

Lattice::Node *Lattice::Insert(int pos, int length) {
  Node *node = NewNode();
  node->pos = pos;
  node->length = length;
  const int utf8_length =
      static_cast<int>(surface(pos + length) - surface(pos));
  node->piece.set(surface(pos), utf8_length);
  begin_nodes_[pos].push_back(node);
  end_nodes_[pos + node->length].push_back(node);

  return node;
}

std::vector<Lattice::Node *> Lattice::Viterbi() {
  const int len = size();
  CHECK_GT(len, 0);

  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      rnode->prev = nullptr;
      float best_score = 0.0;
      Node *best_node = nullptr;
      for (Node *lnode : end_nodes_[pos]) {
        const float score = lnode->backtrace_score + rnode->score;
        if (best_node == nullptr || score > best_score) {
          best_node = lnode;
          best_score = score;
        }
      }
      CHECK(best_node);
      rnode->prev = best_node;
      rnode->backtrace_score = best_score;
    }
  }

  // backtrace
  std::vector<Node *> results;
  for (Node *node = begin_nodes_[len][0]->prev; node->prev != nullptr;
       node = node->prev) {
    results.push_back(node);
  }

  std::reverse(results.begin(), results.end());

  return results;
}

float Lattice::PopulateMarginal(float freq,
                                std::vector<float> *expected) const {
  CHECK_NOTNULL(expected);

  // Returns log(exp(x) + exp(y)).
  // if flg is true, returns log(exp(y)) == y.
  // log(\sum_i exp(a[i])) can be computed as
  // for (int i = 0; i < a.size(); ++i)
  //   x = LogSumExp(x, a[i], i == 0);
  auto LogSumExp = [](float x, float y, bool init_mode) -> float {
    if (init_mode) {
      return y;
    }
    const float vmin = std::min(x, y);
    const float vmax = std::max(x, y);
    constexpr float kMinusLogEpsilon = 50;
    if (vmax > vmin + kMinusLogEpsilon) {
      return vmax;
    } else {
      return vmax + log(exp(vmin - vmax) + 1.0);
    }
  };

  const int len = size();
  CHECK_GT(len, 0);

  // alpha and beta (accumulative log prob) in Forward Backward.
  // the index of alpha/beta is Node::node_id.
  std::vector<float> alpha(all_nodes_.size(), 0.0);
  std::vector<float> beta(all_nodes_.size(), 0.0);

  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      for (Node *lnode : end_nodes_[pos]) {
        alpha[rnode->node_id] = LogSumExp(alpha[rnode->node_id],
                                          lnode->score + alpha[lnode->node_id],
                                          lnode == end_nodes_[pos][0]);
      }
    }
  }

  for (int pos = len; pos >= 0; --pos) {
    for (Node *lnode : end_nodes_[pos]) {
      for (Node *rnode : begin_nodes_[pos]) {
        beta[lnode->node_id] =
            LogSumExp(beta[lnode->node_id], rnode->score + beta[rnode->node_id],
                      rnode == begin_nodes_[pos][0]);
      }
    }
  }

  const float Z = alpha[begin_nodes_[len][0]->node_id];
  for (int pos = 0; pos < len; ++pos) {
    for (Node *node : begin_nodes_[pos]) {
      if (node->id >= 0) {
        // the index of |expected| is a Node::id, which is a vocabulary id.
        (*expected)[node->id] += freq * exp(alpha[node->node_id] + node->score +
                                            beta[node->node_id] - Z);
      }
    }
  }

  return freq * Z;
}

std::vector<std::vector<Lattice::Node *>> Lattice::NBest(size_t nbest_size) {
  CHECK_GT(size(), 0);
  CHECK_GE(nbest_size, 1);

  // Uses A* search to enumerate N-bests.
  // Given a lattice, enumerates hypotheses (paths) from EOS.
  // At each partial path x, compute f(x) as follows
  //   f(x) = g(x) + h(x).
  // g(x): the sum of scores from  EOS to the left-most node in x.
  // h(x): a heuristic that estimates the largest score from x to BOS.
  // f(x): the priority to pop a new hypothesis from the priority queue.
  //
  // As left-to-right Viterbi search can tell the *exact* value of h(x),
  // we can obtain the exact n-best results with A*.
  struct Hypothesis {
    Node *node;
    Hypothesis *next;
    float fx;
    float gx;
  };

  class HypothesisComparator {
   public:
    const bool operator()(Hypothesis *h1, Hypothesis *h2) {
      return (h1->fx < h2->fx);
    }
  };

  using Agenda = std::priority_queue<Hypothesis *, std::vector<Hypothesis *>,
                                     HypothesisComparator>;

  Agenda agenda;
  std::vector<Hypothesis *> allocated;
  std::vector<std::vector<Node *>> results;

  auto NewHypothesis = [&allocated]() {
    Hypothesis *h = new Hypothesis;
    memset(h, 0, sizeof(*h));
    allocated.push_back(h);
    return h;
  };

  auto *eos = NewHypothesis();
  eos->node = eos_node();
  eos->next = nullptr;
  eos->fx = eos->node->score;
  eos->gx = eos->node->score;
  agenda.push(eos);

  // Run Viterbi first to fill backtrace score.
  Viterbi();

  while (!agenda.empty()) {
    auto *top = agenda.top();
    agenda.pop();
    auto *node = top->node;

    // Reaches to BOS
    if (node == bos_node()) {
      results.resize(results.size() + 1);
      for (auto *n = top->next; n->next != nullptr; n = n->next) {
        results.back().push_back(n->node);
      }
      if (results.size() == nbest_size) {
        break;
      }
      continue;
    }

    // Expands new node ending at node->pos
    for (Node *lnode : end_nodes(node->pos)) {
      auto *hyp = NewHypothesis();
      hyp->node = lnode;
      hyp->gx = lnode->score + top->gx;  // just adds node->score
      hyp->fx =
          lnode->backtrace_score + top->gx;  // backtrace_score is h(node).
      hyp->next = top;
      agenda.push(hyp);
    }
  }

  port::STLDeleteElements(&allocated);
  return results;
}

ModelBase::ModelBase() {}
ModelBase::~ModelBase() {}

void ModelBase::PopulateNodes(Lattice *lattice) const {
  CHECK_NOTNULL(lattice);
  CHECK_NOTNULL(trie_);

  auto GetCharsLength = [](const char *begin, int len) {
    const char *end = begin + len;
    int result = 0;
    while (begin < end) {
      begin += std::min<int>(string_util::OneCharLen(begin), end - begin);
      ++result;
    }
    return result;
  };

  constexpr float kUnkPenalty = 10.0;
  const float unk_score = min_score() - kUnkPenalty;

  const int len = lattice->size();
  const char *end = lattice->sentence() + lattice->utf8_size();

  // Initializes the buffer for return values.
  CHECK_GT(trie_results_size_, 0);

  // +1 just in case.
  std::vector<Darts::DoubleArray::result_pair_type> trie_results(
      trie_results_size_ + 1);

  for (int begin_pos = 0; begin_pos < len; ++begin_pos) {
    const char *begin = lattice->surface(begin_pos);

    // Finds all pieces which are prefix of surface(begin_pos).
    const size_t num_nodes = trie_->commonPrefixSearch(
        begin, trie_results.data(), trie_results.size(),
        static_cast<int>(end - begin));
    CHECK_LT(num_nodes, trie_results.size());

    bool has_single_node = false;

    // Inserts pieces to the lattice.
    for (size_t k = 0; k < num_nodes; ++k) {
      const int length = GetCharsLength(begin, trie_results[k].length);
      Lattice::Node *node = lattice->Insert(begin_pos, length);
      node->id = trie_results[k].value;  // the value of Trie stores vocab_id.
      node->score = GetScore(node->id);  // calls method defined in subclass.
      if (!has_single_node && node->length == 1) {
        has_single_node = true;
      }
    }

    if (!has_single_node) {
      Lattice::Node *node = lattice->Insert(begin_pos, 1);
      node->id = kUnkID;  // add UNK node.
      node->score = unk_score;
    }
  }
}

int ModelBase::PieceToId(StringPiece piece) const {
  auto it = reserved_id_map_.find(piece);
  if (it != reserved_id_map_.end()) {
    return it->second;
  }
  int id = 0;
  trie_->exactMatchSearch(piece.data(), id);
  return id == -1 ? kUnkID : id;
}

void ModelBase::BuildTrie(std::vector<std::pair<std::string, int>> *pieces) {
  CHECK_NOTNULL(pieces);
  CHECK(!pieces->empty());

  // sort by sentencepiece since DoubleArray::build()
  // only accepts sorted strings.
  sort(pieces->begin(), pieces->end());

  // Makes key/value set for DoubleArrayTrie.
  std::vector<const char *> key(pieces->size());
  std::vector<int> value(pieces->size());
  for (size_t i = 0; i < pieces->size(); ++i) {
    key[i] = (*pieces)[i].first.c_str();  // sorted piece.
    value[i] = (*pieces)[i].second;       // vocab_id
  }

  trie_ = port::MakeUnique<Darts::DoubleArray>();
  CHECK_EQ(0,
           trie_->build(key.size(), const_cast<char **>(&key[0]), nullptr,
                        &value[0]))
      << "cannot build double-array";

  // Computes the maximum number of shared prefixes in the trie.
  const int kMaxTrieResultsSize = 1024;
  std::vector<Darts::DoubleArray::result_pair_type> results(
      kMaxTrieResultsSize);
  trie_results_size_ = 0;
  for (const auto &p : *pieces) {
    const int num_nodes = trie_->commonPrefixSearch(
        p.first.data(), results.data(), results.size(), p.first.size());
    trie_results_size_ = std::max(trie_results_size_, num_nodes);
  }
  CHECK_GT(trie_results_size_, 0);
}

Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  min_score_ = FLT_MAX;

  CheckControlSymbols();

  std::vector<std::pair<std::string, int>> pieces;  // <piece, vocab_id>
  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    CHECK(!sp.piece().empty());
    if (sp.type() == ModelProto::SentencePiece::NORMAL ||
        sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
      CHECK(sp.has_score());
      pieces.emplace_back(sp.piece(), i);
    } else {
      port::InsertOrDie(&reserved_id_map_, sp.piece(), i);
    }
    if (sp.type() == ModelProto::SentencePiece::NORMAL) {
      min_score_ = std::min(min_score_, sp.score());
    }
  }

  BuildTrie(&pieces);
}

Model::~Model() {}

std::vector<std::pair<StringPiece, int>> Model::Encode(
    StringPiece normalized) const {
  if (normalized.empty()) {
    return {};
  }

  Lattice lattice;
  lattice.SetSentence(normalized);
  PopulateNodes(&lattice);

  std::vector<std::pair<StringPiece, int>> results;
  for (const auto *node : lattice.Viterbi()) {
    results.emplace_back(node->piece, node->id);
  }

  return results;
}
}  // namespace unigram
}  // namespace sentencepiece
