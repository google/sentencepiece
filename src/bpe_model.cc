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

#include "bpe_model.h"

#include <queue>
#include "util.h"

namespace sentencepiece {
namespace bpe {

Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  CheckControlSymbols();

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    CHECK(!sp.piece().empty());
    if (sp.type() == ModelProto::SentencePiece::NORMAL) {
      CHECK(sp.has_score());
      port::InsertOrDie(&pieces_, sp.piece(), i);
    } else if (sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
      // TODO(taku): implement USER_DEFINED symbol.
      LOG(FATAL) << "User defined symbol is not supported in BPE";
    } else {
      port::InsertOrDie(&reserved_id_map_, sp.piece(), i);
    }
  }
}

Model::~Model() {}

std::vector<std::pair<StringPiece, int>> Model::Encode(
    StringPiece normalized) const {
  if (normalized.empty()) {
    return {};
  }

  struct SymbolPair {
    int left;     // left index of this pair
    int right;    // right index of this pair
    float score;  // score of this pair. large is better.
    size_t size;  // length of this piece
  };

  class SymbolPairComparator {
   public:
    const bool operator()(SymbolPair *h1, SymbolPair *h2) {
      return (h1->score < h2->score ||
              (h1->score == h2->score && h1->left > h2->left));
    }
  };

  struct Symbol {
    int prev;  // prev index of this symbol. -1 for BOS.
    int next;  // next index of tihs symbol. -1 for EOS.
    StringPiece piece;
  };

  using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>,
                                     SymbolPairComparator>;
  Agenda agenda;
  std::vector<Symbol> symbols;
  symbols.reserve(normalized.size());

  // Lookup new symbol pair at [left, right] and inserts it to agenda.
  auto MaybeAddNewSymbolPair = [this, &symbols, &agenda](int left, int right) {
    if (left == -1 || right == -1) return;
    const StringPiece piece(
        symbols[left].piece.data(),
        symbols[left].piece.size() + symbols[right].piece.size());
    const auto it = pieces_.find(piece);
    if (it == pieces_.end()) {
      return;
    }
    auto *h = new SymbolPair;
    h->left = left;
    h->right = right;
    h->score = GetScore(it->second);
    h->size = piece.size();
    agenda.push(h);
  };

  // Splits the input into character sequence
  const char *begin = normalized.data();
  const char *end = normalized.data() + normalized.size();
  int index = 0;
  while (begin < end) {
    int mblen = string_util::OneCharLen(begin);
    if (mblen > end - begin) {
      LOG(ERROR) << "Invalid character length.";
      mblen = end - begin;
    }
    Symbol s;
    s.piece = StringPiece(begin, mblen);
    s.prev = begin == normalized.data() ? -1 : index - 1;
    begin += mblen;
    s.next = begin == end ? -1 : index + 1;
    ++index;
    symbols.emplace_back(s);
  }
  CHECK(!symbols.empty());

  // Lookup all bigrams.
  for (size_t i = 1; i < symbols.size(); ++i) {
    MaybeAddNewSymbolPair(i - 1, i);
  }

  // Main loop.
  while (!agenda.empty()) {
    std::unique_ptr<SymbolPair> top(agenda.top());
    agenda.pop();

    // |top| is no longer available.
    if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
        symbols[top->left].piece.size() + symbols[top->right].piece.size() !=
            top->size) {
      continue;
    }

    // Replaces symbols with |top| rule.
    symbols[top->left].piece = StringPiece(
        symbols[top->left].piece.data(),
        symbols[top->left].piece.size() + symbols[top->right].piece.size());

    // Updates prev/next pointers.
    symbols[top->left].next = symbols[top->right].next;
    if (symbols[top->right].next >= 0) {
      symbols[symbols[top->right].next].prev = top->left;
    }
    symbols[top->right].piece = StringPiece("");

    // Adds new symbol pairs which are newly added after symbol replacement.
    MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
    MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
  }

  std::vector<std::pair<StringPiece, int>> output;
  for (int index = 0; index != -1; index = symbols[index].next) {
    CHECK_GE(index, 0);
    CHECK_LT(index, static_cast<int>(symbols.size()));
    output.emplace_back(symbols[index].piece, PieceToId(symbols[index].piece));
  }

  return output;
}
}  // namespace bpe
}  // namespace sentencepiece
