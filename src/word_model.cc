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

#include "word_model.h"

#include "util.h"

namespace sentencepiece {
namespace word {

Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  CheckControlSymbols();

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    CHECK(!sp.piece().empty());
    if (sp.type() == ModelProto::SentencePiece::NORMAL ||
        sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
      CHECK(sp.has_score());
      port::InsertOrDie(&pieces_, sp.piece(), i);
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

  std::vector<std::pair<StringPiece, int>> output;
  for (const auto &w : SplitIntoWords(normalized)) {
    output.emplace_back(w, PieceToId(w));
  }

  return output;
}

}  // namespace word
}  // namespace sentencepiece
