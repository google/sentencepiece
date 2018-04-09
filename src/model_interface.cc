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

#include "model_interface.h"
#include "sentencepiece_model.pb.h"
#include "util.h"

namespace sentencepiece {

ModelInterface::ModelInterface(const ModelProto &model_proto)
    : model_proto_(&model_proto) {}
ModelInterface::~ModelInterface() {}

int ModelInterface::PieceToId(StringPiece piece) const {
  auto it = reserved_id_map_.find(piece);
  if (it != reserved_id_map_.end()) {
    return it->second;
  }
  auto it2 = pieces_.find(piece);
  if (it2 != pieces_.end()) {
    return it2->second;
  }
  return unk_id_;
}

int ModelInterface::GetPieceSize() const {
  return CHECK_NOTNULL(model_proto_)->pieces_size();
}

std::string ModelInterface::IdToPiece(int id) const {
  return CHECK_NOTNULL(model_proto_)->pieces(id).piece();
}

float ModelInterface::GetScore(int id) const {
  return CHECK_NOTNULL(model_proto_)->pieces(id).score();
}

bool ModelInterface::IsControl(int id) const {
  return (CHECK_NOTNULL(model_proto_)->pieces(id).type() ==
          ModelProto::SentencePiece::CONTROL);
}

bool ModelInterface::IsUnknown(int id) const {
  return (CHECK_NOTNULL(model_proto_)->pieces(id).type() ==
          ModelProto::SentencePiece::UNKNOWN);
}

std::vector<StringPiece> SplitIntoWords(StringPiece text) {
  const char *begin = text.data();
  const char *end = text.data() + text.size();

  // Space symbol (U+2581)
  const StringPiece kSpaceSymbol = "\xe2\x96\x81";

  std::vector<StringPiece> result;
  while (begin < end) {
    const int mblen =
        std::min<int>(string_util::OneCharLen(begin), end - begin);
    if (begin == text.data() || StringPiece(begin, mblen) == kSpaceSymbol) {
      result.emplace_back(begin, 0);  // add empty string piece.
    }
    CHECK(!result.empty());
    result.back() =
        StringPiece(result.back().data(), result.back().size() + mblen);
    begin += mblen;
  }

  return result;
}

}  // namespace sentencepiece
