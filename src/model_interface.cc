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
    : model_proto_(&model_proto), status_(util::OkStatus()) {}
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

int ModelInterface::GetPieceSize() const { return model_proto_->pieces_size(); }

std::string ModelInterface::IdToPiece(int id) const {
  return model_proto_->pieces(id).piece();
}

float ModelInterface::GetScore(int id) const {
  return model_proto_->pieces(id).score();
}

bool ModelInterface::IsControl(int id) const {
  return (model_proto_->pieces(id).type() ==
          ModelProto::SentencePiece::CONTROL);
}

bool ModelInterface::IsUnknown(int id) const {
  return (model_proto_->pieces(id).type() ==
          ModelProto::SentencePiece::UNKNOWN);
}

void ModelInterface::InitializePieces(bool enable_user_defined) {
  pieces_.clear();
  reserved_id_map_.clear();
  unk_id_ = 0;

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    if (!enable_user_defined &&
        sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
      status_ = util::StatusBuilder(util::error::INTERNAL)
                << "user defined symbol is not supported.";
      return;
    }

    const bool is_normal_piece =
        (sp.type() == ModelProto::SentencePiece::NORMAL ||
         sp.type() == ModelProto::SentencePiece::USER_DEFINED);
    if (!port::InsertIfNotPresent(
            is_normal_piece ? &pieces_ : &reserved_id_map_, sp.piece(), i)) {
      status_ = util::StatusBuilder(util::error::INTERNAL)
                << "\"" << sp.piece() << "\" is already defined.";
      return;
    }

    if (sp.type() == ModelProto::SentencePiece::UNKNOWN) unk_id_ = i;
  }
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
    result.back() =
        StringPiece(result.back().data(), result.back().size() + mblen);
    begin += mblen;
  }

  return result;
}

}  // namespace sentencepiece
