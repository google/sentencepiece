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

#include <algorithm>
#include "sentencepiece_model.pb.h"
#include "util.h"

namespace sentencepiece {

PrefixMatcher::PrefixMatcher(const std::set<absl::string_view> &dic) {
  if (dic.empty()) return;
  std::vector<const char *> key;
  key.reserve(dic.size());
  for (const auto &it : dic) key.push_back(it.data());
  trie_ = port::MakeUnique<Darts::DoubleArray>();
  CHECK_EQ(0, trie_->build(key.size(), const_cast<char **>(&key[0]), nullptr,
                           nullptr));
}

int PrefixMatcher::PrefixMatch(absl::string_view w, bool *found) const {
  if (trie_ == nullptr) {
    if (found) *found = false;
    return std::min<int>(w.size(), string_util::OneCharLen(w.data()));
  }

  constexpr int kResultSize = 64;
  Darts::DoubleArray::result_pair_type trie_results[kResultSize];
  const int num_nodes =
      trie_->commonPrefixSearch(w.data(), trie_results, kResultSize, w.size());

  if (found) *found = (num_nodes > 0);
  if (num_nodes == 0) {
    return std::min<int>(w.size(), string_util::OneCharLen(w.data()));
  }

  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }

  return mblen;
}

std::string PrefixMatcher::GlobalReplace(absl::string_view w,
                                         absl::string_view out) const {
  std::string result;
  while (!w.empty()) {
    bool found = false;
    const int mblen = PrefixMatch(w, &found);
    if (found) {
      result.append(out.data(), out.size());
    } else {
      result.append(w.data(), mblen);
    }
    w.remove_prefix(mblen);
  }
  return result;
}

const char *ModelInterface::kUNK() { return "<unk>"; }
const char *ModelInterface::kBOS() { return "<s>"; }
const char *ModelInterface::kEOS() { return "</s>"; }
const char *ModelInterface::kPAD() { return "<pad>"; };

ModelInterface::ModelInterface(const ModelProto &model_proto)
    : model_proto_(&model_proto), status_(util::OkStatus()) {}
ModelInterface::~ModelInterface() {}

int ModelInterface::PieceToId(absl::string_view piece) const {
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

void ModelInterface::InitializePieces(bool use_prefix_matcher) {
  pieces_.clear();
  reserved_id_map_.clear();
  unk_id_ = -1;

  std::set<absl::string_view> user_defined_symbols;

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    if (sp.piece().empty()) {
      status_ = util::InternalError("piece must not be empty.");
      return;
    }

    const bool is_normal_piece =
        (sp.type() == ModelProto::SentencePiece::NORMAL ||
         sp.type() == ModelProto::SentencePiece::USER_DEFINED ||
         sp.type() == ModelProto::SentencePiece::UNUSED);
    if (!port::InsertIfNotPresent(
            is_normal_piece ? &pieces_ : &reserved_id_map_, sp.piece(), i)) {
      status_ = util::InternalError(sp.piece() + " is already defined.");
      return;
    }

    if (use_prefix_matcher &&
        sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
      user_defined_symbols.insert(sp.piece());
    }

    if (sp.type() == ModelProto::SentencePiece::UNKNOWN) {
      if (unk_id_ >= 0) {
        status_ = util::InternalError("unk is already defined.");
        return;
      }
      unk_id_ = i;
    }
  }

  if (unk_id_ == -1) {
    status_ = util::InternalError("unk is not defined.");
    return;
  }

  if (use_prefix_matcher) {
    matcher_ = port::MakeUnique<PrefixMatcher>(user_defined_symbols);
  }
}

std::vector<absl::string_view> SplitIntoWords(absl::string_view text) {
  const char *begin = text.data();
  const char *end = text.data() + text.size();

  // Space symbol (U+2581)
  const absl::string_view kSpaceSymbol = "\xe2\x96\x81";

  std::vector<absl::string_view> result;
  while (begin < end) {
    const int mblen =
        std::min<int>(string_util::OneCharLen(begin), end - begin);
    if (begin == text.data() ||
        absl::string_view(begin, mblen) == kSpaceSymbol) {
      result.emplace_back(begin, 0);  // add empty string piece.
    }
    result.back() =
        absl::string_view(result.back().data(), result.back().size() + mblen);
    begin += mblen;
  }

  return result;
}

}  // namespace sentencepiece
