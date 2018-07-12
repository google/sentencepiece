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

#ifndef MODEL_INTERFACE_H_
#define MODEL_INTERFACE_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {

// "_this_is_a_pen" => ["_this", "_is", "_a", "_pen"]
std::vector<absl::string_view> SplitIntoWords(absl::string_view text);

using EncodeResult = std::vector<std::pair<absl::string_view, int>>;
using NBestEncodeResult = std::vector<std::pair<EncodeResult, float>>;

class ModelProto;

// Given a list of strings, finds the longest string which is a
// prefix of a query.
class PrefixMatcher {
 public:
  // Initializes the PrefixMatcher with `dic`.
  explicit PrefixMatcher(const std::set<absl::string_view> &dic);

  // Finds the longest string in dic, which is a prefix of `w`.
  // Returns the UTF8 byte length of matched string.
  // `found` is set if a prefix match exists.
  // If no entry is found, consumes one Unicode character.
  int PrefixMatch(absl::string_view w, bool *found = nullptr) const;

  // Replaces entries in `w` with `out`.
  std::string GlobalReplace(absl::string_view w, absl::string_view out) const;

 private:
  std::unique_ptr<Darts::DoubleArray> trie_;
};

// Underlying model interface.
// Given a normalized string, returns a sequence of sentence pieces with ids.
class ModelInterface {
 public:
  using PieceToIdMap =
      std::unordered_map<absl::string_view, int, string_util::string_view_hash>;

  static const char kUNK[];
  static const char kBOS[];
  static const char kEOS[];
  static const char kPAD[];

  // `model_proto` should not be deleted until ModelInterface is destroyed.
  explicit ModelInterface(const ModelProto &model_proto);
  ModelInterface() {}

  virtual ~ModelInterface();

  // Returns Status.
  // Encode/Decode functions are valid only when status is OK.
  virtual util::Status status() const { return status_; }

  virtual const ModelProto &model_proto() const { return *model_proto_; }

  // Given a normalized string, returns a sequence of sentence pieces with ids.
  // The concatenation of pieces must be the same as `normalized`.
  virtual EncodeResult Encode(absl::string_view normalized) const = 0;

  // The same as above, but returns nbest result with score.
  virtual NBestEncodeResult NBestEncode(absl::string_view normalized,
                                        int nbest_size) const {
    LOG(ERROR) << "Not implemented.";
    return NBestEncodeResult();
  }

  virtual EncodeResult SampleEncode(absl::string_view normalized,
                                    float alpha) const {
    LOG(ERROR) << "Not implemented.";
    return EncodeResult();
  }

  // Returns the size of sentence pieces, which is the same
  // as the size of vocabulary for NMT.
  virtual int GetPieceSize() const;

  // Returns the vocab id of `piece`.
  // Returns UNK(0) if `piece` is unknown
  virtual int PieceToId(absl::string_view piece) const;

  // Returns the string representation of vocab with `id`.
  // id must be 0 <= id < GetPieceSize().
  virtual std::string IdToPiece(int id) const;

  // Returns the score of `id`.
  // Score represents a log probability of the piece.
  // We can roughly estimate the unigram frequency of the piece.
  virtual float GetScore(int id) const;

  // Returns true if `id` is unknown symbol.
  virtual bool IsUnknown(int id) const;

  // Returns true if `id` is control symbol.
  virtual bool IsControl(int id) const;

  // Returns true if `id` is unused symbol.
  virtual bool IsUnused(int id) const;

  // Returns true if `id` is user defined symbol.
  virtual bool IsUserDefined(int id) const;

 protected:
  void InitializePieces(bool use_prefix_matcher);

  const ModelProto *model_proto_ = nullptr;

  // PrefixMatcher for user defined symbols.
  std::unique_ptr<PrefixMatcher> matcher_;

  // piece -> id map for normal pieces
  PieceToIdMap pieces_;

  // piece -> id map for control and unknown
  PieceToIdMap reserved_id_map_;

  // unknown id.
  int unk_id_ = 0;

  // status.
  util::Status status_;
};
}  // namespace sentencepiece
#endif  // MODEL_INTERFACE_H_
