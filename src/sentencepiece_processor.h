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

#ifndef SENTENCEPIECE_PROCESSOR_H_
#define SENTENCEPIECE_PROCESSOR_H_

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace absl {
class string_view;
}  // namespace absl

namespace sentencepiece {

// SentencePieceProcessor:
// Simple and language independent tokenizer and de-tokenizer for
// Neural Network Machine Translation.
//
// SentencePieceProcessor provides Encode() and Decode() methods,
// which correspond to tokenization and de-tokenization respectively.
//
// - Encode:
//   Given a raw source sentence, encode it into a sequence
//   of pieces or vocabulary ids.
//
// - Decode:
//   Given a sequence of pieces or vocabulary ids, decode it
//   into a de-tokenized raw sentence.
//
// SentencePieceProcessor provides a lossless data conversion
// that allows the original raw sentence to be perfectly reconstructed
// from the encoded data, i.e., Decode(Encode(input)) == input.
// This characteristics is useful, as we can make the de-tokenization
// completely language independent.
//
// Usage:
//   SentencePieceProcessor sp;
//   sp.Load("//path/to/model");
//
//   vector<string> sps;
//   sp.Encode("hello world.", &sps);
//
//   vector<int> ids;
//   sp.Encode("hello world.", &ids);
//
//   string detok;
//   sp.Decode(sps, &detok);
//   CHECK_EQ("hello world.", detok);
//
//   sp.Decode(ids, &detok);
//   CHECK_EQ("hello world.", detok);
//
//  We can also use SentencePieceText which manages the byte-offsets
//  between user input (output) and internal sentence pieces.
//
//   SentencePieceText spt;
//   sp.Encode("hello world.", &spt);
//   // Emits the byte range of each piece.
//   for (const auto &piece : spt.pieces()) {
//      LOG(INFO) << piece.begin() << " " << piece.end();
//   }
//
//   sp.Decode({0, 1, 2, 3..}, &spt);
//   for (const auto &piece : spt.pieces()) {
//      LOG(INFO) << piece.begin() << " " << piece.end();
//   }
//

class SentencePieceText;
class NBestSentencePieceText;
class ModelInterface;
class ModelProto;

#ifndef SWIG
using EncodeResult = std::vector<std::pair<absl::string_view, int>>;
#endif  // SWIG

namespace normalizer {
class Normalizer;
}  // namespace normalizer

namespace util {
namespace error {
enum Code {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
};
}  // namespace error

class Status {
 public:
  Status();
  ~Status();
  Status(error::Code code, const char *error_message);
  Status(error::Code code, const std::string &error_message);
  Status(const Status &s);
  void operator=(const Status &s);
  bool operator==(const Status &s) const;
  bool operator!=(const Status &s) const;
  inline bool ok() const { return rep_ == nullptr; }

  void set_error_message(const char *str);
  const char *error_message() const;
  error::Code code() const;
  std::string ToString() const;

  void IgnoreError();

 private:
  struct Rep;
  std::unique_ptr<Rep> rep_;
};

// Minimum string_view class that is used only for
// the argument of public APIs.
class min_string_view {
 public:
  min_string_view() : ptr_(nullptr), length_(0) {}
  min_string_view(const std::string &str)
      : ptr_(str.data()), length_(str.size()) {}
  min_string_view(const char *str) : ptr_(str), length_(std::strlen(str)) {}
  min_string_view(const char *data, size_t len) : ptr_(data), length_(len) {}

  const char *data() const { return ptr_; }
  size_t size() const { return length_; }

 private:
  const char *ptr_ = nullptr;
  size_t length_ = 0;
};

// Redefine std::string for serialized_proto interface as Python's string is
// a Unicode string. We can enforce the return value to be raw byte sequence
// with SWIG's typemap.
using bytes = std::string;
}  // namespace util

class SentencePieceProcessor {
 public:
  SentencePieceProcessor();
  virtual ~SentencePieceProcessor();

  // Loads model from `filename`.
  // Returns false if `filename` cannot be loaded.
  virtual util::Status Load(util::min_string_view filename);

  // Loads model from `filename`.
  // Crash if `filename` cannot be loaded.
  virtual void LoadOrDie(util::min_string_view filename);

  // Loads model from `is`.
  // Returns false if `is` cannot be loaded.
  virtual util::Status Load(std::istream *is);

  // Loads model from `model_proto`.
  // `model_proto` is copied.
  virtual util::Status Load(const ModelProto &model_proto);

  // Loads model from `model_proto`.
  // `model_proto` is moved.
  virtual util::Status Load(std::unique_ptr<ModelProto> &&model_proto);

  // Loads model from `serialized`, which is a string-serialized model proto.
  // Useful to load the model from a platform independent blob object.
  virtual util::Status LoadFromSerializedProto(
      util::min_string_view serialized);

  // Returns the status. Encode/Decode methods are valid when status is OK.
  virtual util::Status status() const;

  // Sets encode extra_option sequence.
  virtual util::Status SetEncodeExtraOptions(
      util::min_string_view extra_option);

  // Sets decode extra_option sequence.
  virtual util::Status SetDecodeExtraOptions(
      util::min_string_view extra_option);

  //////////////////////////////////////////////////////////////
  // Vocabulary restriction.
  // Background:
  // https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt

  // Restricts the vocabulary set.
  // The input sentences are encoded into the tokens in `valid_vocab`.
  virtual util::Status SetVocabulary(
      const std::vector<std::string> &valid_vocab);

  // Reverts the vocabulary restriction.
  virtual util::Status ResetVocabulary();

  // Loads the valid vocabulary set from `filename` in TSV format.
  // Format:  <token> <tab> <freq>.
  // Any token with frequency < threshold will be treated as OOV.
  virtual util::Status LoadVocabulary(util::min_string_view filename,
                                      int threshold);

  //////////////////////////////////////////////////////////////
  // Simple API.
  //
  // Given a UTF8 input, encodes it into a sequence of sentence pieces.
  virtual util::Status Encode(util::min_string_view input,
                              std::vector<std::string> *pieces) const;

  // Given a UTF8 input, encodes it into a sequence of ids.
  virtual util::Status Encode(util::min_string_view input,
                              std::vector<int> *ids) const;

  // Given a sequence of pieces, decodes it into a detokenized output.
  virtual util::Status Decode(const std::vector<std::string> &pieces,
                              std::string *detokenized) const;

  // Given a sequence of ids, decodes it into a detokenized output.
  virtual util::Status Decode(const std::vector<int> &ids,
                              std::string *detokenized) const;

  //////////////////////////////////////////////////////////////
  // NBest API.
  // Same as Encode, but returns nbest results.
  virtual util::Status NBestEncode(
      util::min_string_view input, int nbest_size,
      std::vector<std::vector<std::string>> *pieces) const;

  // Same as Encode, but returns nbest results.
  virtual util::Status NBestEncode(util::min_string_view input, int nbest_size,
                                   std::vector<std::vector<int>> *ids) const;

  //////////////////////////////////////////////////////////////
  // Sampling API
  // When `nbest_size` is positive value, approximately samples one segmentation
  // from nbest candidates.
  // When `nbest_size` is negative value, samples one segmentation from
  // the hypotheses (Lattice) according to the generation probabilities using
  // forward-filtering and backward-sampling algorithm.
  // `alpha` is a smoothing parameter.  The best segmentation
  // (Viterbi segmentation) is more likely sampled when setting larger
  // alpha. When alpha is 0.0, one segmentation is uniformly sampled from the
  // nbest or lattice.
  // `nbest_size` and `alpha` correspond to parameters `l` and `alpha`
  // in https://arxiv.org/abs/1804.10959  (nbest_size < 0 means l = infinity)
  virtual util::Status SampleEncode(util::min_string_view input, int nbest_size,
                                    float alpha,
                                    std::vector<std::string> *pieces) const;

  // Same as above, but returns a sequence of ids.
  virtual util::Status SampleEncode(util::min_string_view input, int nbest_size,
                                    float alpha, std::vector<int> *ids) const;

  //////////////////////////////////////////////////////////////
  // Advanced API returning SentencePieceText, which manages
  // utf8-byte alignments between user-input/detokenized text
  // and internal sentencepiece sequence.
  //
  // Given a UTF8 input, encodes it into SentencePieceText.
  virtual util::Status Encode(util::min_string_view input,
                              SentencePieceText *spt) const;

  // Same as above, but returns NBestSentencePieceText.
  virtual util::Status NBestEncode(util::min_string_view input, int nbest_size,
                                   NBestSentencePieceText *nbest_spt) const;

  // Same as above, but samples one segmentation from the hypotheses (Lattice).
  virtual util::Status SampleEncode(util::min_string_view input, int nbest_size,
                                    float alpha, SentencePieceText *spt) const;

  // Given a sequence of pieces, decodes it into SentencePieceText.
  virtual util::Status Decode(const std::vector<std::string> &pieces,
                              SentencePieceText *spt) const;

  // Given a sequence of ids, decodes it into SentencePieceText.
  virtual util::Status Decode(const std::vector<int> &ids,
                              SentencePieceText *spt) const;

  //////////////////////////////////////////////////////////////
  // Handy methods that return the result directly.
  // These functions ignore internal errors.
#ifdef SWIG
#define DEFINE_SPP_DIRECT_FUNC_IMPL(FuncName, OutType, ...) \
  OutType output;                                           \
  const auto _status = FuncName(__VA_ARGS__, &output);      \
  if (!_status.ok()) throw _status;                         \
  return output;
#else
#define DEFINE_SPP_DIRECT_FUNC_IMPL(FuncName, OutType, ...) \
  OutType output;                                           \
  FuncName(__VA_ARGS__, &output).IgnoreError();             \
  return output;
#endif

  virtual std::vector<std::string> EncodeAsPieces(
      util::min_string_view input) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Encode, std::vector<std::string>, input);
  }

  virtual std::vector<int> EncodeAsIds(util::min_string_view input) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Encode, std::vector<int>, input);
  }

  virtual std::vector<std::vector<std::string>> NBestEncodeAsPieces(
      util::min_string_view input, int nbest_size) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(
        NBestEncode, std::vector<std::vector<std::string>>, input, nbest_size);
  }

  virtual std::vector<std::vector<int>> NBestEncodeAsIds(
      util::min_string_view input, int nbest_size) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(NBestEncode, std::vector<std::vector<int>>,
                                input, nbest_size);
  }

  virtual std::vector<std::string> SampleEncodeAsPieces(
      util::min_string_view input, int nbest_size, float alpha) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(SampleEncode, std::vector<std::string>, input,
                                nbest_size, alpha);
  }

  virtual std::vector<int> SampleEncodeAsIds(util::min_string_view input,
                                             int nbest_size,
                                             float alpha) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(SampleEncode, std::vector<int>, input,
                                nbest_size, alpha);
  }

  virtual std::string DecodePieces(
      const std::vector<std::string> &pieces) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Decode, std::string, pieces);
  }

  virtual std::string DecodeIds(const std::vector<int> &ids) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Decode, std::string, ids);
  }

#undef DEFINE_SPP_DIRECT_FUNC_IMPL

  // They are used in Python interface. Returns serialized proto.
  // In python module, we can get access to the full Proto after
  // deserialzing the returned byte sequence.
  virtual util::bytes EncodeAsSerializedProto(
      util::min_string_view input) const;

  virtual util::bytes SampleEncodeAsSerializedProto(util::min_string_view input,
                                                    int nbest_size,
                                                    float alpha) const;

  virtual util::bytes NBestEncodeAsSerializedProto(util::min_string_view input,
                                                   int nbest_size) const;

  virtual util::bytes DecodePiecesAsSerializedProto(
      const std::vector<std::string> &pieces) const;

  virtual util::bytes DecodeIdsAsSerializedProto(
      const std::vector<int> &ids) const;

  //////////////////////////////////////////////////////////////
  // Vocabulary management methods.
  //
  // Returns the size of sentence pieces, which is the same as
  // the size of vocabulary for NMT.
  virtual int GetPieceSize() const;

  // Returns the vocab id of `piece`.
  // Returns UNK(0) if `piece` is unknown.
  virtual int PieceToId(util::min_string_view piece) const;

  // Returns the string representation of vocab with `id`.
  virtual const std::string &IdToPiece(int id) const;

  // Returns the score of `id`.
  // Usually score is an emission log probability of unigram language model.
  virtual float GetScore(int id) const;

  // Returns true if `id` is unknown symbol.
  virtual bool IsUnknown(int id) const;

  // Returns true if `id` is control symbol.
  virtual bool IsControl(int id) const;

  // Returns true if `id` is unused symbol.
  virtual bool IsUnused(int id) const;

  // Returns the reserved id.
  // Returns -1 if not defined.

  // Returns unknown (<unk>) id.
  virtual int unk_id() const;

  // Returns BOS (<s>) id.
  virtual int bos_id() const;

  // Returns EOS (</s>) id.
  virtual int eos_id() const;

  // Returns PAD (<pad>) id.
  virtual int pad_id() const;

#ifndef SWIG
  //////////////////////////////////////////////////////////////
  // Model management.
  //
  // Allows injection of a mock model instance. `model` is moved.
  void SetModel(std::unique_ptr<ModelInterface> &&model);

  // Allows injection of a normalizer instance. `normalizer` is moved.
  void SetNormalizer(std::unique_ptr<normalizer::Normalizer> &&normalizer);
#endif

  // Returns immutable model proto. Useful to obtain extended
  // or experimental parameters encoded in model_proto.
  const ModelProto &model_proto() const;

 private:
  enum ExtraOption { REVERSE, BOS, EOS };

  util::Status ParseExtraOptions(util::min_string_view extra_option,
                                 std::vector<ExtraOption> *extra_options) const;

  util::Status ApplyExtraOptions(const std::vector<ExtraOption> &extra_options,
                                 SentencePieceText *spt) const;

  util::Status PopulateSentencePieceText(
      util::min_string_view input, util::min_string_view normalized,
      const std::vector<size_t> &norm_to_orig, const EncodeResult &result,
      SentencePieceText *spt) const;

  std::unique_ptr<ModelInterface> model_;
  std::unique_ptr<normalizer::Normalizer> normalizer_;

  // Underlying model protocol buffer. The same lifetime as model_.
  std::unique_ptr<ModelProto> model_proto_;

  std::vector<ExtraOption> encode_extra_options_;
  std::vector<ExtraOption> decode_extra_options_;
};
}  // namespace sentencepiece
#endif  // SENTENCEPIECE_PROCESSOR_H_
