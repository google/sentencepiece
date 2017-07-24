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

#include <memory>
#include <string>
#include <vector>

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
class ModelInterface;
class ModelProto;

namespace normalizer {
class Normalizer;
}  // namespace normalizer

class SentencePieceProcessor {
 public:
  SentencePieceProcessor();
  virtual ~SentencePieceProcessor();

  // Loads model from |filename|.
  // Returns false if |filename| cannot be loaded.
  virtual bool Load(const std::string &filename);

  // Loads model from |is|.
  // Returns false if |is| cannot be loaded.
  virtual bool Load(std::istream *is);

  // Loads model from |filename|.
  // Dies if |filename| cannot be loaded.
  virtual void LoadOrDie(const std::string &filename);

  // Loads model from |is|.
  // Dies if |is| cannot be loaded.
  virtual void LoadOrDie(std::istream *is);

  // Sets encode extra_option sequence.
  virtual void SetEncodeExtraOptions(const std::string &extra_option);

  // Sets dncode extra_option sequence.
  virtual void SetDecodeExtraOptions(const std::string &extra_option);

  //////////////////////////////////////////////////////////////
  // Simple API.
  //
  // Given a UTF8 input, encodes it into a sequence of sentence pieces.
  virtual void Encode(const std::string &input,
                      std::vector<std::string> *pieces) const;

  // Given a UTF8 input, encodes it into a sequence of ids.
  virtual void Encode(const std::string &input, std::vector<int> *ids) const;

  // Given a sequence of pieces, decodes it into a detokenized output.
  virtual void Decode(const std::vector<std::string> &pieces,
                      std::string *detokenized) const;

  // Given a sequence of ids, decodes it into a detokenized output.
  virtual void Decode(const std::vector<int> &ids,
                      std::string *detokenized) const;

  //////////////////////////////////////////////////////////////
  // Advanced API returning SentencePieceText, which manages
  // utf8-byte alignments between user-input/detokenized text
  // and internal sentencepiece sequence.
  //
  // Given a UTF8 input, encodes it into SentencePieceText.
  virtual void Encode(const std::string &input, SentencePieceText *spt) const;

  // Given a sequence of pieces, decodes it into SentencePieceText.
  virtual void Decode(const std::vector<std::string> &pieces,
                      SentencePieceText *spt) const;

  // Given a sequence of ids, decodes it into SentencePieceText.
  virtual void Decode(const std::vector<int> &ids,
                      SentencePieceText *spt) const;

  //////////////////////////////////////////////////////////////
  // Vocabulary management methods.
  //
  // Returns the size of sentence pieces, which is the same as
  // the size of vocabulary for NMT.
  virtual int GetPieceSize() const;

  // Returns the vocab id of |piece|.
  // Returns UNK(0) if |piece| is unknown.
  virtual int PieceToId(const std::string &piece) const;

  // Returns the string representation of vocab with |id|.
  virtual std::string IdToPiece(int id) const;

  // Returns the score of |id|.
  // Usually score is an emission log probability of unigram language model.
  virtual float GetScore(int id) const;

  // Returns true if |id| is unknown symbol.
  virtual bool IsUnknown(int id) const;

  // Returns true if |id| is control symbol.
  virtual bool IsControl(int id) const;

  //////////////////////////////////////////////////////////////
  // Model management.
  //
  // Allows injection of a mock model instance. |model| is moved.
  void SetModel(std::unique_ptr<ModelInterface> &&model);

  // Allows injection of a normalizer instance. |normalizer| is moved.
  void SetNormalizer(std::unique_ptr<normalizer::Normalizer> &&normalizer);

  // Returns immutable model proto. Useful to obtain extended
  // or experimental parameters encoded in model_proto.
  const ModelProto &model_proto() const;

 private:
  enum ExtraOption { REVERSE, BOS, EOS };

  static std::vector<ExtraOption> ParseExtraOptions(
      const std::string &extra_option);
  void ApplyExtraOptions(const std::vector<ExtraOption> &extra_options,
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
