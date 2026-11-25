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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

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
//   sp.Encode("hello world.", &sps).IgnoreError();
//
//   vector<int> ids;
//   sp.Encode("hello world.", &ids).IgnoreError();
//
//   string detok;
//   sp.Decode(sps, &detok);
//   CHECK_EQ("hello world.", detok).IgnoreError();
//
//   sp.Decode(ids, &detok);
//   CHECK_EQ("hello world.", detok).IgnoreError();
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

class NBestSentencePieceText;
class SentencePieceText;
class ModelProto;
class NormalizerSpec;

namespace lite {
class SentencePieceLiteProcessor;
}  // namespace lite

// Default ThreadPool implemented using Abseil functionality.
// If you want to use a custom implementation, please inherit from it.
//
// Note: This ThreadPool does not support recursive calls. Scheduling a new task
// on the same ThreadPool from within an already scheduled task will cause a
// severe deadlock. Please use a different ThreadPool instance instead.
class ThreadPool {
 public:
  ThreadPool() = delete;
  explicit ThreadPool(size_t num_threads);
  virtual ~ThreadPool();

  virtual void Schedule(std::function<void()> func);
  [[nodiscard]] virtual size_t num_threads() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Currently, the C++ API does not include a dedicated batch processing API.
// However, you can safely perform batch processing in coordination with the
// existing ThreadPool by using the RunBatch utility below.
//
// Executes tasks concurrently with dynamic load-balancing. Stops early if any
// task returns an error.
// `total_tasks`: Number of tasks to execute (= batch size)
// `task_func`:   Function to process a task by index.
// `pool`:        ThreadPool for scheduling workers.
//
// Sample:
//
// ThreadPool pool(32);
// std::vector<std::string> ins = {...};
// std::vector<std::vector<int>> outs(ins.size());
// auto status = sentencepiece::RunBatch(inputs.size(), [&](size_t i) {
//   return spm.Encode(ins[i], &outs[i]);
// }, pool);
absl::Status RunBatch(size_t total_tasks,
                      std::function<absl::Status(size_t index)> task_func,
                      ThreadPool& pool);

namespace util {
using bytes = std::string;
}  // namespace util

class SentencePieceProcessor {
 public:
  SentencePieceProcessor();
  virtual ~SentencePieceProcessor();

  // Loads model from `filename`.
  // Returns false if `filename` cannot be loaded.
  virtual absl::Status Load(absl::string_view filename);

  // Loads model from `filename`.
  // Crash if `filename` cannot be loaded.
  virtual void LoadOrDie(absl::string_view filename);

  // Loads model from `model_proto`.
  // `model_proto` is copied.
  virtual absl::Status Load(const ModelProto& model_proto);

  // Loads model from `model_proto`.
  // `model_proto` is moved.
  virtual absl::Status Load(std::unique_ptr<ModelProto> model_proto);

  // Loads model from `serialized`, which is a string-serialized model proto.
  // Useful to load the model from a platform independent blob object.
  virtual absl::Status LoadFromSerializedProto(absl::string_view serialized);

  // Loads model from `serialized`, which is a serialized FlatBuffer model
  // binary (.spm.fb). `serialized` is copied into an internal owned buffer.
  virtual absl::Status LoadFromSerializedFlatbuffer(
      absl::string_view serialized);

  // Loads model from `flatbuffer_view` without copying (zero-copy).
  // The caller must guarantee that `flatbuffer_view` remains valid for the
  // lifetime of this SentencePieceProcessor instance. If `flatbuffer_view` is
  // not 8-byte aligned, it is copied into an internal owned buffer instead
  // (equivalent to `LoadFromSerializedFlatbuffer`).
  virtual absl::Status LoadFromFlatbufferView(
      absl::string_view flatbuffer_view);

  // Returns the status. Encode/Decode methods are valid when status is OK.
  virtual absl::Status status() const;

  // Sets encode extra_option sequence.
  virtual absl::Status SetEncodeExtraOptions(absl::string_view extra_option);

  // Sets decode extra_option sequence.
  virtual absl::Status SetDecodeExtraOptions(absl::string_view extra_option);

  //////////////////////////////////////////////////////////////
  // Simple Encode and Decode API.
  //
  // Given a UTF8 input, encodes it into a sequence of sentence pieces.
  virtual absl::Status Encode(absl::string_view input,
                              std::vector<std::string>* pieces) const;

  // Given a UTF8 input, encodes it into a sequence of ids.
  virtual absl::Status Encode(absl::string_view input,
                              std::vector<int>* ids) const;

  // Given a sequence of pieces, decodes it into a detokenized output.
  virtual absl::Status Decode(absl::Span<const std::string> pieces,
                              std::string* detokenized) const;

  // Given a sequence of pieces, decodes it into a detokenized output.
  virtual absl::Status Decode(absl::Span<const absl::string_view> pieces,
                              std::string* detokenized) const;

  // Given a sequence of ids, decodes it into a detokenized output.
  virtual absl::Status Decode(absl::Span<const int> ids,
                              std::string* detokenized) const;

  //////////////////////////////////////////////////////////////
  // NBest API (Deprecated).
  //
  // Deprecated: Use `SampleEncode` for subword regularization instead.
  // Returns the exact 1-best Viterbi segmentation followed by deduplicated
  // approximate candidates sampled via `SampleNormalized` in non-increasing
  // model score order (`1 <= nbest_size <= 512`; Unigram models only).
  // Returns fewer than `nbest_size` candidates when sampling finds fewer
  // distinct segmentations (and 1 empty candidate with `score = 0.0` when the
  // normalized input is empty).
  [[deprecated(
      "NBestEncode is deprecated; use SampleEncode for subword "
      "regularization.")]]
  virtual absl::Status NBestEncode(
      absl::string_view input, int nbest_size,
      std::vector<std::vector<std::string>>* pieces) const;

  // Deprecated: Same as above, returning sequences of token IDs (Unigram only).
  [[deprecated(
      "NBestEncode is deprecated; use SampleEncode for subword "
      "regularization.")]]
  virtual absl::Status NBestEncode(absl::string_view input, int nbest_size,
                                   std::vector<std::vector<int>>* ids) const;

  //////////////////////////////////////////////////////////////
  // Sampling API.
  //
  // Unigram and BPE support sampling mode.
  // Note: In 0.3.0 (`sentencepiece::lite` runtime), `nbest_size` is ignored for
  // both Unigram and BPE; sampling always draws directly from the full lattice
  // (equivalent to `nbest_size = -1` in 0.2.x).
  // - Unigram (--model_type=unigram):
  // Samples one segmentation from the full lattice using Gumbel-Max dynamic
  // programming with smoothing parameter `alpha` (inverse temperature). The
  // best segmentation (Viterbi segmentation) is more likely sampled when
  // setting larger `alpha`. When `alpha` is 0.0, one segmentation is uniformly
  // sampled from the lattice.
  //
  // - BPE (--model_type=bpe):
  // `alpha`: The dropout probability `p` of bpe merge operations in
  // https://arxiv.org/abs/1910.13267 Nbest-based sampling is not supported so
  // nbest_size parameter is ignored in BPE.
  virtual absl::Status SampleEncode(absl::string_view input, int nbest_size,
                                    float alpha,
                                    std::vector<std::string>* pieces) const;

  // Same as above, but returns a sequence of ids.
  virtual absl::Status SampleEncode(absl::string_view input, int nbest_size,
                                    float alpha, std::vector<int>* ids) const;

  //////////////////////////////////////////////////////////////
  // Advanced API returning SentencePieceText, which manages
  // utf8-byte alignments between user-input/detokenized text
  // and internal sentencepiece sequence.
  //
  // Given a UTF8 input, encodes it into SentencePieceText.
  //
  // When using these APIs, sentencepiece.pb.h header files must be included.

  virtual absl::Status Encode(absl::string_view input,
                              SentencePieceText* spt) const;

  // Deprecated: Use `SampleEncode` for subword regularization instead.
  // Returns the exact 1-best Viterbi segmentation followed by deduplicated
  // approximate sampled candidates as `NBestSentencePieceText`
  // (`1 <= nbest_size <= 512`; Unigram models only).
  [[deprecated(
      "NBestEncode is deprecated; use SampleEncode for subword "
      "regularization.")]]
  virtual absl::Status NBestEncode(absl::string_view input, int nbest_size,
                                   NBestSentencePieceText* nbest_spt) const;

  virtual absl::Status SampleEncode(absl::string_view input, int nbest_size,
                                    float alpha, SentencePieceText* spt) const;

  virtual absl::Status Decode(absl::Span<const absl::string_view> pieces,
                              SentencePieceText* spt) const;

  virtual absl::Status Decode(absl::Span<const int> ids,
                              SentencePieceText* spt) const;

  //////////////////////////////////////////////////////////////
  // API methods for encoding sequences in parallel.
  // This is particularly useful for long inputs.

  // chunk_len controls how long each chunk to be tokenized in parallel is.
  // For best results, set this to ~10000.
  virtual absl::Status ParallelEncode(absl::string_view input, int chunk_len,
                                      ThreadPool& thread_pool,
                                      std::vector<std::string>* pieces) const;
  virtual absl::Status ParallelEncode(absl::string_view input, int chunk_len,
                                      ThreadPool& thread_pool,
                                      std::vector<int>* ids) const;
  virtual absl::Status ParallelEncode(absl::string_view input, int chunk_len,
                                      ThreadPool& thread_pool,
                                      SentencePieceText* spt) const;

#define DEFINE_SPP_DIRECT_FUNC_IMPL(FuncName, OutType, ...) \
  OutType output;                                           \
  const auto status = FuncName(__VA_ARGS__, &output);       \
  return output;

  //////////////////////////////////////////////////////////////
  // Handy methods that return the result directly.
  // These functions ignore internal errors.
  [[nodiscard]] virtual std::vector<std::string> EncodeAsPieces(
      absl::string_view input) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Encode, std::vector<std::string>, input);
  }

  [[nodiscard]] virtual std::vector<int> EncodeAsIds(
      absl::string_view input) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Encode, std::vector<int>, input);
  }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  [[deprecated(
      "NBestEncodeAsPieces is deprecated; use SampleEncodeAsPieces "
      "instead.")]] [[nodiscard]] virtual std::vector<std::vector<std::string>>
  NBestEncodeAsPieces(absl::string_view input, int nbest_size) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(
        NBestEncode, std::vector<std::vector<std::string>>, input, nbest_size);
  }

  [[deprecated(
      "NBestEncodeAsIds is deprecated; use SampleEncodeAsIds "
      "instead.")]] [[nodiscard]] virtual std::vector<std::vector<int>>
  NBestEncodeAsIds(absl::string_view input, int nbest_size) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(NBestEncode, std::vector<std::vector<int>>,
                                input, nbest_size);
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  [[nodiscard]] virtual std::vector<std::string> SampleEncodeAsPieces(
      absl::string_view input, int nbest_size, float alpha) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(SampleEncode, std::vector<std::string>, input,
                                nbest_size, alpha);
  }

  [[nodiscard]] virtual std::vector<int> SampleEncodeAsIds(
      absl::string_view input, int nbest_size, float alpha) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(SampleEncode, std::vector<int>, input,
                                nbest_size, alpha);
  }

  virtual std::vector<std::string> ParallelEncodeAsPieces(
      absl::string_view input, int chunk_len, ThreadPool& therad_pool) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(ParallelEncode, std::vector<std::string>, input,
                                chunk_len, therad_pool);
  }

  virtual std::vector<int> ParallelEncodeAsIds(absl::string_view input,
                                               int chunk_len,
                                               ThreadPool& therad_pool) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(ParallelEncode, std::vector<int>, input,
                                chunk_len, therad_pool);
  }

  [[nodiscard]] virtual std::string DecodePieces(
      const std::vector<absl::string_view>& pieces) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Decode, std::string, pieces);
  }

  [[nodiscard]] virtual std::string DecodeIds(
      const std::vector<int>& ids) const {
    DEFINE_SPP_DIRECT_FUNC_IMPL(Decode, std::string, ids);
  }

#undef DEFINE_SPP_DIRECT_FUNC_IMPL

  //////////////////////////////////////////////////////////////
  // Normalization methods.

  // Normalize `input`.
  virtual absl::Status Normalize(absl::string_view input,
                                 std::string* normalized) const;

  // Normalize `input`. Stores the utf8-byte offset from
  // the normalized string to the original input.
  virtual absl::Status Normalize(absl::string_view input,
                                 std::string* normalized,
                                 std::vector<size_t>* norm_to_orig) const;

  [[nodiscard]] virtual std::string Normalize(absl::string_view input) const;

  //////////////////////////////////////////////////////////////
  // Vocabulary management methods.
  //
  // Returns the size of sentence pieces, which is the same as
  // the size of vocabulary for NMT.
  [[nodiscard]] virtual int GetPieceSize() const;

  // Returns the vocab id of `piece`.
  // Returns `unk_id()` if `piece` is unknown, or -1 if no model is loaded.
  [[nodiscard]] virtual int PieceToId(absl::string_view piece) const;

  // Returns the string representation of vocab with `id`.
  // Returns "<unk>" if the ID is out of bounds or if no model is loaded.
  [[nodiscard]] virtual absl::string_view IdToPiece(int id) const;

  // Returns the string representation of vocab with `id`.
  // Returns false when id is out of range.
  virtual bool SafeIdToPiece(int id, std::string* piece) const;

  // Returns the score of `id`.
  // Usually score is an emission log probability of unigram language
  // model.
  [[nodiscard]] virtual float GetScore(int id) const;

  // Returns true if `id` is unknown symbol.
  [[nodiscard]] virtual bool IsUnknown(int id) const;

  // Returns true if `id` is control symbol.
  [[nodiscard]] virtual bool IsControl(int id) const;

  // Returns true if `id` is unused symbol.
  [[nodiscard]] virtual bool IsUnused(int id) const;

  // Returns true if `id` is byte symbol.
  [[nodiscard]] virtual bool IsByte(int id) const;

  // Returns the reserved id.
  // Returns -1 if not defined.
  //
  // Note: Valid IDs are returned only when they are strictly defined as
  // CONTROL tokens (or UNKNOWN for unk_id). If they are defined as
  // USER_DEFINED, these methods will return -1, as USER_DEFINED symbols
  // are treated as normal symbols (protected from segmentation) rather
  // than strict special control symbols.
  //
  // Consequently, encoding extra options (like "bos" / "eos") and Python
  // wrapper flags (like add_bos=True / add_eos=True) will return an error if
  // the corresponding tokens are not strictly defined as CONTROL tokens.

  // Returns unknown (<unk>) id.
  [[nodiscard]] virtual int unk_id() const;

  // Returns BOS (<s>) id.
  [[nodiscard]] virtual int bos_id() const;

  // Returns EOS (</s>) id.
  [[nodiscard]] virtual int eos_id() const;

  // Returns PAD (<pad>) id.
  [[nodiscard]] virtual int pad_id() const;

  // Returns immutable model proto. Useful to obtain extended
  // or experimental parameters encoded in model_proto.
  [[nodiscard]] const ModelProto& model_proto() const;

  // returns immutable model proto as std::string.
  // Useful to save the state of this instance via Python's pickle object.
  [[nodiscard]] util::bytes serialized_model_proto() const;

  // Returns a string_view pointing to the underlying FlatBuffer model binary,
  // or "" if no model is loaded.
  [[nodiscard]] absl::string_view serialized_flatbuffer() const;

 private:
  enum ExtraOption { REVERSE, BOS, EOS, UNK_PIECE };

  absl::Status FinishLoad();

  absl::Status ParseExtraOptions(absl::string_view extra_option,
                                 std::vector<ExtraOption>* extra_options) const;

  template <typename T>
  absl::Status ApplyExtraOptions(absl::Span<const ExtraOption> extra_options,
                                 T* output) const;

  absl::Status DecodePiecesInternal(absl::Span<const absl::string_view> pieces,
                                    std::string* detokenized,
                                    SentencePieceText* spt) const;

  bool HasUnkPieceOption() const;

  absl::Status EncodeNormalizedInternal(
      absl::string_view normalized, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr) const;

  absl::Status EncodeWordNormalized(
      absl::string_view normalized, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr) const;

  void PopulatePieceStrings(absl::Span<const int> ids,
                            absl::Span<const std::string_view> piece_views,
                            std::vector<std::string>* pieces) const;

  absl::Status PopulateSentencePieceText(
      absl::string_view input, absl::string_view normalized,
      absl::Span<const size_t> norm_to_orig, absl::Span<const int> ids,
      absl::Span<const std::string_view> pieces, SentencePieceText* spt) const;

  absl::Status ParallelEncodeInternal(absl::string_view input, size_t chunk_len,
                                      ThreadPool& thread_pool,
                                      std::vector<std::string>* pieces,
                                      std::vector<int>* ids,
                                      SentencePieceText* spt) const;

  static constexpr absl::string_view kDefaultUnkPiece = "<unk>";
  static constexpr absl::string_view kDefaultBosPiece = "<s>";
  static constexpr absl::string_view kDefaultEosPiece = "</s>";
  static constexpr absl::string_view kDefaultUnkSurface = " \xE2\x81\x87 ";

  std::unique_ptr<class ModelProtoManager> model_proto_manager_;
  const lite::SentencePieceLiteProcessor* lite_processor_ = nullptr;

  int model_type_ = 1;
  absl::string_view unk_piece_ = kDefaultUnkPiece;
  absl::string_view bos_piece_ = kDefaultBosPiece;
  absl::string_view eos_piece_ = kDefaultEosPiece;
  absl::string_view unk_surface_ = kDefaultUnkSurface;
  bool byte_fallback_ = false;
  int byte0_id_ = -1;
  bool treat_whitespace_as_suffix_ = false;
  bool add_dummy_prefix_ = true;
  bool remove_extra_whitespaces_ = true;

  std::vector<ExtraOption> encode_extra_options_;
  std::vector<ExtraOption> decode_extra_options_;

  absl::Status status_ =
      absl::FailedPreconditionError("Model is not initialized.");
};

// Set seed value of random generator.
// Do not set static_cast<unique_int>(-1),
// as this seed is reserved for initializing from
// std::random_device.
void SetRandomGeneratorSeed(unsigned int seed);

// Set the global log level. The default loglevel is 0.
// The log is emitted only when min_log_level >= output_log_level.
void SetMinLogLevel(int v);

// IO related functions to absorb model formats.
namespace io {
// Loads `model_proto` from `filename`.
// We can instantiate SentencePieceProcessor as follows:
//
//  auto model_proto = absl::make_unique<ModelProto>();
//  io::LoadModelProto("//path/spm.model", model_proto.get());
//  SentencePieceProcessor sp;
//  CHECK_OK(sp.Load(std::move(model_proto)));
absl::Status LoadModelProto(absl::string_view, ModelProto* model_proto);

// Saves `model_proto` as `filename`.
absl::Status SaveModelProto(absl::string_view, const ModelProto& model_proto);
}  // namespace io
}  // namespace sentencepiece
#endif  // SENTENCEPIECE_PROCESSOR_H_
