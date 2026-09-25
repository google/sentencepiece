// Copyright 2026 Google LLC
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
// limitations under the License.

#ifndef SENTENCEPIECE_LITE_SENTENCEPIECE_LITE_H_
#define SENTENCEPIECE_LITE_SENTENCEPIECE_LITE_H_

// The SentencePiece Lite Runtime is a lightweight, dependency-free C++ library
// that executes tokenization directly on FlatBuffers model binaries (.spm.fb).
// It supports Unigram and BPE tokenization, byte fallback, and full text
// normalization.
//
// Key Design Highlights:
// * Zero Dependencies: Depends only on FlatBuffers and standard C++20.
// * Zero-Copy Startup: Maps and reads model vocabulary, scores, and tries in
//   place without heap allocation or parsing overhead.
// * Stateless & Thread-Safe: The processor instance is read-only after
//   initialization, making it fully thread-safe across concurrent threads.
// * Minimalist Integer API: High-performance text encoding/decoding operating
//   directly on string views and integer token vectors.
//
// Development & Contributing Rules:
// 1. Minimalism Design: Focus on core functionality required for tokenization.
//    Avoid introducing fancy or non-essential features (e.g., operations that
//    can be implemented purely in post-processing or pre-processing outside
//    this runtime).
// 2. Zero External Dependencies: Do not introduce extra dependencies such as
//    Abseil or Protobuf.
// 3. Minimal Initialization Allocation: Minimize heap allocations during model
//    initialization (avoid allocating large containers like std::vector or
//    std::map).
// 4. Performance Verification: Always run performance benchmarks to ensure no
//    regressions are introduced.
// 5. Simplicity & Cleanliness: Avoid over-engineering, and keep the codebase
//    clean, concise, and simple.

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef SENTENCEPIECE_LITE_USE_ABSL
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#endif

namespace sentencepiece::lite {

// StatusCode is a lightweight replacement for absl::StatusCode.
// The underlying integer values are identical to the corresponding values
// in absl::StatusCode.
// To convert a StatusCode `code` to absl::Status, you can use:
//   absl::Status(static_cast<absl::StatusCode>(code), "error message");
enum class StatusCode : int {
  kOk = 0,
  kInvalidArgument = 3,
  kFailedPrecondition = 9,
  kInternal = 13,
};

#ifdef SENTENCEPIECE_LITE_USE_ABSL
template <typename T>
using Span = absl::Span<T>;

template <typename Signature>
using FunctionRef = absl::FunctionRef<Signature>;
#else
// A lightweight, zero-dependency implementation of a span.
// We avoid absl::Span and std::span to keep the public API of this "lite"
// library completely dependency-free across C++ standards.
template <typename T>
class Span {
 public:
  Span() = default;
  Span(T* data, size_t size) : data_(data), size_(size) {}

  template <size_t N>
  Span(T (&array)[N])  // NOLINT(google-explicit-constructor)
      : data_(array), size_(N) {}

  template <typename Container,
            typename = decltype(std::declval<Container&>().data())>
  Span(Container& c)  // NOLINT(google-explicit-constructor)
      : data_(c.data()), size_(c.size()) {}

  template <typename Container,
            typename = decltype(std::declval<const Container&>().data())>
  Span(const Container& c)  // NOLINT(google-explicit-constructor)
      : data_(c.data()), size_(c.size()) {}

  template <typename Container,
            typename = decltype(std::declval<const Container&>().data())>
  Span(const Container&& c) = delete;

  T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

  T& operator[](size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

// A lightweight, zero-dependency, non-owning function reference for C++17.
// Replaces std::function and absl::FunctionRef without heap allocation.
template <typename Signature>
class FunctionRef;

template <typename ReturnType, typename... Args>
class FunctionRef<ReturnType(Args...)> {
 public:
  template <typename Callable,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Callable>, FunctionRef> &&
                std::is_invocable_r_v<ReturnType, Callable, Args...> > >
  constexpr FunctionRef(Callable&& callable) noexcept
      : callable_ptr_(reinterpret_cast<intptr_t>(&callable)),
        invoker_([](intptr_t ptr, Args... args) -> ReturnType {
          return (*reinterpret_cast<std::add_pointer_t<Callable> >(ptr))(
              std::forward<Args>(args)...);
        }) {}

  ReturnType operator()(Args... args) const {
    return invoker_(callable_ptr_, std::forward<Args>(args)...);
  }

 private:
  intptr_t callable_ptr_ = 0;
  ReturnType (*invoker_)(intptr_t, Args...) = nullptr;
};
#endif

class Model;

// Main tokenization runtime processor executing Unigram and BPE tokenization
// on FlatBuffers model binaries (.spm.fb).
class SentencePieceLiteProcessor;
using SentencePieceLite = SentencePieceLiteProcessor;

class SentencePieceLiteProcessor {
 public:
  SentencePieceLiteProcessor() = delete;

  // Constructs the processor and loads the model from the memory buffer.
  // Note: The memory buffer must remain valid and outlive this processor
  // instance, as the processor accesses the buffer directly without copying.
  explicit SentencePieceLiteProcessor(std::string_view buffer);

  // Constructs the processor by retaining shared ownership of the model buffer.
  explicit SentencePieceLiteProcessor(
      std::shared_ptr<std::string> shared_buffer);
  ~SentencePieceLiteProcessor();

  // Returns the status of the model loading/initialization.
  StatusCode status() const;

  // Normalizes text using model rules. If not nullptr, `(*offset)[x]` stores
  // the byte index in `input` corresponding to `output[x]` (with a sentinel at
  // `output->size()`).
  StatusCode Normalize(std::string_view input, std::string* output,
                       std::vector<size_t>* offset = nullptr) const;

  // Encode text to token IDs
  StatusCode Encode(std::string_view input, std::vector<int>* ids) const;

  // Encodes normalized text. `pieces` stores string views into `normalized` if
  // not nullptr. Assumes input is already normalized.
  StatusCode EncodeNormalized(
      std::string_view normalized, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr) const;

  // Encodes a single pre-tokenized, normalized chunk without running
  // `PretokenizeAtSafeBoundaries`. Unlike `EncodeNormalized`, this method
  // skips pre-tokenization and directly encodes the chunk.
  StatusCode EncodeNormalizedChunk(
      std::string_view normalized_chunk, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr) const;

  // Encodes normalized text using stochastic sampling for data augmentation
  // and subword regularization. When `uniform_sampler` is provided, sampling
  // mode is active across all non-negative `alpha` (>= 0.0) values:
  //
  // - Unigram (--model_type=unigram):
  //   `alpha`: Temperature noise scale controlling the magnitude of Gumbel
  //   perturbations (`score + alpha * Gumbel()`) where each segmentation `S`
  //   is sampled proportional to `P(S)^alpha` (Kudo, 2018,
  //   https://arxiv.org/abs/1804.10959). While the original Kudo (2018) paper
  //   uses Forward-Filtering and Backward-Sampling (FFBS), this lightweight
  //   runtime adopts the simpler Gumbel-Max/Softmax trick during Viterbi
  //   search, yielding theoretically identical stochastic behavior.
  //   * `alpha` == 0.0: Zero noise addition (`score + 0`), returning the best
  //     deterministic Viterbi segmentation.
  //   * `alpha` == 1.0: Corresponds to standard Gibbs posterior sampling
  //     without smoothing or sharpening.
  //   * `alpha` > 0.0: Larger `alpha` yields more diverse subword variations.
  //     Empirically recommended values are around 0.1 to 0.2 (Kudo, 2018).
  //
  // - BPE (--model_type=bpe):
  //   `alpha`: The dropout probability `p` in [0.0, 1.0] of BPE merge
  //   operations (Provilkov et al., 2020, https://arxiv.org/abs/1910.13267).
  //   * `alpha` == 0.0: Zero dropout, returning standard deterministic BPE.
  //   * `alpha` in (0.0, 1.0): Drops merge steps with probability `alpha`.
  //     Empirically recommended values are around 0.1 (Provilkov et al., 2020).
  //   * `alpha` >= 1.0: All merges are dropped, yielding character/base-symbol
  //     segmentation.
  //
  // `alpha` must be non-negative (`alpha` >= 0.0). `uniform_sampler` generates
  // uniform reals in [0.0, 1.0). `pieces` stores string views into `normalized`
  // if not nullptr. Assumes normalized input.
  StatusCode SampleNormalized(
      std::string_view normalized, float alpha,
      FunctionRef<float()> uniform_sampler, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr) const;

  // Pre-tokenizes normalized text into chunks at safe, non-overlapping boundary
  // points, invoking `receiver` for each chunk with zero heap allocation.
  StatusCode PretokenizeAtSafeBoundaries(
      std::string_view normalized,
      FunctionRef<void(std::string_view)> receiver) const;

  // Convenience overload that collects chunks into a std::vector.
  StatusCode PretokenizeAtSafeBoundaries(
      std::string_view normalized, std::vector<std::string_view>* out) const {
    if (out == nullptr) return StatusCode::kInvalidArgument;
    out->clear();
    return PretokenizeAtSafeBoundaries(
        normalized, [out](std::string_view chunk) { out->push_back(chunk); });
  }

  // Decodes token IDs to text. `pieces` stores string views into `output` if
  // not nullptr.
  StatusCode Decode(Span<const int> ids, std::string* output,
                    std::vector<std::string_view>* pieces = nullptr) const;

  // Returns the ID of the given piece.
  // Returns -1 if the piece is not found or if no model is loaded.
  int PieceToId(std::string_view piece) const;

  // Returns the string piece corresponding to the given ID.
  // Returns "<unk>" if the ID is out of bounds or if no model is loaded.
  std::string_view IdToPiece(int id) const;

  // Returns the score of the given ID.
  // Returns 0.0 if the ID is out of bounds or if no model is loaded.
  float GetScore(int id) const;

  // Returns the unk id of the sentence piece processor.
  // Returns -1 if disabled or undefined.
  int unk_id() const;

  // Returns the bos id of the sentence piece processor.
  // Returns -1 if disabled or undefined.
  int bos_id() const;

  // Returns the eos id of the sentence piece processor.
  // Returns -1 if disabled or undefined.
  int eos_id() const;

  // Returns the pad id of the sentence piece processor.
  // Returns -1 if disabled or undefined.
  int pad_id() const;

  // Returns the size of the vocabulary.
  // Returns 0 if no model is loaded.
  size_t vocab_size() const;

  // Returns the type of the piece corresponding to the given ID.
  // The returned value corresponds to the PieceType enum defined in the
  // FlatBuffers schema:
  // DEFAULT=0, NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
  // Returns -1 if the ID is out of bounds or if no model is loaded.
  int piece_type(int id) const;

  // For testing/internal usage only.
  bool HasNonNullDirectMappingVectorForTesting() const;
  void SetScoreResetThresholdForTesting(float threshold);

 private:
  std::shared_ptr<std::string> shared_buffer_;
  std::unique_ptr<Model> model_;
  StatusCode status_ = StatusCode::kFailedPrecondition;
};

}  // namespace sentencepiece::lite

#endif  // SENTENCEPIECE_LITE_SENTENCEPIECE_LITE_H_
