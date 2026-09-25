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

#ifndef SENTENCEPIECE_LITE_SENTENCEPIECE_MODEL_CONVERTERS_H_
#define SENTENCEPIECE_LITE_SENTENCEPIECE_MODEL_CONVERTERS_H_

#include <string>

#include "absl/status/statusor.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::lite {

// Options for converting ModelProto to FlatBuffers.
struct ConverterOptions {
  // If true, skips building the character bigram trie during conversion.
  // - In BPE mode: This option is ignored (bigrams are always built).
  // - In Unigram mode: Setting this to true speeds up model conversion and
  //   reduces model size. However, pre-tokenization
  //   (PretokenizeAtSafeBoundaries) becomes unavailable because it relies on
  //   the character bigram trie.
  bool skip_char_bigrams = false;

  // If true, automatically converts pieces containing null bytes ('\0') into
  // UNUSED pieces with dummy names (<unused_ID>), instead of failing with an
  // InvalidArgument error.
  // Note: This option is only valid when byte fallback is enabled
  // (byte_fallback = true) in the model. If byte fallback is disabled,
  // conversion will fail with an error because null bytes cannot be safely
  // delegated to byte tokens (<0x00>).
  bool treat_null_byte_as_unused = false;
};

// Converts a standard SentencePiece ModelProto into a serialized FlatBuffers
// binary blob (std::string) suitable for SentencePieceLiteProcessor.
//
// Performance note:
// 1. Initialization / Load Time:
//    On-the-fly conversion is fast enough for production initialization and is
//    equivalent in speed (or even faster when skip_char_bigrams = true) to
//    standard OSS SentencePieceProcessor::LoadFromSerializedProto(proto) for
//    Unigram models. For BPE models, conversion includes precomputing direct
//    mapping shortcuts across all pieces.
//    - For large 256k Unigram models (e.g., ulm_spm.256k):
//      OSS Load: ~647 ms vs Lite Fast Init: ~624 ms (faster than OSS!).
//    - For large 262k BPE models (e.g., Gemma 3 262k):
//      OSS Load: ~211 ms vs Lite Fast Init: ~1,606 ms (includes shortcut
//      precomp).
//
// 2. Memory / Heap Consumption:
//    Although the serialized FlatBuffers blob (*fb_model_or) is larger than the
//    raw ModelProto file (due to precomputed Darts tries and direct mappings),
//    the actual runtime heap size in SentencePieceLiteProcessor is roughly
//    equal to the FlatBuffers blob size because it operates in zero-copy mode.
//    This is significantly smaller (nearly 3x to 4x smaller) than the normal
//    OSS runtime heap!
//    - For 256k Unigram models (ulm_spm.256k):
//      Raw Proto: 4.5 MB | Lite FB Heap: 9.6 MB | OSS Runtime Heap: 27.7 MB
//      (65% reduction!).
//    - For 262k BPE models (Gemma 3 262k):
//      Raw Proto: 4.6 MB | Lite FB Heap: 10.3 MB | OSS Runtime Heap: 39.8 MB
//      (74% reduction!).
//
// Sample usage (On-the-Fly Conversion):
// ```cpp
//   #include <memory>
//   #include "sentencepiece_lite.h"
//   #include "sentencepiece_model_converters.h"
//
//   // 1. Load or receive standard ModelProto
//   ::sentencepiece::ModelProto proto;
//   CHECK(proto.ParseFromString(model_bytes));
//
//   // 2. Convert on-the-fly to FlatBuffers (using default options)
//   auto fb_model_or = ::sentencepiece::lite::ToFlatbuffer(proto);
//   CHECK_OK(fb_model_or.status());
//
//   // 3. Initialize processor.
//   // Note: Since SentencePieceLiteProcessor operates in zero-copy mode
//   without
//   // copying the underlying buffer, the FlatBuffers string must remain
//   // persistent and outlive the processor. Moving the buffer into a
//   // std::shared_ptr is recommended for safe lifecycle management.
//   auto shared_fb_model = std::make_shared<std::string>(
//       std::move(*fb_model_or));
//   ::sentencepiece::lite::SentencePieceLiteProcessor processor(
//       shared_fb_model);
//   CHECK_EQ(processor.status(), ::sentencepiece::lite::StatusCode::kOk);
//
//   // 4. Tokenize
//   std::vector<int> ids;
//   CHECK_EQ(processor.Encode("Hello world", &ids),
//            ::sentencepiece::lite::StatusCode::kOk);
// ```
absl::StatusOr<std::string> ToFlatbuffer(
    const ::sentencepiece::ModelProto& proto,
    const ConverterOptions& options = {});

}  // namespace sentencepiece::lite

#endif  // SENTENCEPIECE_LITE_SENTENCEPIECE_MODEL_CONVERTERS_H_
