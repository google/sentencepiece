// Copyright 2026 Google Inc.
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

#ifndef MODEL_PROTO_MANAGER_H_
#define MODEL_PROTO_MANAGER_H_

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece {

inline absl::Status ToAbslStatus(lite::StatusCode code,
                                 absl::string_view message = "") {
  if (code == lite::StatusCode::kOk) {
    return absl::OkStatus();
  }
  return absl::Status(static_cast<absl::StatusCode>(code), message);
}

// Manages ModelProto validation, FlatBuffer compilation, and on-demand
// ModelProto reconstruction for SentencePieceProcessor.
//
// Only `trainer_spec` and `normalizer_spec` (with `precompiled_charsmap`
// stripped and referenced as an `absl::string_view` directly from the
// FlatBuffer on little-endian hosts) are retained in memory after
// initialization; the full `ModelProto` (including vocabulary `pieces` from
// `SentencePieceLiteProcessor::IdToPiece()`, `GetScore()`, and `piece_type()`)
// is reconstructed on-the-fly and cached when `model_proto()` or
// `serialized_model_proto()` is requested.
class ModelProtoManager {
 public:
  ModelProtoManager() = default;
  ~ModelProtoManager() = default;

  ModelProtoManager(const ModelProtoManager&) = delete;
  ModelProtoManager& operator=(const ModelProtoManager&) = delete;

  // Validates `model_proto`, retains `trainer_spec` and `normalizer_spec`
  // (excluding `precompiled_charsmap`), compiles the FlatBuffer model, and
  // initializes `SentencePieceLiteProcessor`.
  absl::Status Load(std::unique_ptr<ModelProto> model_proto);

  // Loads from a serialized FlatBuffer model binary by copying `serialized`
  // into an owned buffer.
  absl::Status LoadFromSerializedFlatbuffer(absl::string_view serialized);

  // Loads from a FlatBuffer memory view in zero-copy mode without copying the
  // underlying buffer. The caller must guarantee that `flatbuffer_view` remains
  // valid for the lifetime of this manager (until `Reset()` or destruction).
  // If `flatbuffer_view` is not 8-byte aligned, it falls back to
  // `LoadFromSerializedFlatbuffer()` and copies the buffer.
  absl::Status LoadFromFlatbufferView(absl::string_view flatbuffer_view);

  // Resets all managed state.
  void Reset();

  // Returns the initialized `SentencePieceLiteProcessor`, or nullptr if not
  // loaded.
  const lite::SentencePieceLiteProcessor* lite_processor() const {
    return lite_processor_.get();
  }

  const TrainerSpec& trainer_spec() const { return trainer_spec_; }
  const NormalizerSpec& normalizer_spec() const { return normalizer_spec_; }
  bool has_normalizer_spec() const { return has_normalizer_spec_; }
  float min_score() const { return min_score_; }

  // Returns a view of `precompiled_charsmap` pointing directly into the
  // FlatBuffer (or into `big_endian_charsmap_` on big-endian hosts).
  absl::string_view precompiled_charsmap() const {
    return precompiled_charsmap_;
  }

  // Returns a view of the underlying FlatBuffer model binary, or "" if not
  // loaded.
  absl::string_view serialized_flatbuffer() const { return flatbuffer_view_; }

  // Reconstructs and caches the full `ModelProto` on-the-fly from the retained
  // specs, `precompiled_charsmap_`, and `lite_processor_`. Thread-safe.
  const ModelProto& model_proto() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Serializes the reconstructed `ModelProto`, or returns "" if not loaded.
  std::string serialized_model_proto() const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  static absl::Status ValidateModel(ModelProto* model_proto, int* unk_id);
  static void MaybeRewriteModelProto(ModelProto* model_proto, int unk_id);
  void PopulateMinScore();
  void PopulatePrecompiledCharsmapFromFlatbuffer(absl::string_view fb_bytes);
  void PopulateMetadataFromFlatbuffer(absl::string_view fb_bytes);
  absl::Status FinalizeFlatbufferLoad(absl::string_view fb_view,
                                      absl::string_view error_msg);

  std::unique_ptr<lite::SentencePieceLiteProcessor> lite_processor_;
  TrainerSpec trainer_spec_;
  NormalizerSpec normalizer_spec_;
  absl::string_view flatbuffer_view_;
  absl::string_view precompiled_charsmap_;
  std::string big_endian_charsmap_;
  float min_score_ = 0.0f;
  bool has_trainer_spec_ = false;
  bool has_normalizer_spec_ = false;
  bool has_precompiled_charsmap_ = false;

  mutable absl::Mutex mutex_;
  mutable std::unique_ptr<ModelProto> cached_model_proto_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace sentencepiece

#endif  // MODEL_PROTO_MANAGER_H_
