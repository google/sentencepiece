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
// limitations under the License.

#ifndef CORE_UPB_WRAPPER_H_
#define CORE_UPB_WRAPPER_H_

#include <cstdlib>
#include <cstring>
#include <functional>
#include <istream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"



// Include the upb generated headers
#include "sentencepiece.upb.h"
#include "sentencepiece_model.upb.h"
#include "upb_message_wrapper.h"

namespace sentencepiece {

class TrainerSpec;
class NormalizerSpec;
class SelfTestData;
class SentencePieceText;
class NBestSentencePieceText;
class ModelProto;
class NBestSentencePieceText_Sub;

namespace upb {
class ModelProtoWrapper;
}  // namespace upb

class ModelProto_SentencePiece {
 public:
  enum Type {
    NORMAL = sentencepiece_ModelProto_SentencePiece_NORMAL,
    UNKNOWN = sentencepiece_ModelProto_SentencePiece_UNKNOWN,
    CONTROL = sentencepiece_ModelProto_SentencePiece_CONTROL,
    USER_DEFINED = sentencepiece_ModelProto_SentencePiece_USER_DEFINED,
    BYTE = sentencepiece_ModelProto_SentencePiece_BYTE,
    UNUSED = sentencepiece_ModelProto_SentencePiece_UNUSED,
  };

  ModelProto_SentencePiece(upb::ModelProtoWrapper* parent, int index)
      : parent_(parent), index_(index) {}
  ModelProto_SentencePiece() : parent_(nullptr), index_(-1) {}

  inline ProtoStr piece() const;
  inline float score() const;
  inline Type type() const;
  inline void set_type(Type type);
  inline void set_piece(absl::string_view piece);
  inline void set_piece(const char* data, size_t size) {
    set_piece(absl::string_view(data, size));
  }
  inline void set_score(float score);

 private:
  upb::ModelProtoWrapper* parent_;
  int index_;
};

namespace upb {



class SentencePieceTextWrapper;
}  // namespace upb

class SentencePieceText_SentencePiece {
 public:
  SentencePieceText_SentencePiece(upb::SentencePieceTextWrapper* parent,
                                  int index)
      : parent_(parent), index_(index) {}
  SentencePieceText_SentencePiece() : parent_(nullptr), index_(-1) {}

  inline absl::string_view piece() const;
  inline void set_piece(absl::string_view piece);
  inline void set_piece(const char* data, size_t size);

  inline uint32_t id() const;
  inline void set_id(uint32_t id);

  inline absl::string_view surface() const;
  inline void set_surface(absl::string_view surface);
  inline void set_surface(const char* data, size_t size);

  inline uint32_t begin() const;
  inline void set_begin(uint32_t begin);

  inline uint32_t end() const;
  inline void set_end(uint32_t end);

  friend void swap(SentencePieceText_SentencePiece a,
                   SentencePieceText_SentencePiece b);

  static const SentencePieceText_SentencePiece& default_instance() {
    static SentencePieceText_SentencePiece instance(nullptr, -1);
    return instance;
  }

 private:
  upb::SentencePieceTextWrapper* parent_;
  int index_;
};

namespace upb {

// Forward declarations
class ModelProtoWrapper;
class SentencePieceTextWrapper;
class NBestSentencePieceTextWrapper;
class SelfTestDataWrapper;
class SelfTestData_SampleWrapper;



class TrainerSpecWrapper {
  friend class upb::ModelProtoWrapper;

 public:
  TrainerSpecWrapper() {
    upb_Arena* arena = upb_Arena_New();
    holder_.Reset(sentencepiece_TrainerSpec_new(arena), arena, true);
  }
  explicit TrainerSpecWrapper(const sentencepiece_TrainerSpec* msg,
                              upb_Arena* arena = nullptr)
      : holder_(const_cast<sentencepiece_TrainerSpec*>(msg), arena, false) {}

  TrainerSpecWrapper(
      const sentencepiece_TrainerSpec* msg, upb_Arena* arena,
      std::function<void(const sentencepiece_TrainerSpec*)> on_change)
      : holder_(const_cast<sentencepiece_TrainerSpec*>(msg), arena, false),
        on_change_(on_change) {}

  TrainerSpecWrapper(const TrainerSpecWrapper& other) {
    upb_Arena* arena = upb_Arena_New();
    holder_.Reset(sentencepiece_TrainerSpec_new(arena), arena, true);
    CopyFrom(other);
  }

  DEFINE_UPB_ASSIGNMENT_OPERATOR(TrainerSpecWrapper, sentencepiece_TrainerSpec)

  virtual ~TrainerSpecWrapper() = default;

  DEFINE_UPB_SERIALIZATION_METHODS(TrainerSpecWrapper,
                                   sentencepiece_TrainerSpec)

  void OnArenaReset() {}

  enum ModelType {
    UNIGRAM = sentencepiece_TrainerSpec_UNIGRAM,
    BPE = sentencepiece_TrainerSpec_BPE,
    WORD = sentencepiece_TrainerSpec_WORD,
    CHAR = sentencepiece_TrainerSpec_CHAR,
  };

  DEFINE_UPB_REPEATED_STRING_ACCESSOR(input, sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(input_format, "", sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(model_prefix, "", sentencepiece_TrainerSpec)
  DEFINE_UPB_ENUM_ACCESSOR(model_type, ModelType, UNIGRAM,
                           sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(vocab_size, int32_t, 8000,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_REPEATED_STRING_ACCESSOR(accept_language,
                                      sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(self_test_sample_size, int32_t, 0,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(enable_differential_privacy, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(differential_privacy_noise_level, float, 0.0f,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(differential_privacy_clipping_threshold,
                                uint64_t, 0, sentencepiece_TrainerSpec)

  DEFINE_UPB_PRIMITIVE_ACCESSOR(character_coverage, float, 0.9995f,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(input_sentence_size, uint64_t, 0,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(shuffle_input_sentence, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(seed_sentencepiece_size, int32_t, 1000000,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(shrinking_factor, float, 0.75f,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(max_sentence_length, int32_t, 4192,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(num_threads, int32_t, 16,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(num_sub_iterations, int32_t, 2,
                                sentencepiece_TrainerSpec)

  DEFINE_UPB_PRIMITIVE_ACCESSOR(max_sentencepiece_length, int32_t, 16,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(split_by_unicode_script, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(split_by_number, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(split_by_whitespace, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(treat_whitespace_as_suffix, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(allow_whitespace_only_pieces, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(split_digits, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(pretokenization_delimiter, "",
                             sentencepiece_TrainerSpec)

  DEFINE_UPB_REPEATED_STRING_ACCESSOR(control_symbols,
                                      sentencepiece_TrainerSpec)
  DEFINE_UPB_REPEATED_STRING_ACCESSOR(user_defined_symbols,
                                      sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(required_chars, "", sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(byte_fallback, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(vocabulary_output_piece_score, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(hard_vocab_limit, bool, true,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(use_all_vocab, bool, false,
                                sentencepiece_TrainerSpec)

  DEFINE_UPB_PRIMITIVE_ACCESSOR(unk_id, int32_t, 0, sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(bos_id, int32_t, 1, sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(eos_id, int32_t, 2, sentencepiece_TrainerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(pad_id, int32_t, -1, sentencepiece_TrainerSpec)

  DEFINE_UPB_STRING_ACCESSOR(unk_piece, "<unk>", sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(bos_piece, "<s>", sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(eos_piece, "</s>", sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(pad_piece, "<pad>", sentencepiece_TrainerSpec)

  DEFINE_UPB_HAS_FIELD_ACCESSOR(unk_surface, sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(unk_surface, " \xE2\x81\x87 ",
                             sentencepiece_TrainerSpec)

  DEFINE_UPB_PRIMITIVE_ACCESSOR(train_extremely_large_corpus, bool, false,
                                sentencepiece_TrainerSpec)
  DEFINE_UPB_STRING_ACCESSOR(seed_sentencepieces_file, "",
                             sentencepiece_TrainerSpec)

 private:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_TrainerSpec> holder_;
  std::function<void(const sentencepiece_TrainerSpec*)> on_change_;
};

class NormalizerSpecWrapper {
  friend class upb::ModelProtoWrapper;

 public:
  NormalizerSpecWrapper() {
    upb_Arena* arena = upb_Arena_New();
    holder_.Reset(sentencepiece_NormalizerSpec_new(arena), arena, true);
  }
  explicit NormalizerSpecWrapper(const sentencepiece_NormalizerSpec* msg,
                                 upb_Arena* arena = nullptr)
      : holder_(const_cast<sentencepiece_NormalizerSpec*>(msg), arena, false) {}

  NormalizerSpecWrapper(
      const sentencepiece_NormalizerSpec* msg, upb_Arena* arena,
      std::function<void(const sentencepiece_NormalizerSpec*)> on_change)
      : holder_(const_cast<sentencepiece_NormalizerSpec*>(msg), arena, false),
        on_change_(on_change) {}

  NormalizerSpecWrapper(const NormalizerSpecWrapper& other) {
    upb_Arena* arena = upb_Arena_New();
    holder_.Reset(sentencepiece_NormalizerSpec_new(arena), arena, true);
    CopyFrom(other);
  }

  DEFINE_UPB_ASSIGNMENT_OPERATOR(NormalizerSpecWrapper, sentencepiece_NormalizerSpec)

  virtual ~NormalizerSpecWrapper() = default;

  DEFINE_UPB_SERIALIZATION_METHODS(NormalizerSpecWrapper,
                                   sentencepiece_NormalizerSpec)

  void OnArenaReset() {}

  DEFINE_UPB_STRING_ACCESSOR(name, "", sentencepiece_NormalizerSpec)
  DEFINE_UPB_STRING_ACCESSOR(precompiled_charsmap, "",
                             sentencepiece_NormalizerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(add_dummy_prefix, bool, true,
                                sentencepiece_NormalizerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(remove_extra_whitespaces, bool, true,
                                sentencepiece_NormalizerSpec)
  DEFINE_UPB_PRIMITIVE_ACCESSOR(escape_whitespaces, bool, true,
                                sentencepiece_NormalizerSpec)
  DEFINE_UPB_STRING_ACCESSOR(normalization_rule_tsv, "",
                             sentencepiece_NormalizerSpec)

 private:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_NormalizerSpec> holder_;
  std::function<void(const sentencepiece_NormalizerSpec*)> on_change_;
};

class SelfTestData_SampleWrapper {
 public:
  explicit SelfTestData_SampleWrapper(
      const sentencepiece_SelfTestData_Sample* msg)
      : holder_(const_cast<sentencepiece_SelfTestData_Sample*>(msg), nullptr, false) {}
  SelfTestData_SampleWrapper(sentencepiece_SelfTestData_Sample* msg,
                             upb_Arena* arena)
      : holder_(msg, arena, false) {}
  DEFINE_UPB_STRING_ACCESSOR(input, "", sentencepiece_SelfTestData_Sample)
  DEFINE_UPB_STRING_ACCESSOR(expected, "", sentencepiece_SelfTestData_Sample)

 private:
  sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_SelfTestData_Sample> holder_;
  std::function<void(const sentencepiece_SelfTestData_Sample*)> on_change_;
};

class SelfTestDataSamplesRepeatedWrapper : public ::sentencepiece::upb_internal::ConstRepeatedWrapperBase<
    SelfTestDataSamplesRepeatedWrapper, SelfTestData_SampleWrapper> {
 public:
  SelfTestDataSamplesRepeatedWrapper(
      const sentencepiece_SelfTestData_Sample* const* elements, size_t count)
      : elements_(elements), count_(count) {}

  SelfTestData_SampleWrapper Get(int index) const {
    return SelfTestData_SampleWrapper(elements_[index]);
  }
  int Size() const { return count_; }

 private:
  const sentencepiece_SelfTestData_Sample* const* elements_;
  size_t count_;
};

class SelfTestDataWrapper {
 public:
  explicit SelfTestDataWrapper(const sentencepiece_SelfTestData* msg)
      : holder_(const_cast<sentencepiece_SelfTestData*>(msg), nullptr, false) {}
  SelfTestDataWrapper(sentencepiece_SelfTestData* msg, upb_Arena* arena)
      : holder_(msg, arena, false) {}

  using SamplesRepeatedWrapper = SelfTestDataSamplesRepeatedWrapper;

  SelfTestDataSamplesRepeatedWrapper samples() const {
    size_t size = 0;
    const sentencepiece_SelfTestData_Sample* const* elements = nullptr;
    const auto* m = static_cast<const sentencepiece_SelfTestData*>(holder_.msg());
    if (m) {
      elements = sentencepiece_SelfTestData_samples(m, &size);
    }
    return SelfTestDataSamplesRepeatedWrapper(elements, size);
  }
  int samples_size() const { return samples().size(); }
  SelfTestData_SampleWrapper* add_samples() {
    auto* m = static_cast<sentencepiece_SelfTestData*>(holder_.msg());
    if (m && holder_.arena()) {
      sentencepiece_SelfTestData_add_samples(m, holder_.arena());
      int new_index = samples_size() - 1;
      sample_wrappers_.resize(samples_size());
      size_t size;
      sentencepiece_SelfTestData_Sample** samples =
          sentencepiece_SelfTestData_mutable_samples(m, &size);
      sample_wrappers_[new_index] =
          std::make_unique<SelfTestData_SampleWrapper>(samples[new_index],
                                                       holder_.arena());
      return sample_wrappers_[new_index].get();
    }
    return nullptr;
  }

 private:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_SelfTestData> holder_;
  mutable std::vector<std::unique_ptr<SelfTestData_SampleWrapper>>
      sample_wrappers_;
};

class ModelProtoPiecesRepeatedWrapper;
class ModelProtoMutablePiecesRepeatedWrapper;

class ModelProtoWrapper {
 public:
  ModelProtoWrapper();

  explicit ModelProtoWrapper(const sentencepiece_ModelProto* msg,
                             upb_Arena* arena);

  virtual ~ModelProtoWrapper();

  ModelProtoWrapper(const ModelProtoWrapper& other) : ModelProtoWrapper() {
    CopyFrom(other);
  }
  ModelProtoWrapper& operator=(const ModelProtoWrapper& other) {
    if (this != &other) {
      Clear();
      CopyFrom(other);
    }
    return *this;
  }

  DEFINE_UPB_SERIALIZATION_METHODS(ModelProtoWrapper, sentencepiece_ModelProto)

  void OnArenaReset() {
    piece_wrappers_.clear();
    mutable_pieces_wrapper_.reset();
    mutable_normalizer_spec_.reset();
    mutable_trainer_spec_.reset();
    normalizer_spec_cache_.reset();
    denormalizer_spec_cache_.reset();
    mutable_denormalizer_spec_.reset();
    mutable_self_test_data_.reset();
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    if (m) {
      normalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
          sentencepiece_ModelProto_normalizer_spec(m), holder_.arena());
      denormalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
          sentencepiece_ModelProto_denormalizer_spec(m), holder_.arena());
    }
  }

  bool ParseFromIstream(std::istream* input) {
    std::string bytes((std::istreambuf_iterator<char>(*input)),
                      std::istreambuf_iterator<char>());
    return ParseFromString(bytes);
  }

  int pieces_size() const {
    if (!holder_.msg()) return 0;
    size_t size = 0;
    sentencepiece_ModelProto_pieces(static_cast<const sentencepiece_ModelProto*>(holder_.msg()), &size);
    return size;
  }

  inline ::sentencepiece::ModelProto_SentencePiece pieces(int index) const {
    return ::sentencepiece::ModelProto_SentencePiece(
        const_cast<ModelProtoWrapper*>(this), index);
  }
  inline ::sentencepiece::ModelProto_SentencePiece pieces(int index) {
    return ::sentencepiece::ModelProto_SentencePiece(this, index);
  }

  ModelProtoPiecesRepeatedWrapper pieces() const;
  ModelProtoMutablePiecesRepeatedWrapper* mutable_pieces();

  inline ModelProto_SentencePiece* mutable_pieces(int index) {
    UPB_WRAPPER_RET_VAL_IF_OOB(index, pieces_size(), nullptr);
    LazyInitPieceWrappersCache();
    return piece_wrappers_[index].get();
  }

  ProtoStr piece_at(int index) const {
    size_t size = 0;
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(m, &size);
    UPB_WRAPPER_RET_VAL_IF_OOB(index, size, "");
    return ProtoStr(
        sentencepiece_ModelProto_SentencePiece_piece(pieces[index]));
  }
  float score_at(int index) const {
    size_t size = 0;
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(m, &size);
    UPB_WRAPPER_RET_VAL_IF_OOB(index, size, 0.0f);
    return sentencepiece_ModelProto_SentencePiece_score(pieces[index]);
  }
  int type_at(int index) const {
    size_t size = 0;
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(m, &size);
    UPB_WRAPPER_RET_VAL_IF_OOB(index, size, 0);
    return sentencepiece_ModelProto_SentencePiece_type(pieces[index]);
  }
  void set_type_at(int index, int type) {
    auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
    if (m) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(m, &size);
      UPB_WRAPPER_RET_IF_OOB(index, size);
      sentencepiece_ModelProto_SentencePiece_set_type(
          pieces[index],
          static_cast<sentencepiece_ModelProto_SentencePiece_Type>(type));
    }
  }
  void set_piece_at(int index, absl::string_view piece) {
    auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
    if (m) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(m, &size);
      UPB_WRAPPER_RET_IF_OOB(index, size);
      sentencepiece_ModelProto_SentencePiece_set_piece(
          pieces[index], MakeUpbString(piece, holder_.arena()));
    }
  }
  void set_piece_at(int index, const std::string& piece) {
    set_piece_at(index, absl::string_view(piece));
  }
  void set_score_at(int index, float score) {
    auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
    if (m) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(m, &size);
      UPB_WRAPPER_RET_IF_OOB(index, size);
      sentencepiece_ModelProto_SentencePiece_set_score(pieces[index], score);
    }
  }

  inline bool has_trainer_spec() const {
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    return m && sentencepiece_ModelProto_has_trainer_spec(m);
  }
  inline TrainerSpec trainer_spec() const;
  inline ::sentencepiece::TrainerSpec* mutable_trainer_spec() const;

  inline bool has_normalizer_spec() const {
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    return m && sentencepiece_ModelProto_has_normalizer_spec(m);
  }
  inline const NormalizerSpec& normalizer_spec() const;
  inline ::sentencepiece::NormalizerSpec* mutable_normalizer_spec() const;

  inline bool has_denormalizer_spec() const {
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    return m && sentencepiece_ModelProto_has_denormalizer_spec(m);
  }
  inline const NormalizerSpec& denormalizer_spec() const;
  inline ::sentencepiece::NormalizerSpec* mutable_denormalizer_spec() const;
  inline void set_trainer_spec(const TrainerSpec& spec);
  inline void set_normalizer_spec(const NormalizerSpec& spec);
  inline void set_denormalizer_spec(const NormalizerSpec& spec);

  inline bool has_self_test_data() const {
    const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
    return m && sentencepiece_ModelProto_has_self_test_data(m);
  }
  inline SelfTestData self_test_data() const;
  inline ::sentencepiece::SelfTestData* mutable_self_test_data() const;

 protected:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_ModelProto> holder_;

  std::vector<std::unique_ptr<ModelProto_SentencePiece>> piece_wrappers_;
  std::unique_ptr<ModelProtoMutablePiecesRepeatedWrapper> mutable_pieces_wrapper_;
  mutable std::unique_ptr<NormalizerSpec> mutable_normalizer_spec_;
  mutable std::unique_ptr<TrainerSpec> mutable_trainer_spec_;
  mutable std::unique_ptr<NormalizerSpec> normalizer_spec_cache_;
  mutable std::unique_ptr<NormalizerSpec> denormalizer_spec_cache_;
  mutable std::unique_ptr<NormalizerSpec> mutable_denormalizer_spec_;
  mutable std::unique_ptr<SelfTestData> mutable_self_test_data_;

 private:
  template <typename SpecWrapperType, typename CSpecType,
            CSpecType* (*MutableFn)(sentencepiece_ModelProto*, upb_Arena*),
            void (*SetFn)(sentencepiece_ModelProto*, CSpecType*),
            const CSpecType* (*GetFn)(const sentencepiece_ModelProto*)>
  inline SpecWrapperType* GetOrCreateMutableSpec(
      std::unique_ptr<SpecWrapperType>& mutable_spec_cache,
      std::unique_ptr<SpecWrapperType>* spec_cache_ptr = nullptr) const {
    if (!mutable_spec_cache) {
      auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
      if (m) {
        CSpecType* sub_msg = MutableFn(m, holder_.arena());
        if (spec_cache_ptr && *spec_cache_ptr) {
          (*spec_cache_ptr)->holder_.Reset(sub_msg, holder_.arena(), false);
        }
        auto* non_const_this = const_cast<ModelProtoWrapper*>(this);
        mutable_spec_cache = std::make_unique<SpecWrapperType>(
            sub_msg, holder_.arena(),
            [non_const_this, spec_cache_ptr](const CSpecType* new_sub_msg) {
              auto* parent_m = static_cast<sentencepiece_ModelProto*>(non_const_this->holder_.msg());
              SetFn(parent_m, const_cast<CSpecType*>(new_sub_msg));
              if (spec_cache_ptr && *spec_cache_ptr) {
                auto* m = static_cast<sentencepiece_ModelProto*>(non_const_this->holder_.msg());
                auto* spec = const_cast<CSpecType*>(GetFn(m));
                (*spec_cache_ptr)->holder_.Reset(spec, non_const_this->holder_.arena(), false);
              }
            });
      } else {
        mutable_spec_cache = std::make_unique<SpecWrapperType>(nullptr);
      }
    }
    return mutable_spec_cache.get();
  }

  void LazyInitPieceWrappersCache() {
    ::sentencepiece::upb_internal::LazyInitCache(piece_wrappers_, this, pieces_size());
  }

};

class ModelProtoPiecesRepeatedWrapper : public ::sentencepiece::upb_internal::ConstRepeatedWrapperBase<
    ModelProtoPiecesRepeatedWrapper, ModelProto_SentencePiece> {
 public:
  ModelProtoPiecesRepeatedWrapper(const ModelProtoWrapper* parent) : parent_(parent) {}
  ModelProto_SentencePiece Get(int index) const { return parent_->pieces(index); }
  int Size() const { return parent_->pieces_size(); }
 private:
  const ModelProtoWrapper* parent_;
};

class ModelProtoMutablePiecesRepeatedWrapper : public ::sentencepiece::upb_internal::MutableRepeatedWrapperBase<
    ModelProtoMutablePiecesRepeatedWrapper, ModelProto_SentencePiece> {
 public:
  ModelProtoMutablePiecesRepeatedWrapper(ModelProtoWrapper* parent) : parent_(parent) {}
  ModelProto_SentencePiece* GetMutable(int index) { return parent_->mutable_pieces(index); }
  int Size() const { return parent_->pieces_size(); }
 private:
  ModelProtoWrapper* parent_;
};

inline ModelProtoPiecesRepeatedWrapper ModelProtoWrapper::pieces() const {
  return ModelProtoPiecesRepeatedWrapper(this);
}

inline ModelProtoMutablePiecesRepeatedWrapper*
ModelProtoWrapper::mutable_pieces() {
  if (!mutable_pieces_wrapper_) {
    mutable_pieces_wrapper_ =
        std::make_unique<ModelProtoMutablePiecesRepeatedWrapper>(this);
  }
  return mutable_pieces_wrapper_.get();
}

class SentencePieceTextPiecesRepeatedWrapper;
class SentencePieceTextMutablePiecesRepeatedWrapper;

class SentencePieceTextWrapper {
 public:
  SentencePieceTextWrapper();
  explicit SentencePieceTextWrapper(std::nullptr_t);

  virtual ~SentencePieceTextWrapper();

  SentencePieceTextWrapper(const SentencePieceTextWrapper& other)
      : SentencePieceTextWrapper() {
    CopyFrom(other);
  }
  SentencePieceTextWrapper& operator=(const SentencePieceTextWrapper& other) {
    if (this != &other) {
      Clear();
      CopyFrom(other);
    }
    return *this;
  }

  DEFINE_UPB_SERIALIZATION_METHODS(SentencePieceTextWrapper,
                                   sentencepiece_SentencePieceText)

  void OnArenaReset() {
    piece_wrappers_.clear();
    mutable_pieces_wrapper_.reset();
    if (holder_.msg()) {
      LazyInitPieceWrappersCache();
    }
  }

  virtual sentencepiece_SentencePieceText* mutable_msg() {
    return static_cast<sentencepiece_SentencePieceText*>(holder_.msg());
  }
  virtual const sentencepiece_SentencePieceText* msg() const {
    return static_cast<const sentencepiece_SentencePieceText*>(holder_.msg());
  }

  virtual upb_Arena* arena() { return holder_.arena(); }
  virtual const upb_Arena* arena() const { return holder_.arena(); }

  void clear_pieces() {
    if (mutable_msg()) {
      sentencepiece_SentencePieceText_clear_pieces(mutable_msg());
    }
    piece_wrappers_.clear();
  }

  absl::string_view text() const {
    if (!msg()) {
      return "";
    }
    upb_StringView sv = sentencepiece_SentencePieceText_text(msg());
    return absl::string_view(sv.data, sv.size);
  }
  void set_text(absl::string_view text) {
    sentencepiece_SentencePieceText_set_text(mutable_msg(),
                                             MakeUpbString(text, arena()));
  }
  void set_text(const char* data, size_t size) {
    set_text(absl::string_view(data, size));
  }

  float score() const {
    return msg() ? sentencepiece_SentencePieceText_score(msg()) : 0.0;
  }
  void set_score(float score) {
    sentencepiece_SentencePieceText_set_score(mutable_msg(), score);
  }

  int pieces_size() const {
    const auto* m = msg();
    if (!m) return 0;
    size_t size = 0;
    sentencepiece_SentencePieceText_pieces(m, &size);
    return size;
  }

  const SentencePieceText_SentencePiece& pieces(int index) const {
    LazyInitPieceWrappersCache();
    UPB_WRAPPER_RET_VAL_IF_OOB(index, piece_wrappers_.size(),
                               SentencePieceText_SentencePiece::default_instance());
    return *piece_wrappers_[index];
  }

  SentencePieceText_SentencePiece& pieces(int index) {
    LazyInitPieceWrappersCache();
    static SentencePieceText_SentencePiece dummy(nullptr, -1);
    UPB_WRAPPER_RET_VAL_IF_OOB(index, piece_wrappers_.size(), dummy);
    return *piece_wrappers_[index];
  }

  void ReservePieces(int size) {
    if (mutable_msg()) {
      // Do nothing. upb doesn't have a simple "reserve capacity" API,
      // and resize_pieces actually grows the array with uninitialized elements.
    }
  }

  SentencePieceText_SentencePiece* add_pieces() {
    if (!mutable_msg()) return nullptr;
    sentencepiece_SentencePieceText_add_pieces(mutable_msg(), arena());
    int index = pieces_size() - 1;
    auto wrapper =
        std::make_unique<SentencePieceText_SentencePiece>(this, index);
    SentencePieceText_SentencePiece* ptr = wrapper.get();
    piece_wrappers_.push_back(std::move(wrapper));
    return ptr;
  }

  SentencePieceTextPiecesRepeatedWrapper pieces() const;
  SentencePieceTextMutablePiecesRepeatedWrapper* mutable_pieces();



  SentencePieceText_SentencePiece* mutable_pieces(int index) {
    if (index < 0 || index >= pieces_size()) return nullptr;
    LazyInitPieceWrappersCache();
    return piece_wrappers_[index].get();
  }

  absl::string_view piece_at(int index) const {
    size_t size = 0;
    const auto* pieces = sentencepiece_SentencePieceText_pieces(msg(), &size);
    if (index < 0 || index >= size) return "";
    upb_StringView sv =
        sentencepiece_SentencePieceText_SentencePiece_piece(pieces[index]);
    return absl::string_view(sv.data, sv.size);
  }

  void set_piece_at(int index, absl::string_view piece) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return;
    sentencepiece_SentencePieceText_SentencePiece_set_piece(
        pieces[index], MakeUpbString(piece, arena()));
  }

  uint32_t id_at(int index) const {
    size_t size = 0;
    const auto* pieces = sentencepiece_SentencePieceText_pieces(msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return 0;
    return sentencepiece_SentencePieceText_SentencePiece_id(pieces[index]);
  }
  void set_id_at(int index, uint32_t id) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return;
    sentencepiece_SentencePieceText_SentencePiece_set_id(pieces[index], id);
  }

  absl::string_view surface_at(int index) const {
    size_t size = 0;
    const auto* pieces = sentencepiece_SentencePieceText_pieces(msg(), &size);
    if (index < 0 || index >= size) return "";
    upb_StringView sv =
        sentencepiece_SentencePieceText_SentencePiece_surface(pieces[index]);
    return absl::string_view(sv.data, sv.size);
  }
  void set_surface_at(int index, absl::string_view surface) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return;
    sentencepiece_SentencePieceText_SentencePiece_set_surface(
        pieces[index], MakeUpbString(surface, arena()));
  }

  uint32_t begin_at(int index) const {
    size_t size = 0;
    const auto* pieces = sentencepiece_SentencePieceText_pieces(msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return 0;
    return sentencepiece_SentencePieceText_SentencePiece_begin(pieces[index]);
  }
  void set_begin_at(int index, uint32_t begin) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return;
    sentencepiece_SentencePieceText_SentencePiece_set_begin(pieces[index],
                                                            begin);
  }

  uint32_t end_at(int index) const {
    size_t size = 0;
    const auto* pieces = sentencepiece_SentencePieceText_pieces(msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return 0;
    return sentencepiece_SentencePieceText_SentencePiece_end(pieces[index]);
  }
  void set_end_at(int index, uint32_t end) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return;
    sentencepiece_SentencePieceText_SentencePiece_set_end(pieces[index], end);
  }

  void SwapElementsData(int i, int j) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    UPB_WRAPPER_RET_IF_OOB(i, size);
    UPB_WRAPPER_RET_IF_OOB(j, size);
    std::swap(pieces[i], pieces[j]);
  }

  void LazyInitPieceWrappersCache() const {
    ::sentencepiece::upb_internal::LazyInitCache(
        piece_wrappers_, const_cast<SentencePieceTextWrapper*>(this),
        pieces_size());
  }

 private:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_SentencePieceText> holder_;

  mutable std::vector<std::unique_ptr<SentencePieceText_SentencePiece>>
      piece_wrappers_;
  std::unique_ptr<SentencePieceTextMutablePiecesRepeatedWrapper> mutable_pieces_wrapper_;
};



class NBestSentencePieceTextConstNbestsRepeatedWrapper;
class NBestSentencePieceTextMutableNbestsRepeatedWrapper;

class NBestSentencePieceTextWrapper {
 public:
  NBestSentencePieceTextWrapper() {
    upb_Arena* arena = upb_Arena_New();
    holder_.Reset(sentencepiece_NBestSentencePieceText_new(arena), arena, true);
  }
  virtual ~NBestSentencePieceTextWrapper() = default;

  NBestSentencePieceTextWrapper(const NBestSentencePieceTextWrapper& other)
      : NBestSentencePieceTextWrapper() {
    CopyFrom(other);
  }
  NBestSentencePieceTextWrapper& operator=(
      const NBestSentencePieceTextWrapper& other) {
    if (this != &other) {
      Clear();
      CopyFrom(other);
    }
    return *this;
  }

  DEFINE_UPB_SERIALIZATION_METHODS(NBestSentencePieceTextWrapper,
                                   sentencepiece_NBestSentencePieceText)

  void OnArenaReset() {
    nbest_wrappers_.clear();
    if (holder_.msg()) {
      LazyInitNbestWrappersCache();
    }
  }

  virtual sentencepiece_NBestSentencePieceText* mutable_msg() {
    return static_cast<sentencepiece_NBestSentencePieceText*>(holder_.msg());
  }
  virtual const sentencepiece_NBestSentencePieceText* msg() const {
    return static_cast<const sentencepiece_NBestSentencePieceText*>(holder_.msg());
  }
  virtual upb_Arena* arena() { return holder_.arena(); }
  virtual const upb_Arena* arena() const { return holder_.arena(); }


  int nbests_size() const {
    if (!holder_.msg()) return 0;
    size_t size = 0;
    sentencepiece_NBestSentencePieceText_nbests(holder_.msg(), &size);
    return size;
  }

  void ReserveNbests(int size) {
    if (holder_.msg()) {
      sentencepiece_NBestSentencePieceText_resize_nbests(holder_.msg(), size, holder_.arena());
    }
  }

  inline NBestSentencePieceText_Sub* add_nbests();
  inline const NBestSentencePieceText_Sub& nbests(int index) const;

  NBestSentencePieceTextConstNbestsRepeatedWrapper nbests() const;


  void LazyInitNbestWrappersCache() const {
    ::sentencepiece::upb_internal::LazyInitCache(
        nbest_wrappers_, const_cast<NBestSentencePieceTextWrapper*>(this),
        nbests_size());
  }

 protected:
  mutable sentencepiece::upb_internal::UpbMessageHolder<sentencepiece_NBestSentencePieceText> holder_;

  mutable std::vector<std::unique_ptr<NBestSentencePieceText_Sub>>
      nbest_wrappers_;

 private:
};



}  // namespace upb

class TrainerSpec : public sentencepiece::upb::TrainerSpecWrapper {
 public:
  using TrainerSpecWrapper::BPE;
  using TrainerSpecWrapper::CHAR;
  using TrainerSpecWrapper::ModelType;
  using TrainerSpecWrapper::TrainerSpecWrapper;
  using TrainerSpecWrapper::UNIGRAM;
  using TrainerSpecWrapper::WORD;

  static const TrainerSpec& default_instance() {
    static TrainerSpec instance(nullptr);
    return instance;
  }
};

class NormalizerSpec : public sentencepiece::upb::NormalizerSpecWrapper {
 public:
  using NormalizerSpecWrapper::NormalizerSpecWrapper;

  static const NormalizerSpec& default_instance() {
    static NormalizerSpec instance(nullptr);
    return instance;
  }
};

class SelfTestData : public sentencepiece::upb::SelfTestDataWrapper {
 public:
  using SelfTestDataWrapper::SelfTestDataWrapper;

  static const SelfTestData& default_instance() {
    static SelfTestData instance(nullptr);
    return instance;
  }
};

class ModelProto : public sentencepiece::upb::ModelProtoWrapper {
 public:
  using ModelProtoWrapper::ModelProtoWrapper;
  using ModelProtoWrapper::pieces;
  using SentencePiece = ModelProto_SentencePiece;

  inline ModelProto_SentencePiece* add_pieces();

  static const ModelProto& default_instance() {
    static ModelProto instance(nullptr, nullptr);
    return instance;
  }
};

class NBestSentencePieceText;

class NBestSentencePieceText
    : public sentencepiece::upb::NBestSentencePieceTextWrapper {
 public:
  using NBestSentencePieceTextWrapper::nbests;
  using NBestSentencePieceTextWrapper::NBestSentencePieceTextWrapper;
  using SentencePieceTextWrapperSub = ::sentencepiece::NBestSentencePieceText_Sub;

  upb::NBestSentencePieceTextMutableNbestsRepeatedWrapper* mutable_nbests();
  inline ::sentencepiece::NBestSentencePieceText_Sub* mutable_nbests_at(int index);

  inline const ::sentencepiece::NBestSentencePieceText_Sub& nbests(int index) const;
  inline ::sentencepiece::NBestSentencePieceText_Sub* add_nbests();

  static const NBestSentencePieceText& default_instance() {
    static NBestSentencePieceText instance;
    return instance;
  }

 private:
  std::unique_ptr<upb::NBestSentencePieceTextMutableNbestsRepeatedWrapper> mutable_nbests_wrapper_;
};



// --- SentencePieceText public classes (namespace sentencepiece) ---

class SentencePieceText : public sentencepiece::upb::SentencePieceTextWrapper {
 public:
  using SentencePieceTextWrapper::SentencePieceTextWrapper;
  using SentencePiece = SentencePieceText_SentencePiece;

  static const SentencePieceText& default_instance() {
    static SentencePieceText instance(nullptr);
    return instance;
  }
};

class NBestSentencePieceText_Sub : public SentencePieceText {
 public:
  NBestSentencePieceText_Sub(upb::NBestSentencePieceTextWrapper* parent,
                             int index)
      : SentencePieceText(nullptr), parent_(parent), index_(index) {}

  inline sentencepiece_SentencePieceText* mutable_msg() override;
  inline const sentencepiece_SentencePieceText* msg() const override;
  inline upb_Arena* arena() override;
  inline const upb_Arena* arena() const override;

 private:
  upb::NBestSentencePieceTextWrapper* parent_;
  int index_;
};


// --- repeated wrappers and inline implementations (namespace sentencepiece::upb) ---
namespace upb {

class SentencePieceTextPiecesRepeatedWrapper : public ::sentencepiece::upb_internal::ConstRepeatedWrapperBase<
    SentencePieceTextPiecesRepeatedWrapper, SentencePieceText_SentencePiece> {
 public:
  SentencePieceTextPiecesRepeatedWrapper(const SentencePieceTextWrapper* parent) : parent_(parent) {}
  SentencePieceText_SentencePiece Get(int index) const { return parent_->pieces(index); }
  int Size() const { return parent_->pieces_size(); }
 private:
  const SentencePieceTextWrapper* parent_;
};

class SentencePieceTextMutablePiecesRepeatedWrapper : public ::sentencepiece::upb_internal::MutableRepeatedWrapperBase<
    SentencePieceTextMutablePiecesRepeatedWrapper, SentencePieceText_SentencePiece> {
 public:
  SentencePieceTextMutablePiecesRepeatedWrapper(SentencePieceTextWrapper* parent) : parent_(parent) {}
  SentencePieceText_SentencePiece* GetMutable(int index) { return parent_->mutable_pieces(index); }
  int Size() const { return parent_->pieces_size(); }

  SentencePieceText_SentencePiece* Add() { return parent_->add_pieces(); }
  void SwapElements(int i, int j) { parent_->SwapElementsData(i, j); }
  void Reserve(int size) { parent_->ReservePieces(size); }
 private:
  SentencePieceTextWrapper* parent_;
};

class NBestSentencePieceTextConstNbestsRepeatedWrapper : public ::sentencepiece::upb_internal::ConstRepeatedWrapperBase<
    NBestSentencePieceTextConstNbestsRepeatedWrapper, ::sentencepiece::NBestSentencePieceText_Sub> {
 public:
  NBestSentencePieceTextConstNbestsRepeatedWrapper(const NBestSentencePieceTextWrapper* parent) : parent_(parent) {}
  ::sentencepiece::NBestSentencePieceText_Sub Get(int index) const { return parent_->nbests(index); }
  int Size() const { return parent_->nbests_size(); }
 private:
  const NBestSentencePieceTextWrapper* parent_;
};

class NBestSentencePieceTextMutableNbestsRepeatedWrapper : public ::sentencepiece::upb_internal::MutableRepeatedWrapperBase<
    NBestSentencePieceTextMutableNbestsRepeatedWrapper, ::sentencepiece::NBestSentencePieceText_Sub> {
 public:
  NBestSentencePieceTextMutableNbestsRepeatedWrapper(NBestSentencePieceText* parent) : parent_(parent) {}
  ::sentencepiece::NBestSentencePieceText_Sub* GetMutable(int index) { return parent_->mutable_nbests_at(index); }
  int Size() const { return parent_->nbests_size(); }

  ::sentencepiece::NBestSentencePieceText_Sub* Add() { return parent_->add_nbests(); }
 private:
  NBestSentencePieceText* parent_;
};

// Inline getters for SentencePieceTextWrapper
inline SentencePieceTextPiecesRepeatedWrapper SentencePieceTextWrapper::pieces() const {
  return SentencePieceTextPiecesRepeatedWrapper(this);
}

inline SentencePieceTextMutablePiecesRepeatedWrapper*
SentencePieceTextWrapper::mutable_pieces() {
  if (!mutable_pieces_wrapper_) {
    mutable_pieces_wrapper_ =
        std::make_unique<SentencePieceTextMutablePiecesRepeatedWrapper>(this);
  }
  return mutable_pieces_wrapper_.get();
}

// Inline getters for NBestSentencePieceTextWrapper
inline NBestSentencePieceTextConstNbestsRepeatedWrapper NBestSentencePieceTextWrapper::nbests() const {
  return NBestSentencePieceTextConstNbestsRepeatedWrapper(this);
}

}  // namespace upb

inline upb::NBestSentencePieceTextMutableNbestsRepeatedWrapper*
NBestSentencePieceText::mutable_nbests() {
  if (!mutable_nbests_wrapper_) {
    mutable_nbests_wrapper_ =
        std::make_unique<upb::NBestSentencePieceTextMutableNbestsRepeatedWrapper>(this);
  }
  return mutable_nbests_wrapper_.get();
}

inline ProtoStr ModelProto_SentencePiece::piece() const {
  return parent_ ? parent_->piece_at(index_) : "";
}
inline float ModelProto_SentencePiece::score() const {
  return parent_ ? parent_->score_at(index_) : 0.0f;
}
inline ModelProto_SentencePiece::Type ModelProto_SentencePiece::type() const {
  return parent_ ? static_cast<Type>(parent_->type_at(index_)) : NORMAL;
}
inline void ModelProto_SentencePiece::set_type(Type type) {
  if (parent_) parent_->set_type_at(index_, type);
}

inline TrainerSpec upb::ModelProtoWrapper::trainer_spec() const {
  const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
  return TrainerSpec(
      m ? sentencepiece_ModelProto_trainer_spec(m) : nullptr, holder_.arena());
}
inline const NormalizerSpec& upb::ModelProtoWrapper::normalizer_spec() const {
  static const NormalizerSpec default_instance(nullptr);
  return normalizer_spec_cache_ ? *normalizer_spec_cache_ : default_instance;
}
inline const NormalizerSpec& upb::ModelProtoWrapper::denormalizer_spec() const {
  static const NormalizerSpec default_instance(nullptr);
  return denormalizer_spec_cache_ ? *denormalizer_spec_cache_ : default_instance;
}
inline SelfTestData upb::ModelProtoWrapper::self_test_data() const {
  const auto* m = static_cast<const sentencepiece_ModelProto*>(holder_.msg());
  return SelfTestData(m ? sentencepiece_ModelProto_self_test_data(m)
                           : nullptr);
}

inline upb_Arena* NBestSentencePieceText_Sub::arena() {
  return parent_ ? parent_->arena() : nullptr;
}
inline const upb_Arena* NBestSentencePieceText_Sub::arena() const {
  return parent_ ? parent_->arena() : nullptr;
}

inline NBestSentencePieceText_Sub*
upb::NBestSentencePieceTextWrapper::add_nbests() {
  auto* m = static_cast<sentencepiece_NBestSentencePieceText*>(holder_.msg());
  if (!m) return nullptr;
  sentencepiece_NBestSentencePieceText_add_nbests(m, holder_.arena());

  int index = nbests_size() - 1;

  auto wrapper = std::make_unique<NBestSentencePieceText_Sub>(this, index);
  NBestSentencePieceText_Sub* ptr = wrapper.get();
  nbest_wrappers_.push_back(std::move(wrapper));
  return ptr;
}

inline const NBestSentencePieceText_Sub&
upb::NBestSentencePieceTextWrapper::nbests(int index) const {
  LazyInitNbestWrappersCache();
  static const NBestSentencePieceText_Sub default_sub(nullptr, -1);
  UPB_WRAPPER_RET_VAL_IF_OOB(index, nbest_wrappers_.size(), default_sub);
  return *nbest_wrappers_[index];
}


inline sentencepiece_SentencePieceText*
NBestSentencePieceText_Sub::mutable_msg() {
  if (!parent_ || index_ < 0) return nullptr;
  size_t size = 0;
  sentencepiece_SentencePieceText** nbests =
      sentencepiece_NBestSentencePieceText_mutable_nbests(
          parent_->mutable_msg(), &size);
  if (static_cast<size_t>(index_) >= size) return nullptr;
  return nbests[index_];
}
inline const sentencepiece_SentencePieceText* NBestSentencePieceText_Sub::msg()
    const {
  if (!parent_ || index_ < 0) return nullptr;
  size_t size = 0;
  const sentencepiece_SentencePieceText* const* nbests =
      sentencepiece_NBestSentencePieceText_nbests(parent_->msg(), &size);
  if (static_cast<size_t>(index_) >= size) return nullptr;
  return nbests[index_];
}

inline absl::string_view SentencePieceText_SentencePiece::piece() const {
  if (!parent_) {
    return "";
  }
  return parent_->piece_at(index_);
}
inline void SentencePieceText_SentencePiece::set_piece(
    absl::string_view piece) {
  if (parent_) {
    parent_->set_piece_at(index_, piece);
  }
}
inline void SentencePieceText_SentencePiece::set_piece(const char* data,
                                                       size_t size) {
  set_piece(absl::string_view(data, size));
}

inline uint32_t SentencePieceText_SentencePiece::id() const {
  return parent_ ? parent_->id_at(index_) : 0;
}
inline void SentencePieceText_SentencePiece::set_id(uint32_t id) {
  if (parent_) parent_->set_id_at(index_, id);
}

inline absl::string_view SentencePieceText_SentencePiece::surface() const {
  if (!parent_) {
    return "";
  }
  return parent_->surface_at(index_);
}
inline void SentencePieceText_SentencePiece::set_surface(
    absl::string_view surface) {
  if (parent_) {
    parent_->set_surface_at(index_, surface);
  }
}
inline void SentencePieceText_SentencePiece::set_surface(const char* data,
                                                         size_t size) {
  set_surface(absl::string_view(data, size));
}

inline uint32_t SentencePieceText_SentencePiece::begin() const {
  return parent_ ? parent_->begin_at(index_) : 0;
}
inline void SentencePieceText_SentencePiece::set_begin(uint32_t begin) {
  if (parent_) parent_->set_begin_at(index_, begin);
}

inline uint32_t SentencePieceText_SentencePiece::end() const {
  return parent_ ? parent_->end_at(index_) : 0;
}
inline void SentencePieceText_SentencePiece::set_end(uint32_t end) {
  if (parent_) parent_->set_end_at(index_, end);
}

inline void swap(SentencePieceText_SentencePiece a,
                 SentencePieceText_SentencePiece b) {
  a.parent_->SwapElementsData(a.index_, b.index_);
}

inline const ::sentencepiece::NBestSentencePieceText_Sub& NBestSentencePieceText::nbests(
    int index) const {
  return NBestSentencePieceTextWrapper::nbests(index);
}
inline ::sentencepiece::NBestSentencePieceText_Sub* NBestSentencePieceText::add_nbests() {
  return NBestSentencePieceTextWrapper::add_nbests();
}
inline ::sentencepiece::NBestSentencePieceText_Sub* NBestSentencePieceText::mutable_nbests_at(
    int index) {
  UPB_WRAPPER_RET_VAL_IF_OOB(index, nbests_size(), nullptr);
  LazyInitNbestWrappersCache();
  return nbest_wrappers_[index].get();
}

inline ::sentencepiece::NormalizerSpec*
upb::ModelProtoWrapper::mutable_normalizer_spec() const {
  return GetOrCreateMutableSpec<::sentencepiece::NormalizerSpec, sentencepiece_NormalizerSpec,
                                sentencepiece_ModelProto_mutable_normalizer_spec,
                                sentencepiece_ModelProto_set_normalizer_spec,
                                sentencepiece_ModelProto_normalizer_spec>(
      mutable_normalizer_spec_, &normalizer_spec_cache_);
}

inline ::sentencepiece::NormalizerSpec*
upb::ModelProtoWrapper::mutable_denormalizer_spec() const {
  return GetOrCreateMutableSpec<::sentencepiece::NormalizerSpec, sentencepiece_NormalizerSpec,
                                sentencepiece_ModelProto_mutable_denormalizer_spec,
                                sentencepiece_ModelProto_set_denormalizer_spec,
                                sentencepiece_ModelProto_denormalizer_spec>(
      mutable_denormalizer_spec_, &denormalizer_spec_cache_);
}

inline ModelProto_SentencePiece* ModelProto::add_pieces() {
  auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
  if (m) {
    sentencepiece_ModelProto_add_pieces(m, holder_.arena());
    int new_index = pieces_size() - 1;
    piece_wrappers_.resize(pieces_size());
    piece_wrappers_[new_index] =
        std::make_unique<ModelProto_SentencePiece>(this, new_index);
    return piece_wrappers_[new_index].get();
  }
  return nullptr;
}

inline ::sentencepiece::TrainerSpec*
upb::ModelProtoWrapper::mutable_trainer_spec() const {
  return GetOrCreateMutableSpec<::sentencepiece::TrainerSpec, sentencepiece_TrainerSpec,
                                sentencepiece_ModelProto_mutable_trainer_spec,
                                sentencepiece_ModelProto_set_trainer_spec,
                                sentencepiece_ModelProto_trainer_spec>(
      mutable_trainer_spec_);
}

inline ::sentencepiece::SelfTestData*
upb::ModelProtoWrapper::mutable_self_test_data() const {
  if (!mutable_self_test_data_) {
    auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
    if (m) {
      sentencepiece_SelfTestData* sub_msg =
          sentencepiece_ModelProto_mutable_self_test_data(m, holder_.arena());
      mutable_self_test_data_ =
          std::make_unique<::sentencepiece::SelfTestData>(sub_msg, holder_.arena());
    } else {
      mutable_self_test_data_ =
          std::make_unique<::sentencepiece::SelfTestData>(nullptr, nullptr);
    }
  }
  return mutable_self_test_data_.get();
}

inline void upb::ModelProtoWrapper::set_trainer_spec(const TrainerSpec& spec) {
  auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
  if (m) {
    UPB_WRAPPER_COPY_AND_SET_SUB_MSG(m, spec,
                                     sentencepiece_ModelProto_set_trainer_spec,
                                     sentencepiece_TrainerSpec_parse,
                                     mutable_trainer_spec_,
                                     ::sentencepiece::TrainerSpec,
                                     holder_.arena());
  }
}

inline void upb::ModelProtoWrapper::set_normalizer_spec(
    const NormalizerSpec& spec) {
  auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
  if (m) {
    UPB_WRAPPER_COPY_AND_SET_SUB_MSG(m, spec,
                                     sentencepiece_ModelProto_set_normalizer_spec,
                                     sentencepiece_NormalizerSpec_parse,
                                     normalizer_spec_cache_,
                                     ::sentencepiece::NormalizerSpec,
                                     holder_.arena());
  }
}

inline void upb::ModelProtoWrapper::set_denormalizer_spec(
    const NormalizerSpec& spec) {
  auto* m = static_cast<sentencepiece_ModelProto*>(holder_.msg());
  if (m) {
    UPB_WRAPPER_COPY_AND_SET_SUB_MSG(m, spec,
                                     sentencepiece_ModelProto_set_denormalizer_spec,
                                     sentencepiece_NormalizerSpec_parse,
                                     mutable_denormalizer_spec_,
                                     ::sentencepiece::NormalizerSpec,
                                     holder_.arena());
  }
}

inline void ModelProto_SentencePiece::set_piece(absl::string_view piece) {
  parent_->set_piece_at(index_, piece);
}
inline void ModelProto_SentencePiece::set_score(float score) {
  parent_->set_score_at(index_, score);
}

// ModelProtoWrapper implementations
inline upb::ModelProtoWrapper::ModelProtoWrapper() {
  upb_Arena* arena = upb_Arena_New();
  holder_.Reset(sentencepiece_ModelProto_new(arena), arena, true);
  normalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
      sentencepiece_ModelProto_normalizer_spec(static_cast<const sentencepiece_ModelProto*>(holder_.msg())), holder_.arena());
  denormalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
      sentencepiece_ModelProto_denormalizer_spec(static_cast<const sentencepiece_ModelProto*>(holder_.msg())), holder_.arena());
}
inline upb::ModelProtoWrapper::ModelProtoWrapper(const sentencepiece_ModelProto* msg,
                                                 upb_Arena* arena)
    : holder_(const_cast<sentencepiece_ModelProto*>(msg), arena, false) {}
inline upb::ModelProtoWrapper::~ModelProtoWrapper() = default;

// SentencePieceTextWrapper implementations
inline upb::SentencePieceTextWrapper::SentencePieceTextWrapper() {
  upb_Arena* arena = upb_Arena_New();
  holder_.Reset(sentencepiece_SentencePieceText_new(arena), arena, true);
}
inline upb::SentencePieceTextWrapper::SentencePieceTextWrapper(std::nullptr_t)
    : holder_(nullptr, nullptr, false) {}
inline upb::SentencePieceTextWrapper::~SentencePieceTextWrapper() = default;

}  // namespace sentencepiece

#endif  // CORE_UPB_WRAPPER_H_
