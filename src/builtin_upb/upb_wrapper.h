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

namespace sentencepiece {
class ProtoStr : public absl::string_view {
 public:
  ProtoStr(absl::string_view sv) : absl::string_view(sv) {}
  ProtoStr(const char* s) : absl::string_view(s ? s : "") {}
  ProtoStr(upb_StringView sv) : absl::string_view(sv.data, sv.size) {}

  operator std::string() const { return std::string(*this); }
  bool operator==(const std::string& other) const {
    return absl::string_view(*this) == other;
  }
  friend bool operator==(const std::string& a, const ProtoStr& b) {
    return a == absl::string_view(b);
  }
};

inline upb_StringView MakeUpbString(absl::string_view val, upb_Arena* arena) {
  upb_StringView sv;
  if (arena && !val.empty()) {
    char* ptr = static_cast<char*>(upb_Arena_Malloc(arena, val.size()));
    memcpy(ptr, val.data(), val.size());
    sv.data = ptr;
    sv.size = val.size();
  } else {
    sv.data = nullptr;
    sv.size = 0;
  }
  return sv;
}

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

#define DEFINE_UPB_SERIALIZATION_METHODS(ClassName, CPrefix)                   \
  std::string SerializeAsString() const {                                      \
    if (!msg_) return "";                                                      \
    size_t len;                                                                \
    upb_Arena* tmp_arena = upb_Arena_New();                                    \
    char* ptr = CPrefix##_serialize(msg_, tmp_arena, &len);                    \
    std::string ret;                                                           \
    if (ptr) {                                                                 \
      ret.assign(ptr, len);                                                    \
    }                                                                          \
    upb_Arena_Free(tmp_arena);                                                 \
    return ret;                                                                \
  }                                                                            \
  bool ParseFromArray(const void* data, int size) {                            \
    if (owns_msg_ && arena_) {                                                 \
      upb_Arena_Free(arena_);                                                  \
    }                                                                          \
    arena_ = upb_Arena_New();                                                  \
    owns_msg_ = true;                                                          \
    msg_ = CPrefix##_parse(reinterpret_cast<const char*>(data), size, arena_); \
    OnArenaReset();                                                            \
    return msg_ != nullptr;                                                    \
  }                                                                            \
  bool ParseFromString(const std::string& bytes) {                             \
    return ParseFromArray(bytes.data(), bytes.size());                         \
  }                                                                            \
  void CopyFrom(const ClassName& other) {                                      \
    std::string bytes = other.SerializeAsString();                             \
    ParseFromString(bytes);                                                    \
  }                                                                            \
  void Clear() {                                                               \
    if (owns_msg_ && arena_) {                                                 \
      upb_Arena_Free(arena_);                                                  \
    }                                                                          \
    arena_ = upb_Arena_New();                                                  \
    owns_msg_ = true;                                                          \
    msg_ = CPrefix##_new(arena_);                                              \
    OnArenaReset();                                                            \
  }

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

class RepeatedStringWrapper {
 public:
  RepeatedStringWrapper(const upb_StringView* arr, size_t size)
      : arr_(arr), size_(size) {}

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ProtoStr;
    using difference_type = std::ptrdiff_t;
    using pointer = ProtoStr*;
    using reference = ProtoStr;

    Iterator(const upb_StringView* arr, size_t index)
        : arr_(arr), index_(index) {}
    Iterator& operator++() {
      ++index_;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++index_;
      return tmp;
    }
    bool operator==(const Iterator& other) const {
      return index_ == other.index_;
    }
    bool operator!=(const Iterator& other) const {
      return index_ != other.index_;
    }
    ProtoStr operator*() const { return ProtoStr(arr_[index_]); }

   private:
    const upb_StringView* arr_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(arr_, 0); }
  Iterator end() const { return Iterator(arr_, size_); }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  ProtoStr operator[](size_t index) const { return ProtoStr(arr_[index]); }

 private:
  const upb_StringView* arr_;
  size_t size_;
};

#define DEFINE_UPB_PRIMITIVE_ACCESSOR(FieldName, Type, DefaultVal, UpbPrefix) \
  Type FieldName() const {                                                    \
    return msg_ ? UpbPrefix##_##FieldName(msg_) : DefaultVal;                 \
  }                                                                           \
  void set_##FieldName(Type val) {                                            \
    if (msg_) {                                                               \
      UpbPrefix##_set_##FieldName(msg_, val);                                 \
      if (on_change_) on_change_(msg_);                                       \
    }                                                                         \
  }                                                                           \
  void clear_##FieldName() {                                                  \
    if (msg_) {                                                               \
      UpbPrefix##_clear_##FieldName(msg_);                                    \
      if (on_change_) on_change_(msg_);                                       \
    }                                                                         \
  }

#define DEFINE_UPB_STRING_ACCESSOR(FieldName, DefaultVal, UpbPrefix)    \
  ProtoStr FieldName() const {                                          \
    if (!msg_ || !UpbPrefix##_has_##FieldName(msg_)) return DefaultVal; \
    return UpbPrefix##_##FieldName(msg_);                               \
  }                                                                     \
  void set_##FieldName(absl::string_view val) {                         \
    if (msg_ && arena_) {                                               \
      UpbPrefix##_set_##FieldName(msg_, MakeUpbString(val, arena_));    \
      if (on_change_) on_change_(msg_);                                 \
    }                                                                   \
  }                                                                     \
  void set_##FieldName(const char* data, size_t size) {                 \
    set_##FieldName(absl::string_view(data, size));                     \
  }                                                                     \
  void clear_##FieldName() {                                            \
    if (msg_) {                                                         \
      UpbPrefix##_clear_##FieldName(msg_);                              \
      if (on_change_) on_change_(msg_);                                 \
    }                                                                   \
  }

#define DEFINE_UPB_HAS_FIELD_ACCESSOR(FieldName, UpbPrefix) \
  bool has_##FieldName() const {                            \
    return msg_ && UpbPrefix##_has_##FieldName(msg_);       \
  }

#define DEFINE_UPB_ENUM_ACCESSOR(FieldName, EnumType, DefaultVal, UpbPrefix) \
  EnumType FieldName() const {                                               \
    return msg_ ? static_cast<EnumType>(UpbPrefix##_##FieldName(msg_))       \
                : DefaultVal;                                                \
  }                                                                          \
  void set_##FieldName(EnumType type) {                                      \
    if (msg_) {                                                              \
      UpbPrefix##_set_##FieldName(msg_, static_cast<int32_t>(type));         \
      if (on_change_) on_change_(msg_);                                      \
    }                                                                        \
  }                                                                          \
  void clear_##FieldName() {                                                 \
    if (msg_) {                                                              \
      UpbPrefix##_clear_##FieldName(msg_);                                   \
      if (on_change_) on_change_(msg_);                                      \
    }                                                                        \
  }

#define DEFINE_UPB_REPEATED_STRING_ACCESSOR(FieldName, UpbPrefix)            \
  int FieldName##_size() const {                                             \
    if (!msg_) return 0;                                                     \
    size_t size;                                                             \
    UpbPrefix##_##FieldName(msg_, &size);                                    \
    return size;                                                             \
  }                                                                          \
  ProtoStr FieldName(int index) const {                                      \
    if (!msg_) return "";                                                    \
    size_t size;                                                             \
    const upb_StringView* arr = UpbPrefix##_##FieldName(msg_, &size);        \
    if (index < 0 || static_cast<size_t>(index) >= size) return "";          \
    return arr[index];                                                       \
  }                                                                          \
  RepeatedStringWrapper FieldName() const {                                  \
    if (!msg_) return RepeatedStringWrapper(nullptr, 0);                     \
    size_t size;                                                             \
    const upb_StringView* arr = UpbPrefix##_##FieldName(msg_, &size);        \
    return RepeatedStringWrapper(arr, size);                                 \
  }                                                                          \
  void add_##FieldName(const std::string& val) {                             \
    if (msg_ && arena_) {                                                    \
      UpbPrefix##_add_##FieldName(msg_, MakeUpbString(val, arena_), arena_); \
      if (on_change_) on_change_(msg_);                                      \
    }                                                                        \
  }                                                                          \
  void clear_##FieldName() {                                                 \
    if (msg_) {                                                              \
      UpbPrefix##_clear_##FieldName(msg_);                                   \
      if (on_change_) on_change_(msg_);                                      \
    }                                                                        \
  }

class TrainerSpecWrapper {
  friend class upb::ModelProtoWrapper;

 public:
  TrainerSpecWrapper() : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_TrainerSpec_new(arena_);
  }
  explicit TrainerSpecWrapper(const sentencepiece_TrainerSpec* msg,
                              upb_Arena* arena = nullptr)
      : msg_(const_cast<sentencepiece_TrainerSpec*>(msg)),
        arena_(arena),
        owns_msg_(false) {}

  TrainerSpecWrapper(
      const sentencepiece_TrainerSpec* msg, upb_Arena* arena,
      std::function<void(const sentencepiece_TrainerSpec*)> on_change)
      : msg_(const_cast<sentencepiece_TrainerSpec*>(msg)),
        arena_(arena),
        owns_msg_(false),
        on_change_(on_change) {}

  TrainerSpecWrapper(const TrainerSpecWrapper& other)
      : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_TrainerSpec_new(arena_);
    CopyFrom(other);
  }

  TrainerSpecWrapper& operator=(const TrainerSpecWrapper& other) {
    if (this != &other) {
      if (owns_msg_) {
        if (arena_) upb_Arena_Free(arena_);
        arena_ = upb_Arena_New();
        owns_msg_ = true;
        CopyFrom(other);
      } else {
        if (other.msg_) {
          size_t size = 0;
          upb_Arena* tmp_arena = upb_Arena_New();
          char* buf =
              sentencepiece_TrainerSpec_serialize(other.msg_, tmp_arena, &size);
          if (buf) {
            msg_ = sentencepiece_TrainerSpec_parse(buf, size, arena_);
            if (on_change_) on_change_(msg_);
          }
          upb_Arena_Free(tmp_arena);
        } else {
          msg_ = nullptr;
          if (on_change_) on_change_(msg_);
        }
      }
    }
    return *this;
  }

  virtual ~TrainerSpecWrapper() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
  }

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
  sentencepiece_TrainerSpec* msg_;
  upb_Arena* arena_;
  bool owns_msg_ = false;
  std::function<void(const sentencepiece_TrainerSpec*)> on_change_;
};

class NormalizerSpecWrapper {
  friend class upb::ModelProtoWrapper;

 public:
  NormalizerSpecWrapper() : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_NormalizerSpec_new(arena_);
  }
  explicit NormalizerSpecWrapper(const sentencepiece_NormalizerSpec* msg,
                                 upb_Arena* arena = nullptr)
      : msg_(const_cast<sentencepiece_NormalizerSpec*>(msg)),
        arena_(arena),
        owns_msg_(false) {}

  NormalizerSpecWrapper(
      const sentencepiece_NormalizerSpec* msg, upb_Arena* arena,
      std::function<void(const sentencepiece_NormalizerSpec*)> on_change)
      : msg_(const_cast<sentencepiece_NormalizerSpec*>(msg)),
        arena_(arena),
        owns_msg_(false),
        on_change_(on_change) {}

  NormalizerSpecWrapper(const NormalizerSpecWrapper& other)
      : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_NormalizerSpec_new(arena_);
    CopyFrom(other);
  }

  NormalizerSpecWrapper& operator=(const NormalizerSpecWrapper& other) {
    if (this != &other) {
      if (owns_msg_) {
        if (arena_) upb_Arena_Free(arena_);
        arena_ = upb_Arena_New();
        owns_msg_ = true;
        CopyFrom(other);
      } else {
        if (other.msg_) {
          size_t size = 0;
          upb_Arena* tmp_arena = upb_Arena_New();
          char* buf = sentencepiece_NormalizerSpec_serialize(other.msg_,
                                                             tmp_arena, &size);
          if (buf) {
            msg_ = sentencepiece_NormalizerSpec_parse(buf, size, arena_);
            if (on_change_) on_change_(msg_);
          }
          upb_Arena_Free(tmp_arena);
        } else {
          msg_ = nullptr;
          if (on_change_) on_change_(msg_);
        }
      }
    }
    return *this;
  }

  virtual ~NormalizerSpecWrapper() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
  }

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
  sentencepiece_NormalizerSpec* msg_;
  upb_Arena* arena_;
  bool owns_msg_ = false;
  std::function<void(const sentencepiece_NormalizerSpec*)> on_change_;
};

class SelfTestData_SampleWrapper {
 public:
  explicit SelfTestData_SampleWrapper(
      const sentencepiece_SelfTestData_Sample* msg)
      : msg_(msg), mutable_msg_(nullptr), arena_(nullptr) {}
  SelfTestData_SampleWrapper(sentencepiece_SelfTestData_Sample* msg,
                             upb_Arena* arena)
      : msg_(msg), mutable_msg_(msg), arena_(arena) {}
  ProtoStr input() const {
    if (!msg_ || !sentencepiece_SelfTestData_Sample_has_input(msg_)) return "";
    return sentencepiece_SelfTestData_Sample_input(msg_);
  }
  ProtoStr expected() const {
    if (!msg_ || !sentencepiece_SelfTestData_Sample_has_expected(msg_))
      return "";
    return sentencepiece_SelfTestData_Sample_expected(msg_);
  }
  void set_input(const std::string& val) {
    if (mutable_msg_ && arena_) {
      sentencepiece_SelfTestData_Sample_set_input(mutable_msg_,
                                                  MakeUpbString(val, arena_));
    }
  }
  void set_expected(const std::string& val) {
    if (mutable_msg_ && arena_) {
      sentencepiece_SelfTestData_Sample_set_expected(
          mutable_msg_, MakeUpbString(val, arena_));
    }
  }

 private:
  const sentencepiece_SelfTestData_Sample* msg_;
  sentencepiece_SelfTestData_Sample* mutable_msg_;
  upb_Arena* arena_;
  std::function<void(const sentencepiece_SelfTestData_Sample*)> on_change_;
};

class SelfTestDataWrapper {
 public:
  explicit SelfTestDataWrapper(const sentencepiece_SelfTestData* msg)
      : msg_(msg), mutable_msg_(nullptr), arena_(nullptr) {}
  SelfTestDataWrapper(sentencepiece_SelfTestData* msg, upb_Arena* arena)
      : msg_(msg), mutable_msg_(msg), arena_(arena) {}

  class SamplesRepeatedWrapper {
   public:
    SamplesRepeatedWrapper(
        const sentencepiece_SelfTestData_Sample* const* elements, size_t count)
        : elements_(elements), count_(count) {}
    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = SelfTestData_SampleWrapper;
      using difference_type = std::ptrdiff_t;
      using pointer = SelfTestData_SampleWrapper*;
      using reference = SelfTestData_SampleWrapper;

      Iterator(const sentencepiece_SelfTestData_Sample* const* elements,
               size_t index)
          : elements_(elements), index_(index) {}
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      Iterator operator++(int) {
        Iterator tmp = *this;
        ++index_;
        return tmp;
      }
      bool operator==(const Iterator& other) const {
        return index_ == other.index_;
      }
      bool operator!=(const Iterator& other) const {
        return index_ != other.index_;
      }
      SelfTestData_SampleWrapper operator*() const {
        return SelfTestData_SampleWrapper(elements_[index_]);
      }

     private:
      const sentencepiece_SelfTestData_Sample* const* elements_;
      size_t index_;
    };

    Iterator begin() const { return Iterator(elements_, 0); }
    Iterator end() const { return Iterator(elements_, count_); }
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    SelfTestData_SampleWrapper operator[](size_t index) const {
      return SelfTestData_SampleWrapper(elements_[index]);
    }

   private:
    const sentencepiece_SelfTestData_Sample* const* elements_;
    size_t count_;
  };

  SamplesRepeatedWrapper samples() const {
    size_t size = 0;
    const sentencepiece_SelfTestData_Sample* const* elements = nullptr;
    if (msg_) {
      elements = sentencepiece_SelfTestData_samples(msg_, &size);
    }
    return SamplesRepeatedWrapper(elements, size);
  }
  int samples_size() const { return samples().size(); }
  SelfTestData_SampleWrapper* add_samples() {
    if (mutable_msg_ && arena_) {
      sentencepiece_SelfTestData_add_samples(mutable_msg_, arena_);
      int new_index = samples_size() - 1;
      sample_wrappers_.resize(samples_size());
      size_t size;
      sentencepiece_SelfTestData_Sample** samples =
          sentencepiece_SelfTestData_mutable_samples(mutable_msg_, &size);
      sample_wrappers_[new_index] =
          std::make_unique<SelfTestData_SampleWrapper>(samples[new_index],
                                                       arena_);
      return sample_wrappers_[new_index].get();
    }
    return nullptr;
  }

 private:
  const sentencepiece_SelfTestData* msg_;
  sentencepiece_SelfTestData* mutable_msg_;
  upb_Arena* arena_;
  mutable std::vector<std::unique_ptr<SelfTestData_SampleWrapper>>
      sample_wrappers_;
};

class ModelProtoWrapper {
 public:
  ModelProtoWrapper() : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_ModelProto_new(arena_);
    normalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
        sentencepiece_ModelProto_normalizer_spec(msg_), arena_);
    denormalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
        sentencepiece_ModelProto_denormalizer_spec(msg_), arena_);
  }

  explicit ModelProtoWrapper(const sentencepiece_ModelProto* msg,
                             upb_Arena* arena)
      : msg_(const_cast<sentencepiece_ModelProto*>(msg)),
        arena_(arena),
        owns_msg_(false) {}

  virtual ~ModelProtoWrapper() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
  }

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
    if (msg_) {
      normalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
          sentencepiece_ModelProto_normalizer_spec(msg_), arena_);
      denormalizer_spec_cache_ = std::make_unique<NormalizerSpec>(
          sentencepiece_ModelProto_denormalizer_spec(msg_), arena_);
    }
  }

  bool ParseFromIstream(std::istream* input) {
    std::string bytes((std::istreambuf_iterator<char>(*input)),
                      std::istreambuf_iterator<char>());
    return ParseFromString(bytes);
  }

  int pieces_size() const {
    if (!msg_) return 0;
    size_t size = 0;
    sentencepiece_ModelProto_pieces(msg_, &size);
    return size;
  }

  inline ::sentencepiece::ModelProto_SentencePiece pieces(int index) const {
    return ::sentencepiece::ModelProto_SentencePiece(
        const_cast<ModelProtoWrapper*>(this), index);
  }
  inline ::sentencepiece::ModelProto_SentencePiece pieces(int index) {
    return ::sentencepiece::ModelProto_SentencePiece(this, index);
  }

  class PiecesRepeatedWrapper {
   public:
    PiecesRepeatedWrapper(const ModelProtoWrapper* parent) : parent_(parent) {}
    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = ModelProto_SentencePiece;
      using difference_type = std::ptrdiff_t;
      using pointer = ModelProto_SentencePiece*;
      using reference = ModelProto_SentencePiece;

      Iterator(const ModelProtoWrapper* parent, int index)
          : parent_(parent), index_(index) {}
      ModelProto_SentencePiece operator*() const {
        return ModelProto_SentencePiece(const_cast<ModelProtoWrapper*>(parent_),
                                        index_);
      }
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      bool operator!=(const Iterator& other) const {
        return index_ != other.index_;
      }

     private:
      const ModelProtoWrapper* parent_;
      int index_;
    };
    Iterator begin() const { return Iterator(parent_, 0); }
    Iterator end() const { return Iterator(parent_, parent_->pieces_size()); }
    int size() const { return parent_->pieces_size(); }

   private:
    const ModelProtoWrapper* parent_;
  };

  PiecesRepeatedWrapper pieces() const { return PiecesRepeatedWrapper(this); }

  class MutablePiecesRepeatedWrapper {
   public:
    MutablePiecesRepeatedWrapper(ModelProtoWrapper* parent) : parent_(parent) {}
    class Iterator {
     public:
      using iterator_category = std::random_access_iterator_tag;
      using value_type = ModelProto_SentencePiece;
      using difference_type = std::ptrdiff_t;
      using pointer = ModelProto_SentencePiece*;
      using reference = ModelProto_SentencePiece&;

      Iterator(ModelProtoWrapper* parent, int index)
          : parent_(parent), index_(index) {}
      ModelProto_SentencePiece& operator*() const {
        return *(parent_->mutable_pieces(index_));
      }
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      Iterator operator++(int) {
        Iterator tmp = *this;
        ++index_;
        return tmp;
      }
      Iterator& operator--() {
        --index_;
        return *this;
      }
      Iterator operator--(int) {
        Iterator tmp = *this;
        --index_;
        return tmp;
      }

      Iterator& operator+=(difference_type n) {
        index_ += n;
        return *this;
      }
      Iterator& operator-=(difference_type n) {
        index_ -= n;
        return *this;
      }

      friend Iterator operator+(Iterator it, difference_type n) {
        it += n;
        return it;
      }
      friend Iterator operator+(difference_type n, Iterator it) {
        it += n;
        return it;
      }
      friend Iterator operator-(Iterator it, difference_type n) {
        it -= n;
        return it;
      }
      friend difference_type operator-(const Iterator& a, const Iterator& b) {
        return a.index_ - b.index_;
      }

      ModelProto_SentencePiece& operator[](difference_type n) const {
        return *(parent_->mutable_pieces(index_ + n));
      }

      bool operator==(const Iterator& other) const {
        return index_ == other.index_ && parent_ == other.parent_;
      }
      bool operator!=(const Iterator& other) const { return !(*this == other); }
      bool operator<(const Iterator& other) const {
        return index_ < other.index_;
      }
      bool operator>(const Iterator& other) const {
        return index_ > other.index_;
      }
      bool operator<=(const Iterator& other) const {
        return index_ <= other.index_;
      }
      bool operator>=(const Iterator& other) const {
        return index_ >= other.index_;
      }

     private:
      ModelProtoWrapper* parent_;
      int index_;
    };
    Iterator begin() { return Iterator(parent_, 0); }
    Iterator end() { return Iterator(parent_, parent_->pieces_size()); }
    int size() const { return parent_->pieces_size(); }
    ModelProto_SentencePiece* Mutable(int index) {
      return parent_->mutable_pieces(index);
    }

   private:
    ModelProtoWrapper* parent_;
  };

  MutablePiecesRepeatedWrapper* mutable_pieces() {
    if (!mutable_pieces_wrapper_) {
      mutable_pieces_wrapper_ =
          std::make_unique<MutablePiecesRepeatedWrapper>(this);
    }
    return mutable_pieces_wrapper_.get();
  }

  inline ModelProto_SentencePiece* mutable_pieces(int index) {
    LazyInitPieceWrappersCache();
    return piece_wrappers_[index].get();
  }

  ProtoStr piece_at(int index) const {
    size_t size = 0;
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(msg_, &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return "";
    return ProtoStr(
        sentencepiece_ModelProto_SentencePiece_piece(pieces[index]));
  }
  float score_at(int index) const {
    size_t size = 0;
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(msg_, &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return 0.0f;
    return sentencepiece_ModelProto_SentencePiece_score(pieces[index]);
  }
  int type_at(int index) const {
    size_t size = 0;
    const sentencepiece_ModelProto_SentencePiece* const* pieces =
        sentencepiece_ModelProto_pieces(msg_, &size);
    if (index < 0 || static_cast<size_t>(index) >= size) return 0;
    return sentencepiece_ModelProto_SentencePiece_type(pieces[index]);
  }
  void set_type_at(int index, int type) {
    if (msg_) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(msg_, &size);
      sentencepiece_ModelProto_SentencePiece_set_type(
          pieces[index],
          static_cast<sentencepiece_ModelProto_SentencePiece_Type>(type));
    }
  }
  void set_piece_at(int index, absl::string_view piece) {
    if (msg_) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(msg_, &size);
      sentencepiece_ModelProto_SentencePiece_set_piece(
          pieces[index], MakeUpbString(piece, arena_));
    }
  }
  void set_piece_at(int index, const std::string& piece) {
    set_piece_at(index, absl::string_view(piece));
  }
  void set_score_at(int index, float score) {
    if (msg_) {
      size_t size = 0;
      sentencepiece_ModelProto_SentencePiece** pieces =
          sentencepiece_ModelProto_mutable_pieces(msg_, &size);
      sentencepiece_ModelProto_SentencePiece_set_score(pieces[index], score);
    }
  }

  inline bool has_trainer_spec() const {
    return msg_ && sentencepiece_ModelProto_has_trainer_spec(msg_);
  }
  inline TrainerSpec trainer_spec() const;
  inline ::sentencepiece::TrainerSpec* mutable_trainer_spec() const;

  inline bool has_normalizer_spec() const {
    return msg_ && sentencepiece_ModelProto_has_normalizer_spec(msg_);
  }
  inline const NormalizerSpec& normalizer_spec() const;
  inline ::sentencepiece::NormalizerSpec* mutable_normalizer_spec() const;

  inline bool has_denormalizer_spec() const {
    return msg_ && sentencepiece_ModelProto_has_denormalizer_spec(msg_);
  }
  inline const NormalizerSpec& denormalizer_spec() const;
  inline ::sentencepiece::NormalizerSpec* mutable_denormalizer_spec() const;
  inline void set_trainer_spec(const TrainerSpec& spec);
  inline void set_normalizer_spec(const NormalizerSpec& spec);
  inline void set_denormalizer_spec(const NormalizerSpec& spec);

  inline bool has_self_test_data() const {
    return msg_ && sentencepiece_ModelProto_has_self_test_data(msg_);
  }
  inline SelfTestData self_test_data() const;
  inline ::sentencepiece::SelfTestData* mutable_self_test_data() const;

 protected:
  sentencepiece_ModelProto* msg_;
  upb_Arena* arena_;
  bool owns_msg_;

  std::vector<std::unique_ptr<ModelProto_SentencePiece>> piece_wrappers_;
  std::unique_ptr<MutablePiecesRepeatedWrapper> mutable_pieces_wrapper_;
  mutable std::unique_ptr<NormalizerSpec> mutable_normalizer_spec_;
  mutable std::unique_ptr<TrainerSpec> mutable_trainer_spec_;
  mutable std::unique_ptr<NormalizerSpec> normalizer_spec_cache_;
  mutable std::unique_ptr<NormalizerSpec> denormalizer_spec_cache_;
  mutable std::unique_ptr<NormalizerSpec> mutable_denormalizer_spec_;
  mutable std::unique_ptr<SelfTestData> mutable_self_test_data_;

 private:
  void LazyInitPieceWrappersCache() {
    if (piece_wrappers_.size() != pieces_size()) {
      piece_wrappers_.resize(pieces_size());
      for (int i = 0; i < pieces_size(); ++i) {
        if (!piece_wrappers_[i]) {
          piece_wrappers_[i] =
              std::make_unique<ModelProto_SentencePiece>(this, i);
        }
      }
    }
  }
};

class SentencePieceTextWrapper {
 public:
  SentencePieceTextWrapper() : owns_msg_(true), arena_(upb_Arena_New()) {
    msg_ = sentencepiece_SentencePieceText_new(arena_);
  }
  explicit SentencePieceTextWrapper(std::nullptr_t)
      : msg_(nullptr), owns_msg_(false), arena_(nullptr) {}

  virtual ~SentencePieceTextWrapper() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
  }

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
    if (msg_) {
      LazyInitPieceWrappersCache();
    }
  }

  virtual sentencepiece_SentencePieceText* mutable_msg() { return msg_; }
  virtual const sentencepiece_SentencePieceText* msg() const { return msg_; }
  virtual upb_Arena* arena() { return arena_; }
  virtual const upb_Arena* arena() const { return arena_; }

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
    if (index < 0 || static_cast<size_t>(index) >= piece_wrappers_.size()) {
      return SentencePieceText_SentencePiece::default_instance();
    }
    return *piece_wrappers_[index];
  }

  SentencePieceText_SentencePiece& pieces(int index) {
    LazyInitPieceWrappersCache();
    if (index < 0 || static_cast<size_t>(index) >= piece_wrappers_.size()) {
      static SentencePieceText_SentencePiece dummy(nullptr, -1);
      return dummy;
    }
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

  class ConstPiecesRepeatedWrapper {
   public:
    ConstPiecesRepeatedWrapper(const SentencePieceTextWrapper* parent)
        : parent_(parent) {}
    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = SentencePieceText_SentencePiece;
      using difference_type = std::ptrdiff_t;
      using pointer = SentencePieceText_SentencePiece*;
      using reference = SentencePieceText_SentencePiece&;

      Iterator(const SentencePieceTextWrapper* parent, int index)
          : parent_(parent), index_(index) {}
      SentencePieceText_SentencePiece& operator*() const {
        return *(const_cast<SentencePieceTextWrapper*>(parent_)->mutable_pieces(
            index_));
      }
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      bool operator!=(const Iterator& other) const {
        return index_ != other.index_;
      }

     private:
      const SentencePieceTextWrapper* parent_;
      int index_;
    };
    Iterator begin() const { return Iterator(parent_, 0); }
    Iterator end() const { return Iterator(parent_, parent_->pieces_size()); }
    int size() const { return parent_->pieces_size(); }

   private:
    const SentencePieceTextWrapper* parent_;
  };

  ConstPiecesRepeatedWrapper pieces() const {
    return ConstPiecesRepeatedWrapper(this);
  }

  class MutablePiecesRepeatedWrapper {
   public:
    MutablePiecesRepeatedWrapper(SentencePieceTextWrapper* parent)
        : parent_(parent) {}
    class Iterator {
     public:
      using iterator_category = std::random_access_iterator_tag;
      using value_type = SentencePieceText_SentencePiece;
      using difference_type = std::ptrdiff_t;
      using pointer = SentencePieceText_SentencePiece*;
      using reference = SentencePieceText_SentencePiece&;

      Iterator(SentencePieceTextWrapper* parent, int index)
          : parent_(parent), index_(index) {}
      Iterator() : parent_(nullptr), index_(-1) {}

      SentencePieceText_SentencePiece& operator*() const {
        return *(parent_->mutable_pieces(index_));
      }
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      Iterator operator++(int) {
        Iterator tmp = *this;
        ++index_;
        return tmp;
      }
      Iterator& operator--() {
        --index_;
        return *this;
      }
      Iterator operator--(int) {
        Iterator tmp = *this;
        --index_;
        return tmp;
      }

      Iterator& operator+=(difference_type n) {
        index_ += n;
        return *this;
      }
      Iterator& operator-=(difference_type n) {
        index_ -= n;
        return *this;
      }

      friend Iterator operator+(Iterator it, difference_type n) {
        it += n;
        return it;
      }
      friend Iterator operator+(difference_type n, Iterator it) {
        it += n;
        return it;
      }
      friend Iterator operator-(Iterator it, difference_type n) {
        it -= n;
        return it;
      }
      friend difference_type operator-(const Iterator& a, const Iterator& b) {
        return a.index_ - b.index_;
      }

      SentencePieceText_SentencePiece& operator[](difference_type n) const {
        return *(parent_->mutable_pieces(index_ + n));
      }

      bool operator==(const Iterator& other) const {
        return index_ == other.index_ && parent_ == other.parent_;
      }
      bool operator!=(const Iterator& other) const { return !(*this == other); }
      bool operator<(const Iterator& other) const {
        return index_ < other.index_;
      }
      bool operator>(const Iterator& other) const {
        return index_ > other.index_;
      }
      bool operator<=(const Iterator& other) const {
        return index_ <= other.index_;
      }
      bool operator>=(const Iterator& other) const {
        return index_ >= other.index_;
      }

     private:
      SentencePieceTextWrapper* parent_;
      int index_;
    };
    Iterator begin() { return Iterator(parent_, 0); }
    Iterator end() { return Iterator(parent_, parent_->pieces_size()); }
    int size() const { return parent_->pieces_size(); }

    SentencePieceText_SentencePiece* Add() { return parent_->add_pieces(); }
    void SwapElements(int i, int j) { parent_->SwapElementsData(i, j); }
    void Reserve(int size) { parent_->ReservePieces(size); }
    SentencePieceText_SentencePiece* Mutable(int index) {
      return parent_->mutable_pieces(index);
    }

   private:
    SentencePieceTextWrapper* parent_;
  };

  MutablePiecesRepeatedWrapper* mutable_pieces() {
    if (!mutable_pieces_wrapper_) {
      mutable_pieces_wrapper_ =
          std::make_unique<MutablePiecesRepeatedWrapper>(this);
    }
    return mutable_pieces_wrapper_.get();
  }

  SentencePieceText_SentencePiece* mutable_pieces(int index) {
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
    sentencepiece_SentencePieceText_SentencePiece_set_end(pieces[index], end);
  }

  void SwapElementsData(int i, int j) {
    size_t size = 0;
    auto** pieces =
        sentencepiece_SentencePieceText_mutable_pieces(mutable_msg(), &size);
    std::swap(pieces[i], pieces[j]);
  }

  void LazyInitPieceWrappersCache() const {
    if (piece_wrappers_.size() != pieces_size()) {
      piece_wrappers_.resize(pieces_size());
      for (int i = 0; i < pieces_size(); ++i) {
        if (!piece_wrappers_[i]) {
          piece_wrappers_[i] =
              std::make_unique<SentencePieceText_SentencePiece>(
                  const_cast<SentencePieceTextWrapper*>(this), i);
        }
      }
    }
  }

 private:
  sentencepiece_SentencePieceText* msg_;
  bool owns_msg_;
  upb_Arena* arena_;

  mutable std::vector<std::unique_ptr<SentencePieceText_SentencePiece>>
      piece_wrappers_;
  std::unique_ptr<MutablePiecesRepeatedWrapper> mutable_pieces_wrapper_;
};

class NBestSentencePieceTextWrapper {
 public:
  NBestSentencePieceTextWrapper() : arena_(upb_Arena_New()), owns_msg_(true) {
    msg_ = sentencepiece_NBestSentencePieceText_new(arena_);
  }
  virtual ~NBestSentencePieceTextWrapper() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
  }

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
    if (msg_) {
      LazyInitNbestWrappersCache();
    }
  }

  virtual sentencepiece_NBestSentencePieceText* mutable_msg() { return msg_; }
  virtual const sentencepiece_NBestSentencePieceText* msg() const {
    return msg_;
  }
  virtual upb_Arena* arena() { return arena_; }
  virtual const upb_Arena* arena() const { return arena_; }

  int nbests_size() const {
    if (!msg_) return 0;
    size_t size = 0;
    sentencepiece_NBestSentencePieceText_nbests(msg_, &size);
    return size;
  }

  void ReserveNbests(int size) {
    if (msg_) {
      sentencepiece_NBestSentencePieceText_resize_nbests(msg_, size, arena_);
    }
  }

  inline NBestSentencePieceText_Sub* add_nbests();
  inline const NBestSentencePieceText_Sub& nbests(int index) const;

  class ConstNbestsRepeatedWrapper {
   public:
    ConstNbestsRepeatedWrapper(const NBestSentencePieceTextWrapper* parent)
        : parent_(parent) {}
    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NBestSentencePieceText_Sub;
      using difference_type = std::ptrdiff_t;
      using pointer = const NBestSentencePieceText_Sub*;
      using reference = const NBestSentencePieceText_Sub&;

      Iterator(const NBestSentencePieceTextWrapper* parent, int index)
          : parent_(parent), index_(index) {}
      inline const NBestSentencePieceText_Sub& operator*() const;
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      bool operator!=(const Iterator& other) const {
        return index_ != other.index_;
      }

     private:
      const NBestSentencePieceTextWrapper* parent_;
      int index_;
    };
    Iterator begin() const { return Iterator(parent_, 0); }
    Iterator end() const { return Iterator(parent_, parent_->nbests_size()); }
    int size() const { return parent_->nbests_size(); }

   private:
    const NBestSentencePieceTextWrapper* parent_;
  };

  ConstNbestsRepeatedWrapper nbests() const {
    return ConstNbestsRepeatedWrapper(this);
  }

  void LazyInitNbestWrappersCache() const {
    if (nbest_wrappers_.size() != nbests_size()) {
      nbest_wrappers_.resize(nbests_size());
    }
    for (int i = 0; i < nbests_size(); ++i) {
      if (!nbest_wrappers_[i]) {
        nbest_wrappers_[i] = std::make_unique<NBestSentencePieceText_Sub>(
            const_cast<NBestSentencePieceTextWrapper*>(this), i);
      }
    }
  }

 protected:
  sentencepiece_NBestSentencePieceText* msg_;
  upb_Arena* arena_;
  bool owns_msg_;

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

class SentencePieceText : public sentencepiece::upb::SentencePieceTextWrapper {
 public:
  using SentencePieceTextWrapper::SentencePieceTextWrapper;
  using SentencePiece = SentencePieceText_SentencePiece;

  static const SentencePieceText& default_instance() {
    static SentencePieceText instance(nullptr);
    return instance;
  }
};

class NBestSentencePieceText;

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

class NBestSentencePieceText
    : public sentencepiece::upb::NBestSentencePieceTextWrapper {
 public:
  using NBestSentencePieceTextWrapper::nbests;
  using NBestSentencePieceTextWrapper::NBestSentencePieceTextWrapper;
  using SentencePieceTextWrapperSub = NBestSentencePieceText_Sub;

  inline const NBestSentencePieceText_Sub& nbests(int index) const;
  inline NBestSentencePieceText_Sub* add_nbests();

  class MutableNbestsRepeatedWrapper {
   public:
    MutableNbestsRepeatedWrapper(NBestSentencePieceText* parent)
        : parent_(parent) {}

    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NBestSentencePieceText_Sub;
      using difference_type = std::ptrdiff_t;
      using pointer = NBestSentencePieceText_Sub*;
      using reference = NBestSentencePieceText_Sub&;

      Iterator(NBestSentencePieceText* parent, int index)
          : parent_(parent), index_(index) {}
      inline NBestSentencePieceText_Sub& operator*() const {
        return *(parent_->mutable_nbests_at(index_));
      }
      Iterator& operator++() {
        ++index_;
        return *this;
      }
      bool operator!=(const Iterator& other) const {
        return index_ != other.index_;
      }

     private:
      NBestSentencePieceText* parent_;
      int index_;
    };

    Iterator begin() { return Iterator(parent_, 0); }
    Iterator end() { return Iterator(parent_, parent_->nbests_size()); }
    int size() const { return parent_->nbests_size(); }

    NBestSentencePieceText_Sub* Add() { return parent_->add_nbests(); }

   private:
    NBestSentencePieceText* parent_;
  };

  MutableNbestsRepeatedWrapper* mutable_nbests() {
    if (!mutable_nbests_wrapper_) {
      mutable_nbests_wrapper_ =
          std::make_unique<MutableNbestsRepeatedWrapper>(this);
    }
    return mutable_nbests_wrapper_.get();
  }

  inline NBestSentencePieceText_Sub* mutable_nbests_at(int index);

  static const NBestSentencePieceText& default_instance() {
    static NBestSentencePieceText instance;
    return instance;
  }

 private:
  std::unique_ptr<MutableNbestsRepeatedWrapper> mutable_nbests_wrapper_;
};

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
  return TrainerSpec(
      msg_ ? sentencepiece_ModelProto_trainer_spec(msg_) : nullptr, arena_);
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
  return SelfTestData(msg_ ? sentencepiece_ModelProto_self_test_data(msg_)
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
  if (!msg_) return nullptr;
  sentencepiece_NBestSentencePieceText_add_nbests(msg_, arena_);

  int index = nbests_size() - 1;

  auto wrapper = std::make_unique<NBestSentencePieceText_Sub>(this, index);
  NBestSentencePieceText_Sub* ptr = wrapper.get();
  nbest_wrappers_.push_back(std::move(wrapper));
  return ptr;
}

inline const NBestSentencePieceText_Sub&
upb::NBestSentencePieceTextWrapper::nbests(int index) const {
  if (index < 0 || static_cast<size_t>(index) >= nbest_wrappers_.size()) {
    static const NBestSentencePieceText_Sub default_sub(nullptr, -1);
    return default_sub;
  }
  return *nbest_wrappers_[index];
}

inline const NBestSentencePieceText_Sub& upb::NBestSentencePieceTextWrapper::
    ConstNbestsRepeatedWrapper::Iterator::operator*() const {
  return parent_->nbests(index_);
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

inline const NBestSentencePieceText_Sub& NBestSentencePieceText::nbests(
    int index) const {
  return NBestSentencePieceTextWrapper::nbests(index);
}
inline NBestSentencePieceText_Sub* NBestSentencePieceText::add_nbests() {
  return NBestSentencePieceTextWrapper::add_nbests();
}
inline NBestSentencePieceText_Sub* NBestSentencePieceText::mutable_nbests_at(
    int index) {
  if (nbest_wrappers_.size() != nbests_size()) {
    nbest_wrappers_.resize(nbests_size());
    for (int i = 0; i < nbests_size(); ++i) {
      if (!nbest_wrappers_[i]) {
        nbest_wrappers_[i] =
            std::make_unique<NBestSentencePieceText_Sub>(this, i);
      }
    }
  }
  return nbest_wrappers_[index].get();
}

inline ::sentencepiece::NormalizerSpec*
upb::ModelProtoWrapper::mutable_normalizer_spec() const {
  if (!mutable_normalizer_spec_) {
    if (msg_) {
      sentencepiece_NormalizerSpec* sub_msg =
          sentencepiece_ModelProto_mutable_normalizer_spec(msg_, arena_);
      if (normalizer_spec_cache_) {
        normalizer_spec_cache_->msg_ = sub_msg;
      }
      mutable_normalizer_spec_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(
              sub_msg, arena_,
              [this](const sentencepiece_NormalizerSpec* new_sub_msg) {
                sentencepiece_ModelProto_set_normalizer_spec(
                    this->msg_,
                    const_cast<sentencepiece_NormalizerSpec*>(new_sub_msg));
                if (this->normalizer_spec_cache_) {
                  this->normalizer_spec_cache_->msg_ =
                      const_cast<sentencepiece_NormalizerSpec*>(
                          sentencepiece_ModelProto_normalizer_spec(this->msg_));
                }
              });
    } else {
      mutable_normalizer_spec_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(nullptr);
    }
  }
  return mutable_normalizer_spec_.get();
}

inline ::sentencepiece::NormalizerSpec*
upb::ModelProtoWrapper::mutable_denormalizer_spec() const {
  if (!mutable_denormalizer_spec_) {
    if (msg_) {
      sentencepiece_NormalizerSpec* sub_msg =
          sentencepiece_ModelProto_mutable_denormalizer_spec(msg_, arena_);
      if (denormalizer_spec_cache_) {
        denormalizer_spec_cache_->msg_ = sub_msg;
      }
      mutable_denormalizer_spec_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(
              sub_msg, arena_,
              [this](const sentencepiece_NormalizerSpec* new_sub_msg) {
                sentencepiece_ModelProto_set_denormalizer_spec(
                    this->msg_,
                    const_cast<sentencepiece_NormalizerSpec*>(new_sub_msg));
                if (this->denormalizer_spec_cache_) {
                  this->denormalizer_spec_cache_->msg_ =
                      const_cast<sentencepiece_NormalizerSpec*>(
                          sentencepiece_ModelProto_denormalizer_spec(
                              this->msg_));
                }
              });
    } else {
      mutable_denormalizer_spec_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(nullptr);
    }
  }
  return mutable_denormalizer_spec_.get();
}

inline ModelProto_SentencePiece* ModelProto::add_pieces() {
  if (msg_) {
    sentencepiece_ModelProto_add_pieces(msg_, arena_);
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
  if (!mutable_trainer_spec_) {
    if (msg_) {
      sentencepiece_TrainerSpec* sub_msg =
          sentencepiece_ModelProto_mutable_trainer_spec(msg_, arena_);
      mutable_trainer_spec_ = std::make_unique<::sentencepiece::TrainerSpec>(
          sub_msg, arena_,
          [this](const sentencepiece_TrainerSpec* new_sub_msg) {
            sentencepiece_ModelProto_set_trainer_spec(
                this->msg_,
                const_cast<sentencepiece_TrainerSpec*>(new_sub_msg));
          });
    } else {
      mutable_trainer_spec_ =
          std::make_unique<::sentencepiece::TrainerSpec>(nullptr);
    }
  }
  return mutable_trainer_spec_.get();
}

inline ::sentencepiece::SelfTestData*
upb::ModelProtoWrapper::mutable_self_test_data() const {
  if (!mutable_self_test_data_) {
    if (msg_) {
      sentencepiece_SelfTestData* sub_msg =
          sentencepiece_ModelProto_mutable_self_test_data(msg_, arena_);
      mutable_self_test_data_ =
          std::make_unique<::sentencepiece::SelfTestData>(sub_msg, arena_);
    } else {
      mutable_self_test_data_ =
          std::make_unique<::sentencepiece::SelfTestData>(nullptr, nullptr);
    }
  }
  return mutable_self_test_data_.get();
}

inline void upb::ModelProtoWrapper::set_trainer_spec(const TrainerSpec& spec) {
  if (msg_ && spec.msg_) {
    size_t size = 0;
    upb_Arena* tmp_arena = upb_Arena_New();
    char* buf =
        sentencepiece_TrainerSpec_serialize(spec.msg_, tmp_arena, &size);
    if (buf) {
      sentencepiece_TrainerSpec* sub_msg =
          sentencepiece_TrainerSpec_parse(buf, size, arena_);
      sentencepiece_ModelProto_set_trainer_spec(msg_, sub_msg);
      mutable_trainer_spec_ =
          std::make_unique<::sentencepiece::TrainerSpec>(sub_msg, arena_);
    }
    upb_Arena_Free(tmp_arena);
  }
}

inline void upb::ModelProtoWrapper::set_normalizer_spec(
    const NormalizerSpec& spec) {
  if (msg_ && spec.msg_) {
    size_t size = 0;
    upb_Arena* tmp_arena = upb_Arena_New();
    char* buf =
        sentencepiece_NormalizerSpec_serialize(spec.msg_, tmp_arena, &size);
    if (buf) {
      sentencepiece_NormalizerSpec* sub_msg =
          sentencepiece_NormalizerSpec_parse(buf, size, arena_);
      sentencepiece_ModelProto_set_normalizer_spec(msg_, sub_msg);
      normalizer_spec_cache_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(sub_msg, arena_);
    }
    upb_Arena_Free(tmp_arena);
  }
}

inline void upb::ModelProtoWrapper::set_denormalizer_spec(
    const NormalizerSpec& spec) {
  if (msg_ && spec.msg_) {
    size_t size = 0;
    upb_Arena* tmp_arena = upb_Arena_New();
    char* buf =
        sentencepiece_NormalizerSpec_serialize(spec.msg_, tmp_arena, &size);
    if (buf) {
      sentencepiece_NormalizerSpec* sub_msg =
          sentencepiece_NormalizerSpec_parse(buf, size, arena_);
      sentencepiece_ModelProto_set_denormalizer_spec(msg_, sub_msg);
      mutable_denormalizer_spec_ =
          std::make_unique<::sentencepiece::NormalizerSpec>(sub_msg, arena_);
    }
    upb_Arena_Free(tmp_arena);
  }
}

inline void ModelProto_SentencePiece::set_piece(absl::string_view piece) {
  parent_->set_piece_at(index_, piece);
}
inline void ModelProto_SentencePiece::set_score(float score) {
  parent_->set_score_at(index_, score);
}

}  // namespace sentencepiece

#endif  // CORE_UPB_WRAPPER_H_
