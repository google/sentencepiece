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

#ifndef CORE_UPB_MESSAGE_WRAPPER_H_
#define CORE_UPB_MESSAGE_WRAPPER_H_

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "upb/upb.h"

#define UPB_WRAPPER_RET_IF_OOB(index, size) \
  if ((index) < 0 || static_cast<size_t>(index) >= static_cast<size_t>(size)) return

#define UPB_WRAPPER_RET_VAL_IF_OOB(index, size, val) \
  if ((index) < 0 || static_cast<size_t>(index) >= static_cast<size_t>(size)) return (val)

namespace sentencepiece {
namespace upb_internal {

template <typename MsgType>
class UpbMessageHolder {
 public:
  UpbMessageHolder(MsgType* msg, upb_Arena* arena, bool owns)
      : msg_(reinterpret_cast<upb_Message*>(msg)), arena_(arena), owns_msg_(owns) {}
  UpbMessageHolder() : msg_(nullptr), arena_(nullptr), owns_msg_(false) {}

  ~UpbMessageHolder() {
    Free();
  }

  // Copy is prohibited, Move is allowed to ensure single ownership
  UpbMessageHolder(const UpbMessageHolder&) = delete;
  UpbMessageHolder& operator=(const UpbMessageHolder&) = delete;

  UpbMessageHolder(UpbMessageHolder&& other) noexcept {
    msg_ = other.msg_;
    arena_ = other.arena_;
    owns_msg_ = other.owns_msg_;
    other.msg_ = nullptr;
    other.arena_ = nullptr;
    other.owns_msg_ = false;
  }

  UpbMessageHolder& operator=(UpbMessageHolder&& other) noexcept {
    if (this != &other) {
      Free();
      msg_ = other.msg_;
      arena_ = other.arena_;
      owns_msg_ = other.owns_msg_;
      other.msg_ = nullptr;
      other.arena_ = nullptr;
      other.owns_msg_ = false;
    }
    return *this;
  }

  void Reset(MsgType* msg, upb_Arena* arena, bool owns) {
    Free();
    msg_ = reinterpret_cast<upb_Message*>(msg);
    arena_ = arena;
    owns_msg_ = owns;
  }

  MsgType* msg() const { return reinterpret_cast<MsgType*>(msg_); }
  upb_Arena* arena() const { return arena_; }
  bool owns_msg() const { return owns_msg_; }

 private:
  void Free() {
    if (owns_msg_ && arena_) {
      upb_Arena_Free(arena_);
    }
    msg_ = nullptr;
    arena_ = nullptr;
    owns_msg_ = false;
  }

  upb_Message* msg_ = nullptr;
  upb_Arena* arena_ = nullptr;
  bool owns_msg_ = false;
};

template <typename WrapperType, typename ParentType>
inline void LazyInitCache(std::vector<std::unique_ptr<WrapperType>>& cache,
                          ParentType* parent,
                          size_t expected_size) {
  if (cache.size() != expected_size) {
    cache.resize(expected_size);
    for (size_t i = 0; i < expected_size; ++i) {
      if (!cache[i]) {
        cache[i] = std::make_unique<WrapperType>(parent, i);
      }
    }
  }
}

}  // namespace upb_internal

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

namespace upb_internal {

// 1. Primitive field helpers
template <typename MsgType, typename FieldType>
inline FieldType GetPrimitiveField(const UpbMessageHolder<MsgType>& holder,
                                   FieldType (*get_fn)(const MsgType*),
                                   FieldType default_val) {
  return holder.msg() ? get_fn(holder.msg()) : default_val;
}

template <typename MsgType, typename FieldType>
inline void SetPrimitiveField(UpbMessageHolder<MsgType>& holder,
                              void (*set_fn)(MsgType*, FieldType),
                              FieldType val,
                              const std::function<void(const MsgType*)>& on_change) {
  if (holder.msg()) {
    set_fn(holder.msg(), val);
    if (on_change) on_change(holder.msg());
  }
}

template <typename MsgType>
inline void ClearField(UpbMessageHolder<MsgType>& holder,
                       void (*clear_fn)(MsgType*),
                       const std::function<void(const MsgType*)>& on_change) {
  if (holder.msg()) {
    clear_fn(holder.msg());
    if (on_change) on_change(holder.msg());
  }
}

// 2. String field helpers
template <typename MsgType>
inline ProtoStr GetStringField(const UpbMessageHolder<MsgType>& holder,
                               bool (*has_fn)(const MsgType*),
                               upb_StringView (*get_fn)(const MsgType*),
                               ProtoStr default_val) {
  if (!holder.msg() || !has_fn(holder.msg())) return default_val;
  return get_fn(holder.msg());
}

template <typename MsgType>
inline void SetStringField(UpbMessageHolder<MsgType>& holder,
                           void (*set_fn)(MsgType*, upb_StringView),
                           absl::string_view val,
                           const std::function<void(const MsgType*)>& on_change) {
  if (holder.msg() && holder.arena()) {
    set_fn(holder.msg(), MakeUpbString(val, holder.arena()));
    if (on_change) on_change(holder.msg());
  }
}

template <typename MsgType>
inline bool HasField(const UpbMessageHolder<MsgType>& holder,
                     bool (*has_fn)(const MsgType*)) {
  return holder.msg() && has_fn(holder.msg());
}

// 3. Enum field helpers
template <typename MsgType, typename EnumType, typename RawType>
inline EnumType GetEnumField(const UpbMessageHolder<MsgType>& holder,
                             RawType (*get_fn)(const MsgType*),
                             EnumType default_val) {
  return holder.msg() ? static_cast<EnumType>(get_fn(holder.msg())) : default_val;
}

template <typename MsgType, typename EnumType, typename RawType>
inline void SetEnumField(UpbMessageHolder<MsgType>& holder,
                         void (*set_fn)(MsgType*, RawType),
                         EnumType val,
                         const std::function<void(const MsgType*)>& on_change) {
  if (holder.msg()) {
    set_fn(holder.msg(), static_cast<RawType>(val));
    if (on_change) on_change(holder.msg());
  }
}

// 4. Repeated string field helpers
template <typename MsgType>
inline int GetRepeatedStringSize(const UpbMessageHolder<MsgType>& holder,
                                 const upb_StringView* (*get_fn)(const MsgType*, size_t*)) {
  if (!holder.msg()) return 0;
  size_t size = 0;
  get_fn(holder.msg(), &size);
  return size;
}

template <typename MsgType>
inline ProtoStr GetRepeatedStringElement(const UpbMessageHolder<MsgType>& holder,
                                         const upb_StringView* (*get_fn)(const MsgType*, size_t*),
                                         int index) {
  if (!holder.msg()) return "";
  size_t size = 0;
  const upb_StringView* arr = get_fn(holder.msg(), &size);
  if (index < 0 || static_cast<size_t>(index) >= size) return "";
  return arr[index];
}

template <typename MsgType>
inline RepeatedStringWrapper GetRepeatedStringWrapper(const UpbMessageHolder<MsgType>& holder,
                                                      const upb_StringView* (*get_fn)(const MsgType*, size_t*)) {
  if (!holder.msg()) return RepeatedStringWrapper(nullptr, 0);
  size_t size = 0;
  const upb_StringView* arr = get_fn(holder.msg(), &size);
  return RepeatedStringWrapper(arr, size);
}

template <typename MsgType>
inline void AddRepeatedString(UpbMessageHolder<MsgType>& holder,
                              bool (*add_fn)(MsgType*, upb_StringView, upb_Arena*),
                              const std::string& val,
                              const std::function<void(const MsgType*)>& on_change) {
  if (holder.msg() && holder.arena()) {
    add_fn(holder.msg(), MakeUpbString(val, holder.arena()), holder.arena());
    if (on_change) on_change(holder.msg());
  }
}

}  // namespace upb_internal

#define DEFINE_UPB_PRIMITIVE_ACCESSOR(FieldName, Type, DefaultVal, UpbPrefix) \
  Type FieldName() const {                                                    \
    return sentencepiece::upb_internal::GetPrimitiveField<                    \
        UpbPrefix, Type>(holder_, UpbPrefix##_##FieldName, DefaultVal);        \
  }                                                                           \
  void set_##FieldName(Type val) {                                            \
    sentencepiece::upb_internal::SetPrimitiveField<                           \
        UpbPrefix, Type>(holder_, UpbPrefix##_set_##FieldName, val, on_change_); \
  }                                                                           \
  void clear_##FieldName() {                                                  \
    sentencepiece::upb_internal::ClearField<                                  \
        UpbPrefix>(holder_, UpbPrefix##_clear_##FieldName, on_change_);        \
  }

#define DEFINE_UPB_STRING_ACCESSOR(FieldName, DefaultVal, UpbPrefix)    \
  ProtoStr FieldName() const {                                          \
    return sentencepiece::upb_internal::GetStringField<                 \
        UpbPrefix>(holder_, UpbPrefix##_has_##FieldName,                \
                   UpbPrefix##_##FieldName, DefaultVal);                \
  }                                                                     \
  void set_##FieldName(absl::string_view val) {                         \
    sentencepiece::upb_internal::SetStringField<                        \
        UpbPrefix>(holder_, UpbPrefix##_set_##FieldName, val, on_change_); \
  }                                                                     \
  void set_##FieldName(const char* data, size_t size) {                 \
    set_##FieldName(absl::string_view(data, size));                     \
  }                                                                     \
  void clear_##FieldName() {                                            \
    sentencepiece::upb_internal::ClearField<                            \
        UpbPrefix>(holder_, UpbPrefix##_clear_##FieldName, on_change_);  \
  }

#define DEFINE_UPB_HAS_FIELD_ACCESSOR(FieldName, UpbPrefix) \
  bool has_##FieldName() const {                            \
    return sentencepiece::upb_internal::HasField<           \
        UpbPrefix>(holder_, UpbPrefix##_has_##FieldName);   \
  }

#define DEFINE_UPB_ENUM_ACCESSOR(FieldName, EnumType, DefaultVal, UpbPrefix) \
  EnumType FieldName() const {                                               \
    return sentencepiece::upb_internal::GetEnumField<                        \
        UpbPrefix, EnumType, int32_t>(holder_, UpbPrefix##_##FieldName, DefaultVal); \
  }                                                                          \
  void set_##FieldName(EnumType type) {                                      \
    sentencepiece::upb_internal::SetEnumField<                               \
        UpbPrefix, EnumType, int32_t>(holder_, UpbPrefix##_set_##FieldName, type, on_change_); \
  }                                                                          \
  void clear_##FieldName() {                                                 \
    sentencepiece::upb_internal::ClearField<                                 \
        UpbPrefix>(holder_, UpbPrefix##_clear_##FieldName, on_change_);       \
  }

#define DEFINE_UPB_REPEATED_STRING_ACCESSOR(FieldName, UpbPrefix)            \
  int FieldName##_size() const {                                             \
    return sentencepiece::upb_internal::GetRepeatedStringSize<               \
        UpbPrefix>(holder_, UpbPrefix##_##FieldName);                        \
  }                                                                          \
  ProtoStr FieldName(int index) const {                                      \
    return sentencepiece::upb_internal::GetRepeatedStringElement<            \
        UpbPrefix>(holder_, UpbPrefix##_##FieldName, index);                 \
  }                                                                          \
  RepeatedStringWrapper FieldName() const {                                  \
    return sentencepiece::upb_internal::GetRepeatedStringWrapper<            \
        UpbPrefix>(holder_, UpbPrefix##_##FieldName);                        \
  }                                                                          \
  void add_##FieldName(const std::string& val) {                             \
    sentencepiece::upb_internal::AddRepeatedString<                          \
        UpbPrefix>(holder_, UpbPrefix##_add_##FieldName, val, on_change_);   \
  }                                                                          \
  void clear_##FieldName() {                                                 \
    sentencepiece::upb_internal::ClearField<                                 \
        UpbPrefix>(holder_, UpbPrefix##_clear_##FieldName, on_change_);       \
  }

namespace upb {

#define DEFINE_UPB_SERIALIZATION_METHODS(ClassName, CPrefix)                   \
  std::string SerializeAsString() const {                                      \
    if (!holder_.msg()) return "";                                             \
    size_t len;                                                                \
    upb_Arena* tmp_arena = upb_Arena_New();                                    \
    char* ptr = CPrefix##_serialize(holder_.msg(), tmp_arena, &len);           \
    std::string ret;                                                           \
    if (ptr) {                                                                 \
      ret.assign(ptr, len);                                                    \
    }                                                                          \
    upb_Arena_Free(tmp_arena);                                                 \
    return ret;                                                                \
  }                                                                            \
  bool ParseFromArray(const void* data, int size) {                            \
    upb_Arena* new_arena = upb_Arena_New();                                    \
    CPrefix* new_msg = CPrefix##_parse(reinterpret_cast<const char*>(data), size, new_arena); \
    if (!new_msg) {                                                            \
      upb_Arena_Free(new_arena);                                               \
      return false;                                                            \
    }                                                                          \
    holder_.Reset(new_msg, new_arena, true);                                   \
    OnArenaReset();                                                            \
    return true;                                                               \
  }                                                                            \
  bool ParseFromString(const std::string& bytes) {                             \
    return ParseFromArray(bytes.data(), bytes.size());                         \
  }                                                                            \
  void CopyFrom(const ClassName& other) {                                      \
    std::string bytes = other.SerializeAsString();                             \
    ParseFromString(bytes);                                                    \
  }                                                                            \
  void Clear() {                                                               \
    upb_Arena* new_arena = upb_Arena_New();                                    \
    CPrefix* new_msg = CPrefix##_new(new_arena);                           \
    holder_.Reset(new_msg, new_arena, true);                                   \
    OnArenaReset();                                                            \
  }

#define UPB_WRAPPER_COPY_AND_SET_SUB_MSG(ParentMsg, SrcWrapper, SetFn, ParseFn, CacheVar, WrapperClass, Arena) \
  do {                                                                         \
    std::string bytes = (SrcWrapper).SerializeAsString();                      \
    if (!bytes.empty()) {                                                      \
      auto* sub_msg = ParseFn(bytes.data(), bytes.size(), Arena);              \
      if (sub_msg) {                                                           \
        SetFn(ParentMsg, sub_msg);                                             \
        (CacheVar) = std::make_unique<WrapperClass>(sub_msg, Arena);           \
      }                                                                        \
    }                                                                          \
  } while (0)

#define DEFINE_UPB_ASSIGNMENT_OPERATOR(ClassName, CPrefix)                     \
  ClassName& operator=(const ClassName& other) {                              \
    if (this != &other) {                                                     \
      if (holder_.owns_msg()) {                                               \
        Clear();                                                              \
        CopyFrom(other);                                                      \
      } else {                                                                \
        if (other.holder_.msg()) {                                            \
          size_t size = 0;                                                    \
          upb_Arena* tmp_arena = upb_Arena_New();                             \
          char* buf =                                                         \
              CPrefix##_serialize(other.holder_.msg(), tmp_arena, &size);     \
          if (buf) {                                                          \
            auto* parsed_msg =                                                \
                CPrefix##_parse(buf, size, holder_.arena());                  \
            holder_.Reset(parsed_msg, holder_.arena(), false);                \
            if (on_change_) on_change_(holder_.msg());                        \
          }                                                                   \
          upb_Arena_Free(tmp_arena);                                          \
        } else {                                                              \
          holder_.Reset(nullptr, holder_.arena(), false);                     \
          if (on_change_) on_change_(nullptr);                                \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    return *this;                                                             \
  }

}  // namespace upb
namespace upb_internal {


// Generic const repeated field wrapper (ForwardIterator)
// Generic const repeated field wrapper (ForwardIterator)
template <typename Derived, typename ElementWrapperType>
class ConstRepeatedWrapperBase {
 public:
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ElementWrapperType;
    using difference_type = std::ptrdiff_t;
    using pointer = ElementWrapperType*;
    using reference = ElementWrapperType;

    Iterator(const ConstRepeatedWrapperBase* container, int index)
        : container_(container), index_(index) {}

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

    ElementWrapperType operator*() const {
      return static_cast<const Derived*>(container_)->Get(index_);
    }

   private:
    const ConstRepeatedWrapperBase* container_;
    int index_;
  };

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, Size()); }
  int size() const { return Size(); }
  bool empty() const { return size() == 0; }
  ElementWrapperType operator[](int index) const {
    return static_cast<const Derived*>(this)->Get(index);
  }

 private:
  int Size() const {
    return static_cast<const Derived*>(this)->Size();
  }
};

// Generic mutable repeated field wrapper (RandomAccessIterator)
template <typename Derived, typename ElementWrapperType>
class MutableRepeatedWrapperBase {
 public:
  class Iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ElementWrapperType;
    using difference_type = std::ptrdiff_t;
    using pointer = ElementWrapperType*;
    using reference = ElementWrapperType&;

    Iterator(MutableRepeatedWrapperBase* container, int index)
        : container_(container), index_(index) {}

    ElementWrapperType& operator*() const {
      return *static_cast<Derived*>(container_)->GetMutable(index_);
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

    ElementWrapperType& operator[](difference_type n) const {
      return *static_cast<Derived*>(container_)->GetMutable(index_ + n);
    }

    bool operator==(const Iterator& other) const {
      return index_ == other.index_ && container_ == other.container_;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }
    bool operator<(const Iterator& other) const { return index_ < other.index_; }
    bool operator>(const Iterator& other) const { return index_ > other.index_; }
    bool operator<=(const Iterator& other) const { return index_ <= other.index_; }
    bool operator>=(const Iterator& other) const { return index_ >= other.index_; }

   private:
    MutableRepeatedWrapperBase* container_;
    int index_;
  };

  Iterator begin() { return Iterator(this, 0); }
  Iterator end() { return Iterator(this, Size()); }
  int size() const { return Size(); }
  bool empty() const { return size() == 0; }
  ElementWrapperType* Mutable(int index) {
    return static_cast<Derived*>(this)->GetMutable(index);
  }

 private:
  int Size() const {
    return static_cast<const Derived*>(this)->Size();
  }
};

}  // namespace upb_internal
}  // namespace sentencepiece

#endif  // CORE_UPB_MESSAGE_WRAPPER_H_
