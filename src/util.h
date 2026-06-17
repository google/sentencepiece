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

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common.h"
#include "config.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/base/internal/endian.h"
#include "third_party/absl/base/thread_annotations.h"
#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/numeric/bits.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_builder.h"
#include "third_party/absl/strings/ascii.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/strings/strip.h"
#include "third_party/absl/synchronization/mutex.h"

static constexpr uint32_t kUnicodeError = 0xFFFD;

namespace sentencepiece {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << absl::StrJoin(v, " ");
  return out;
}

uint32_t GetRandomGeneratorSeed();
int GetNBestTimeout();

// Sets data dir containing the global resources, e.g., pre-compiled
// normalization data.
void SetDataDir(absl::string_view data_dir);

std::string GetDataDir();

// String utilities
namespace string_util {

template <typename T>
inline bool DecodePOD(absl::string_view str, T* result) {
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable");
  if (sizeof(*result) != str.size()) {
    return false;
  }
  std::memcpy(result, str.data(), sizeof(T));
  return true;
}

template <typename T>
inline std::string EncodePOD(const T& value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable");
  return {reinterpret_cast<const char*>(&value), sizeof(T)};
}

template <typename T>
inline std::string IntToHex(T value) {
  return absl::StrFormat("%X", value);
}

template <typename T>
inline T HexToInt(absl::string_view value) {
  absl::ConsumePrefix(&value, "0x");
  T n = 0;
  if (!absl::numbers_internal::safe_strtoi_base(value, &n, 16)) {
    return 0;
  }
  return n;
}

template <typename T>
inline std::string SimpleItoa(T val) {
  return absl::StrCat(val);
}

// Return length of a single UTF-8 source character
inline size_t OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// Return (x & 0xC0) == 0x80;
// Since trail bytes are always in [0x80, 0xBF], we can optimize:
inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

inline bool IsValidCodepoint(char32_t c) {
  return (static_cast<uint32_t>(c) < 0xD800) || (c >= 0xE000 && c <= 0x10FFFF);
}

bool IsStructurallyValid(absl::string_view str);

using UnicodeText = std::vector<char32_t>;

char32_t DecodeUTF8(const char* begin, const char* end, size_t* mblen);

inline char32_t DecodeUTF8(absl::string_view input, size_t* mblen) {
  return DecodeUTF8(input.data(), input.data() + input.size(), mblen);
}

inline bool IsValidDecodeUTF8(absl::string_view input, size_t* mblen) {
  const char32_t c = DecodeUTF8(input, mblen);
  return c != kUnicodeError || *mblen == 3;
}

size_t EncodeUTF8(char32_t c, char* output);

std::string UnicodeCharToUTF8(char32_t c);

UnicodeText UTF8ToUnicodeText(absl::string_view utf8);

std::string UnicodeTextToUTF8(const UnicodeText& utext);

struct UnicodeTextAndOffsets {
  UnicodeText unicode_text;
  std::vector<uint32_t> offsets;
};

// - unicode_text is the UTF-8 string converted to UnicodeText.
// - offsets.size() == unicode_text.size() + 1
// - offsets[0] is always 0.
// - offsets[i] is the offset of unicode_text[i] in the original UTF-8 string.
UnicodeTextAndOffsets UTF8ToUnicodeTextAndOffsets(absl::string_view utf8);

}  // namespace string_util

// other map/ptr utilties
namespace port {

template <class Collection, class Key>
bool ContainsKey(const Collection& collection, const Key& key) {
  return collection.find(key) != collection.end();
}

template <class Collection>
const typename Collection::value_type::second_type& FindOrDie(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
  const auto it = collection.find(key);
  //  if (it == collection.end()) {
  //    LOG(FATAL) << "Map key not found: " << key;
  //  }
  return it->second;
}

template <class Collection>
const typename Collection::value_type::second_type& FindWithDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
  if (const auto it = collection.find(key); it != collection.end()) {
    return it->second;
  }
  return value;
}

template <class Collection>
bool InsertIfNotPresent(Collection* const collection,
                        const typename Collection::value_type& vt) {
  return collection->insert(vt).second;
}

template <class Collection>
bool InsertIfNotPresent(
    Collection* const collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
  return InsertIfNotPresent(collection,
                            typename Collection::value_type(key, value));
}

template <class Collection>
void InsertOrDie(Collection* const collection,
                 const typename Collection::value_type::first_type& key,
                 const typename Collection::value_type::second_type& data) {
  CHECK(InsertIfNotPresent(collection, key, data)) << "duplicate key";
}

}  // namespace port

namespace random {

absl::BitGen* GetRandomGenerator();

template <typename T>
class ReservoirSampler {
 public:
  explicit ReservoirSampler(std::vector<T>* sampled, uint64_t size)
      : sampled_(sampled), size_(size) {}
  explicit ReservoirSampler(std::vector<T>* sampled, uint64_t size,
                            uint64_t seed)
      : sampled_(sampled), size_(size), gen_(std::seed_seq{seed}) {}
  virtual ~ReservoirSampler() = default;

  void Add(const T& item) {
    if (size_ == 0) {
      return;
    }

    ++total_;
    if (sampled_->size() < size_) {
      sampled_->push_back(item);
    } else {
      const auto n = absl::Uniform<uint64_t>(gen_, 0, total_ - 1);
      if (n < sampled_->size()) {
        (*sampled_)[n] = item;
      }
    }
  }

  [[nodiscard]] uint64_t total_size() const { return total_; }

 private:
  std::vector<T>* sampled_ = nullptr;
  uint64_t size_ = 0;
  uint64_t total_ = 0;
  absl::BitGen gen_;
};

}  // namespace random

namespace util {

constexpr bool is_bigendian() {
  return absl::endian::native == absl::endian::big;
}

inline uint32_t Swap32(uint32_t x) { return absl::gbswap_32(x); }

inline std::string JoinPath(absl::string_view path) {
  return {path.data(), path.size()};
}

template <typename... T>
inline std::string JoinPath(absl::string_view first, const T&... rest) {
#ifdef OS_WIN
  return absl::StrCat(JoinPath(first), "\\", JoinPath(rest...));
#else
  return absl::StrCat(JoinPath(first), "/", JoinPath(rest...));
#endif
}

std::string StrError(int errnum);

std::vector<std::string> StrSplitAsCSV(absl::string_view text);

#ifdef OS_WIN
std::wstring Utf8ToWide(const absl::string_view input);
#endif

#define RET_CHECK(condition)                                  \
  if (condition) {                                            \
  } else /* NOLINT */                                         \
    return absl::StatusBuilder(::absl::StatusCode::kInternal) \
           << __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

#define RET_CHECK_EQ(a, b) RET_CHECK((a) == (b))
#define RET_CHECK_NE(a, b) RET_CHECK((a) != (b))
#define RET_CHECK_GE(a, b) RET_CHECK((a) >= (b))
#define RET_CHECK_LE(a, b) RET_CHECK((a) <= (b))
#define RET_CHECK_GT(a, b) RET_CHECK((a) > (b))
#define RET_CHECK_LT(a, b) RET_CHECK((a) < (b))

#define RET_QCHECK_EQ(a, b) RET_CHECK_EQ(a, b)
#define RET_QCHECK_NE(a, b) RET_CHECK_NE(a, b)
#define RET_QCHECK_GE(a, b) RET_CHECK_GE(a, b)
#define RET_QCHECK_LE(a, b) RET_CHECK_LE(a, b)
#define RET_QCHECK_GT(a, b) RET_CHECK_GT(a, b)
#define RET_QCHECK_LT(a, b) RET_CHECK_LT(a, b)

}  // namespace util

namespace log_domain {

double LogSum(const std::vector<double>& xs);

}  // namespace log_domain
}  // namespace sentencepiece
#endif  // UTIL_H_
