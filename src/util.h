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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"

static constexpr uint32_t kUnicodeError = 0xFFFD;

namespace sentencepiece {

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
inline T HexToInt(absl::string_view value) {
  T n = 0;
  if (!absl::SimpleHexAtoi(value, &n)) {
    return 0;
  }
  return n;
}

// Return length of a single UTF-8 source character
inline size_t OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// Return (x & 0xC0) == 0x80;
// Since trail bytes are always in [0x80, 0xBF], we can optimize:
inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

// Return the character length of a UTF-8 string without heap allocation.
inline size_t UTF8Len(absl::string_view str) {
  size_t len = 0;
  for (char c : str) {
    if (!IsTrailByte(c)) ++len;
  }
  return len;
}

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

inline std::string JoinPath(absl::string_view path) {
  return {path.data(), path.size()};
}

template <typename... T>
inline std::string JoinPath(absl::string_view first, const T&... rest) {
#if defined(_WIN32) && !defined(__CYGWIN__)
  return absl::StrCat(JoinPath(first), "\\", JoinPath(rest...));
#else
  return absl::StrCat(JoinPath(first), "/", JoinPath(rest...));
#endif
}

std::vector<std::string> StrSplitAsCSV(absl::string_view text);

}  // namespace util

namespace log_domain {

double LogSum(const std::vector<double>& xs);

}  // namespace log_domain
}  // namespace sentencepiece
#endif  // UTIL_H_
