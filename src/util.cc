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

#include "util.h"
#include <iostream>

namespace sentencepiece {
namespace string_util {

template <typename T>
std::vector<T> SplitInternal(const T &str, const T &delim, bool allow_empty) {
  std::vector<T> result;
  size_t current_pos = 0;
  size_t found_pos = 0;
  while ((found_pos = str.find_first_of(delim, current_pos)) != T::npos) {
    if ((allow_empty && found_pos >= current_pos) ||
        (!allow_empty && found_pos > current_pos)) {
      result.push_back(str.substr(current_pos, found_pos - current_pos));
    }
    current_pos = found_pos + 1;
  }
  if (str.size() > current_pos) {
    result.push_back(str.substr(current_pos, str.size() - current_pos));
  }
  return result;
}

std::vector<std::string> Split(const std::string &str, const std::string &delim,
                               bool allow_empty) {
  return SplitInternal<std::string>(str, delim, allow_empty);
}

std::vector<absl::string_view> SplitPiece(absl::string_view str,
                                          absl::string_view delim,
                                          bool allow_empty) {
  return SplitInternal<absl::string_view>(str, delim, allow_empty);
}

std::string Join(const std::vector<std::string> &tokens,
                 absl::string_view delim) {
  std::string result;
  if (!tokens.empty()) {
    result.append(tokens[0]);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    result.append(tokens[i]);
  }
  return result;
}

std::string Join(const std::vector<int> &tokens, absl::string_view delim) {
  std::string result;
  char buf[32];
  if (!tokens.empty()) {
    const size_t len = Itoa(tokens[0], buf);
    result.append(buf, len);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    const size_t len = Itoa(tokens[i], buf);
    result.append(buf, len);
  }
  return result;
}

std::string StringReplace(absl::string_view s, absl::string_view oldsub,
                          absl::string_view newsub, bool replace_all) {
  std::string ret;
  StringReplace(s, oldsub, newsub, replace_all, &ret);
  return ret;
}

void StringReplace(absl::string_view s, absl::string_view oldsub,
                   absl::string_view newsub, bool replace_all,
                   std::string *res) {
  if (oldsub.empty()) {
    res->append(s.data(), s.size());
    return;
  }

  absl::string_view::size_type start_pos = 0;
  do {
    const absl::string_view::size_type pos = s.find(oldsub, start_pos);
    if (pos == absl::string_view::npos) {
      break;
    }
    res->append(s.data() + start_pos, pos - start_pos);
    res->append(newsub.data(), newsub.size());
    start_pos = pos + oldsub.size();
  } while (replace_all);
  res->append(s.data() + start_pos, s.size() - start_pos);
}

// mblen sotres the number of bytes consumed after decoding.
char32 DecodeUTF8(const char *begin, const char *end, size_t *mblen) {
  const size_t len = end - begin;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
    const char32 cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
    if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
      *mblen = 2;
      return cp;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
    const char32 cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
        IsValidCodepoint(cp)) {
      *mblen = 3;
      return cp;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
    const char32 cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
        IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
      *mblen = 4;
      return cp;
    }
  }

  // Invalid UTF-8.
  *mblen = 1;
  return kUnicodeError;
}

bool IsStructurallyValid(absl::string_view str) {
  const char *begin = str.data();
  const char *end = str.data() + str.size();
  size_t mblen = 0;
  while (begin < end) {
    const char32 c = DecodeUTF8(begin, end, &mblen);
    if (c == kUnicodeError && mblen != 3) return false;
    if (!IsValidCodepoint(c)) return false;
    begin += mblen;
  }
  return true;
}

size_t EncodeUTF8(char32 c, char *output) {
  if (c <= 0x7F) {
    *output = static_cast<char>(c);
    return 1;
  }

  if (c <= 0x7FF) {
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xC0 | c;
    return 2;
  }

  // if `c` is out-of-range, convert it to REPLACEMENT CHARACTER (U+FFFD).
  // This treatment is the same as the original runetochar.
  if (c > 0x10FFFF) c = kUnicodeError;

  if (c <= 0xFFFF) {
    output[2] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xE0 | c;
    return 3;
  }

  output[3] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[2] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[1] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[0] = 0xF0 | c;

  return 4;
}

std::string UnicodeCharToUTF8(const char32 c) { return UnicodeTextToUTF8({c}); }

UnicodeText UTF8ToUnicodeText(absl::string_view utf8) {
  UnicodeText uc;
  const char *begin = utf8.data();
  const char *end = utf8.data() + utf8.size();
  while (begin < end) {
    size_t mblen;
    const char32 c = DecodeUTF8(begin, end, &mblen);
    uc.push_back(c);
    begin += mblen;
  }
  return uc;
}

std::string UnicodeTextToUTF8(const UnicodeText &utext) {
  char buf[8];
  std::string result;
  for (const char32 c : utext) {
    const size_t mblen = EncodeUTF8(c, buf);
    result.append(buf, mblen);
  }
  return result;
}
}  // namespace string_util

namespace random {
#ifdef SPM_NO_THREADLOCAL
namespace {
class RandomGeneratorStorage {
 public:
  RandomGeneratorStorage() {
    pthread_key_create(&key_, &RandomGeneratorStorage::Delete);
  }
  virtual ~RandomGeneratorStorage() { pthread_key_delete(key_); }

  std::mt19937 *Get() {
    auto *result = static_cast<std::mt19937 *>(pthread_getspecific(key_));
    if (result == nullptr) {
      result = new std::mt19937(std::random_device{}());
      pthread_setspecific(key_, result);
    }
    return result;
  }

 private:
  static void Delete(void *value) { delete static_cast<std::mt19937 *>(value); }
  pthread_key_t key_;
};
}  // namespace

std::mt19937 *GetRandomGenerator() {
  static RandomGeneratorStorage *storage = new RandomGeneratorStorage;
  return storage->Get();
}
#else
std::mt19937 *GetRandomGenerator() {
  thread_local static std::mt19937 mt(std::random_device{}());
  return &mt;
}
#endif
}  // namespace random

namespace util {

std::string StrError(int errnum) {
  constexpr int kStrErrorSize = 1024;
  char buffer[kStrErrorSize];
  char *str = nullptr;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  str = strerror_r(errnum, buffer, kStrErrorSize - 1);
#elif defined(_WIN32)
  strerror_s(buffer, kStrErrorSize - 1, errnum);
  str = buffer;
#else
  strerror_r(errnum, buffer, kStrErrorSize - 1);
  str = buffer;
#endif
  std::ostringstream os;
  os << str << " Error #" << errnum;
  return os.str();
}
}  // namespace util

#ifdef OS_WIN
namespace win32 {
std::wstring Utf8ToWide(const std::string &input) {
  int output_length =
      ::MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
  output_length = output_length <= 0 ? 0 : output_length - 1;
  if (output_length == 0) {
    return L"";
  }
  std::unique_ptr<wchar_t[]> input_wide(new wchar_t[output_length + 1]);
  const int result = ::MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1,
                                           input_wide.get(), output_length + 1);
  std::wstring output;
  if (result > 0) {
    output.assign(input_wide.get());
  }
  return output;
}

std::string WideToUtf8(const std::wstring &input) {
  const int output_length = ::WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1,
                                                  nullptr, 0, nullptr, nullptr);
  if (output_length == 0) {
    return "";
  }

  std::unique_ptr<char[]> input_encoded(new char[output_length + 1]);
  const int result =
      ::WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, input_encoded.get(),
                            output_length + 1, nullptr, nullptr);
  std::string output;
  if (result > 0) {
    output.assign(input_encoded.get());
  }
  return output;
}
}  // namespace win32
#endif
}  // namespace sentencepiece
