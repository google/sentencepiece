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

#ifndef COMMON_H_
#define COMMON_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "config.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#define OS_WIN
#else
#define OS_UNIX
#endif

#ifdef OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t char32;
typedef uint32_t uint32;
typedef uint64_t uint64;

static constexpr uint8 kuint8max = ((uint8)0xFF);
static constexpr uint16 kuint16max = ((uint16)0xFFFF);
static constexpr uint32 kuint32max = ((uint32)0xFFFFFFFF);
static constexpr uint64 kuint64max = ((uint64)(0xFFFFFFFFFFFFFFFF));
static constexpr int8 kint8min = ((int8)~0x7F);
static constexpr int8 kint8max = ((int8)0x7F);
static constexpr int16 kint16min = ((int16)~0x7FFF);
static constexpr int16 kint16max = ((int16)0x7FFF);
static constexpr int32 kint32min = ((int32)~0x7FFFFFFF);
static constexpr int32 kint32max = ((int32)0x7FFFFFFF);
static constexpr int64 kint64min = ((int64)(~0x7FFFFFFFFFFFFFFF));
static constexpr int64 kint64max = ((int64)(0x7FFFFFFFFFFFFFFF));

static constexpr uint32 kUnicodeError = 0xFFFD;

#if defined(OS_WIN) && defined(UNICODE) && defined(_UNICODE)
#define WPATH(path) (::sentencepiece::win32::Utf8ToWide(path).c_str())
#else
#define WPATH(path) (path)
#endif

template <typename T, size_t N>
char (&ArraySizeHelper(T (&array)[N]))[N];

#ifndef _MSC_VER
template <typename T, size_t N>
char (&ArraySizeHelper(const T (&array)[N]))[N];
#endif  // !_MSC_VER

#define arraysize(array) (sizeof(ArraySizeHelper(array)))

namespace sentencepiece {
#ifdef OS_WIN
namespace win32 {
std::wstring Utf8ToWide(const std::string &input);
std::string WideToUtf8(const std::wstring &input);
}  // namespace win32
#endif

namespace error {

void Abort();
void Exit(int code);
void SetTestCounter(int c);
void ResetTestMode();
bool GetTestCounter();

class Die {
 public:
  explicit Die(bool die) : die_(die) {}
  ~Die() {
    std::cerr << std::endl;
    if (die_) {
      Abort();
    }
  }
  int operator&(std::ostream &) { return 0; }

 private:
  bool die_;
};

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (t == nullptr) {
    std::cerr << file << "(" << line << ") " << exprtext;
    Abort();
  }
  return std::forward<T>(t);
}
}  // namespace error

namespace logging {
enum LogSeverity {
  LOG_INFO = 0,
  LOG_WARNING = 1,
  LOG_ERROR = 2,
  LOG_FATAL = 3,
  LOG_SEVERITY_SIZE = 4,
};

int GetMinLogLevel();
void SetMinLogLevel(int v);

inline const char *BaseName(const char *path) {
#ifdef OS_WIN
  const char *p = strrchr(path, '\\');
#else
  const char *p = strrchr(path, '/');
#endif
  if (p == nullptr) return path;
  return p + 1;
}
}  // namespace logging
}  // namespace sentencepiece

#define LOG(severity)                                                        \
  (::sentencepiece::logging::GetMinLogLevel() >                              \
   ::sentencepiece::logging::LOG_##severity)                                 \
      ? 0                                                                    \
      : ::sentencepiece::error::Die(                                         \
            ::sentencepiece::logging::LOG_##severity >=                      \
            ::sentencepiece::logging::LOG_FATAL) &                           \
            std::cerr << ::sentencepiece::logging::BaseName(__FILE__) << "(" \
                      << __LINE__ << ") "                                    \
                      << "LOG(" << #severity << ") "

#define CHECK(condition)                                                      \
  (condition) ? 0                                                             \
              : ::sentencepiece::error::Die(true) &                           \
                    std::cerr << ::sentencepiece::logging::BaseName(__FILE__) \
                              << "(" << __LINE__ << ") [" << #condition       \
                              << "] "

#define CHECK_STREQ(a, b) CHECK_EQ(std::string(a), std::string(b))
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_NOTNULL(val)                                    \
  ::sentencepiece::error::CheckNotNull(                       \
      ::sentencepiece::logging::BaseName(__FILE__), __LINE__, \
      "'" #val "' Must be non NULL", (val))

#define FRIEND_TEST(a, b) friend class a##_Test_##b;

#define CHECK_OK(expr)                         \
  do {                                         \
    const auto _status = expr;                 \
    CHECK(_status.ok()) << _status.ToString(); \
  } while (0)

#define CHECK_NOT_OK(expr)                      \
  do {                                          \
    const auto _status = expr;                  \
    CHECK(!_status.ok()) << _status.ToString(); \
  } while (0)

#define RETURN_IF_ERROR(expr)          \
  do {                                 \
    const auto _status = expr;         \
    if (!_status.ok()) return _status; \
  } while (0)

#endif  // COMMON_H_
