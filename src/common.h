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

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/deps/basic_types.h"
#include "src/deps/canonical_errors.h"
#include "src/deps/status.h"
#include "src/deps/status_macros.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#define OS_WIN
#else
#define OS_UNIX
#endif

namespace sentencepiece {
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
}  // namespace error

namespace logging {
enum LogSeverity {
  LOG_INFO = 0,
  LOG_WARNING = 1,
  LOG_ERROR = 2,
  LOG_FATAL = 3,
  LOG_SEVERITY_SIZE = 4,
};

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

#define LOG(severity)                                                     \
  ::sentencepiece::error::Die(::sentencepiece::logging::LOG_##severity >= \
                              ::sentencepiece::logging::LOG_FATAL) &      \
      std::cerr << ::sentencepiece::logging::BaseName(__FILE__) << "("    \
                << __LINE__ << ") "                                       \
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
      "'" #val "' Must be non NULL", (val)

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
