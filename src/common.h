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

#include <cstdint>
#include <iostream>

#include "third_party/absl/log/check.h"
#include "third_party/absl/log/globals.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/strings/string_view.h"

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

using char32 = uint32_t;

#define FRIEND_TEST(a, b) friend class a##_Test_##b;

#define RETURN_IF_ERROR(expr)          \
  do {                                 \
    const auto _status = expr;         \
    if (!_status.ok()) return _status; \
  } while (0)

// CHECK_OK must work on absl::Status, not absl::Status.
#if defined CHECK_OK
#undef CHECK_OK
#endif  // CHECK_OK

#if defined QCHECK_OK
#undef QCHECK_OK
#endif  // QCHECK_OK

#define CHECK_OK(expr)                         \
  do {                                         \
    const auto _status = expr;                 \
    CHECK(_status.ok()) << _status.ToString(); \
  } while (0)

#define QCHECK_OK(expr)                         \
  do {                                          \
    const auto _status = expr;                  \
    QCHECK(_status.ok()) << _status.ToString(); \
  } while (0)

#endif  // COMMON_H_
