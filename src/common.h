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
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_macros.h"
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

#define FRIEND_TEST(a, b) friend class a##_Test_##b;

#ifndef RETURN_IF_ERROR
#define RETURN_IF_ERROR(...) ABSL_RETURN_IF_ERROR(__VA_ARGS__)
#endif

#endif  // COMMON_H_
