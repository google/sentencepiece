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

#ifndef RET_CHECK_H_
#define RET_CHECK_H_

#include "absl/status/status.h"
#include "absl/status/status_builder.h"

#ifndef RET_CHECK
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
#endif

#endif  // RET_CHECK_H_
