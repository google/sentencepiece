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

#ifndef TESTHARNESS_H_
#define TESTHARNESS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common.h"

#ifndef EXPECT_OK
#define EXPECT_OK(c) EXPECT_TRUE((c).ok())
#endif
#ifndef ASSERT_OK
#define ASSERT_OK(c) ASSERT_TRUE((c).ok())
#endif

#endif  // TESTHARNESS_H_
