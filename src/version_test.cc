// Copyright 2026 Google Inc.
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

// Guards against the project version drifting between its sources of truth.
//
// VERSION (from config.h) is generated from VERSION.txt by both CMake and the
// Bazel //src:config_h genrule. Bazel's module(version = ...) in MODULE.bazel
// cannot read a file, so it is a hand-maintained literal; the build passes it
// in as SPM_MODULE_VERSION via module_version() so this test can compare them.

#include <string>

#include "config.h"
#include "gtest/gtest.h"

namespace sentencepiece {

TEST(VersionTest, ConfigVersionIsNotEmpty) {
  EXPECT_FALSE(std::string(VERSION).empty());
}

#ifdef SPM_MODULE_VERSION
TEST(VersionTest, BazelModuleVersionMatchesVersionTxt) {
  EXPECT_EQ(std::string(SPM_MODULE_VERSION), std::string(VERSION))
      << "MODULE.bazel declares module(version = \"" << SPM_MODULE_VERSION
      << "\") but VERSION.txt contains \"" << VERSION
      << "\". Update MODULE.bazel to match VERSION.txt.";
}
#endif

}  // namespace sentencepiece
