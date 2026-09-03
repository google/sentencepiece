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

// Guards against the project version drifting between VERSION.txt and the
// version the active build system declares for the sentencepiece module.
//
// VERSION (from config.h) is generated from VERSION.txt by both build systems.
// SPM_BUILD_SYSTEM_MODULE_VERSION is whatever the build system declares on its
// own, and each one supplies its own value:
//
//   * Bazel: module(version = ...) in MODULE.bazel, which Bazel requires to be
//     a literal and therefore cannot derive from VERSION.txt. This is the copy
//     that can genuinely drift.
//   * CMake: PROJECT_VERSION, as parsed by project(VERSION ...).
//
// The define is absent when a build system supplies no version of its own, in
// which case the comparison is skipped.

#include <string>

#include "config.h"
#include "gtest/gtest.h"

namespace sentencepiece {

TEST(VersionTest, ConfigVersionIsNotEmpty) {
  EXPECT_FALSE(std::string(VERSION).empty());
}

#ifdef SPM_BUILD_SYSTEM_MODULE_VERSION
TEST(VersionTest, BuildSystemModuleVersionMatchesVersionTxt) {
  EXPECT_EQ(std::string(SPM_BUILD_SYSTEM_MODULE_VERSION), std::string(VERSION))
      << "The build system declares version \""
      << SPM_BUILD_SYSTEM_MODULE_VERSION << "\" but VERSION.txt contains \""
      << VERSION << "\". Update the build system's version to match "
      << "VERSION.txt (for Bazel, the module(version = ...) literal in "
      << "MODULE.bazel).";
}
#endif

}  // namespace sentencepiece
