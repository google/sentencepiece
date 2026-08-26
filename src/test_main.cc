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

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "filesystem.h"
#include "init.h"

ABSL_FLAG(std::string, test_srcdir,
          sentencepiece::filesystem::JoinPath("..", "data"), "Data directory.");
ABSL_FLAG(std::string, test_tmpdir, "test_tmp", "Temporary directory.");

int main(int argc, char** argv) {
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  // Set TEST_SRCDIR environment variable so GoogleTest's native
  // testing::SrcDir() can locate test data files when --test_srcdir is passed.
#if defined(_WIN32) && !defined(__CYGWIN__)
  _putenv_s("TEST_SRCDIR", absl::GetFlag(FLAGS_test_srcdir).c_str());
#else
  setenv("TEST_SRCDIR", absl::GetFlag(FLAGS_test_srcdir).c_str(), 1);
#endif
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
