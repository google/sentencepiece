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

#include "filesystem.h"

#include "testharness.h"
#include "util.h"

namespace sentencepiece {

TEST(UtilTest, FilesystemTest) {
  test::ScopedTempFile sf("test_file");

  const char *kData[] = {
      "This"
      "is"
      "a"
      "test"};

  {
    auto output = filesystem::NewWritableFile(sf.filename());
    for (size_t i = 0; i < arraysize(kData); ++i) {
      output->WriteLine(kData[i]);
    }
  }

  {
    auto input = filesystem::NewReadableFile(sf.filename());
    std::string line;
    for (size_t i = 0; i < arraysize(kData); ++i) {
      EXPECT_TRUE(input->ReadLine(&line));
      EXPECT_EQ(kData[i], line);
    }
    EXPECT_FALSE(input->ReadLine(&line));
  }
}

TEST(UtilTest, FilesystemInvalidFileTest) {
  auto input = filesystem::NewReadableFile("__UNKNOWN__FILE__");
  EXPECT_NOT_OK(input->status());
}

}  // namespace sentencepiece
