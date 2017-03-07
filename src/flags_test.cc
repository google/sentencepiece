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

#include "flags.h"
#include "common.h"
#include "testharness.h"

DEFINE_int32(int32_f, 10, "int32_flags");
DEFINE_bool(bool_f, false, "bool_flags");
DEFINE_int64(int64_f, 20, "int64_flags");
DEFINE_uint64(uint64_f, 30, "uint64_flags");
DEFINE_double(double_f, 40.0, "double_flags");
DEFINE_string(string_f, "str", "string_flags");

namespace sentencepiece {
namespace flags {

TEST(FlagsTest, DefaultValueTest) {
  EXPECT_EQ(10, FLAGS_int32_f);
  EXPECT_EQ(false, FLAGS_bool_f);
  EXPECT_EQ(20, FLAGS_int64_f);
  EXPECT_EQ(30, FLAGS_uint64_f);
  EXPECT_EQ(40.0, FLAGS_double_f);
  EXPECT_EQ("str", FLAGS_string_f);
}

TEST(FlagsTest, PrintHelpTest) {
  const std::string help = PrintHelp("foobar");
  EXPECT_NE(std::string::npos, help.find("foobar"));
  EXPECT_NE(std::string::npos, help.find("int32_flags"));
  EXPECT_NE(std::string::npos, help.find("bool_flags"));
  EXPECT_NE(std::string::npos, help.find("int64_flags"));
  EXPECT_NE(std::string::npos, help.find("uint64_flags"));
  EXPECT_NE(std::string::npos, help.find("double_flags"));
  EXPECT_NE(std::string::npos, help.find("string_flags"));
}

TEST(FlagsTest, ParseCommandLineFlagsTest) {
  const char *kFlags[] = {"program",        "--int32_f=100",  "other1",
                          "--bool_f=true",  "--int64_f=200",  "--uint64_f=300",
                          "--double_f=400", "--string_f=foo", "other2",
                          "other3"};

  std::vector<std::string> rest;
  ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags), &rest);

  EXPECT_EQ(100, FLAGS_int32_f);
  EXPECT_EQ(true, FLAGS_bool_f);
  EXPECT_EQ(200, FLAGS_int64_f);
  EXPECT_EQ(300, FLAGS_uint64_f);
  EXPECT_EQ(400.0, FLAGS_double_f);
  EXPECT_EQ("foo", FLAGS_string_f);
  EXPECT_EQ(3, rest.size());
  EXPECT_EQ("other1", rest[0]);
  EXPECT_EQ("other2", rest[1]);
  EXPECT_EQ("other3", rest[2]);
}

TEST(FlagsTest, ParseCommandLineFlagsTest2) {
  const char *kFlags[] = {"program",       "--int32_f", "500",
                          "-int64_f=600",  "-uint64_f", "700",
                          "--bool_f=FALSE"};

  std::vector<std::string> rest;
  ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags), &rest);

  EXPECT_EQ(500, FLAGS_int32_f);
  EXPECT_EQ(600, FLAGS_int64_f);
  EXPECT_EQ(700, FLAGS_uint64_f);
  EXPECT_FALSE(FLAGS_bool_f);
  EXPECT_TRUE(rest.empty());
}

TEST(FlagsTest, ParseCommandLineFlagsTest3) {
  const char *kFlags[] = {"program", "--bool_f", "--int32_f", "800"};

  std::vector<std::string> rest;
  ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags), &rest);

  EXPECT_TRUE(FLAGS_bool_f);
  EXPECT_EQ(800, FLAGS_int32_f);
  EXPECT_TRUE(rest.empty());
}

TEST(FlagsTest, ParseCommandLineFlagsHelpTest) {
  const char *kFlags[] = {"program", "--help"};
  EXPECT_DEATH(
      ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags)));
}

TEST(FlagsTest, ParseCommandLineFlagsVersionTest) {
  const char *kFlags[] = {"program", "--version"};
  EXPECT_DEATH(
      ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags)));
}

TEST(FlagsTest, ParseCommandLineFlagsUnknownTest) {
  const char *kFlags[] = {"program", "--foo"};
  EXPECT_DEATH(
      ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags)));
}

TEST(FlagsTest, ParseCommandLineFlagsInvalidBoolTest) {
  const char *kFlags[] = {"program", "--bool_f=X"};
  EXPECT_DEATH(
      ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags)));
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyStringArgs) {
  const char *kFlags[] = {"program", "--string_f="};
  ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags));
  EXPECT_EQ("", FLAGS_string_f);
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyBoolArgs) {
  const char *kFlags[] = {"program", "--bool_f"};
  ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags));
  EXPECT_TRUE(FLAGS_bool_f);
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyIntArgs) {
  const char *kFlags[] = {"program", "--int32_f"};
  EXPECT_DEATH(
      ParseCommandLineFlags(arraysize(kFlags), const_cast<char **>(kFlags)));
}
}  // namespace flags
}  // namespace sentencepiece
