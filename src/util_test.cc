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

#include "util.h"
#include <map>
#include "testharness.h"

namespace sentencepiece {

TEST(UtilTest, CheckNotNullTest) {
  int a = 0;
  CHECK_NOTNULL(&a);
  EXPECT_DEATH(CHECK_NOTNULL(nullptr));
}

TEST(UtilTest, StartsWith) {
  const std::string str = "abcdefg";
  EXPECT_TRUE(string_util::StartsWith(str, ""));
  EXPECT_TRUE(string_util::StartsWith(str, "a"));
  EXPECT_TRUE(string_util::StartsWith(str, "abc"));
  EXPECT_TRUE(string_util::StartsWith(str, "abcdefg"));
  EXPECT_FALSE(string_util::StartsWith(str, "abcdefghi"));
  EXPECT_FALSE(string_util::StartsWith(str, "foobar"));
}

TEST(UtilTest, EndsWith) {
  const std::string str = "abcdefg";
  EXPECT_TRUE(string_util::EndsWith(str, ""));
  EXPECT_TRUE(string_util::EndsWith(str, "g"));
  EXPECT_TRUE(string_util::EndsWith(str, "fg"));
  EXPECT_TRUE(string_util::EndsWith(str, "abcdefg"));
  EXPECT_FALSE(string_util::EndsWith(str, "aaabcdefg"));
  EXPECT_FALSE(string_util::EndsWith(str, "foobar"));
  EXPECT_FALSE(string_util::EndsWith(str, "foobarbuzbuz"));
}

TEST(UtilTest, Hex) {
  for (char32 a = 0; a < 100000; ++a) {
    const std::string s = string_util::IntToHex<char32>(a);
    CHECK_EQ(a, string_util::HexToInt<char32>(s));
  }

  const int n = 151414;
  CHECK_EQ("24F76", string_util::IntToHex(n));
  CHECK_EQ(n, string_util::HexToInt<int>("24F76"));
}

TEST(UtilTest, SplitTest) {
  std::vector<std::string> tokens;

  tokens = string_util::Split("this is a\ttest", " \t");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::Split("this is a  \t  test", " \t");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::Split("this is a\ttest", " ");
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a\ttest");

  tokens = string_util::Split("  this is a test  ", " ");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::Split("", "");
  EXPECT_TRUE(tokens.empty());
}

TEST(UtilTest, SplitPieceTest) {
  std::vector<StringPiece> tokens;

  tokens = string_util::SplitPiece("this is a\ttest", " \t");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::SplitPiece("this is a  \t  test", " \t");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::SplitPiece("this is a\ttest", " ");
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a\ttest");

  tokens = string_util::SplitPiece("  this is a test  ", " ");
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ(tokens[0], "this");
  EXPECT_EQ(tokens[1], "is");
  EXPECT_EQ(tokens[2], "a");
  EXPECT_EQ(tokens[3], "test");

  tokens = string_util::SplitPiece("", "");
  EXPECT_TRUE(tokens.empty());
}

TEST(UtilTest, JoinTest) {
  std::vector<std::string> tokens;
  tokens.push_back("this");
  tokens.push_back("is");
  tokens.push_back("a");
  tokens.push_back("test");
  EXPECT_EQ(string_util::Join(tokens, " "), "this is a test");
  EXPECT_EQ(string_util::Join(tokens, ":"), "this:is:a:test");
  EXPECT_EQ(string_util::Join(tokens, ""), "thisisatest");
  tokens[2] = "";
  EXPECT_EQ(string_util::Join(tokens, " "), "this is  test");
}

TEST(UtilTest, JoinIntTest) {
  std::vector<int> tokens;
  tokens.push_back(10);
  tokens.push_back(2);
  tokens.push_back(-4);
  tokens.push_back(5);
  EXPECT_EQ(string_util::Join(tokens, " "), "10 2 -4 5");
  EXPECT_EQ(string_util::Join(tokens, ":"), "10:2:-4:5");
  EXPECT_EQ(string_util::Join(tokens, ""), "102-45");
}

TEST(UtilTest, StringPieceTest) {
  StringPiece s;
  EXPECT_EQ(0, s.find("", 0));
}

TEST(UtilTest, StringReplaceTest) {
  EXPECT_EQ("fbb", string_util::StringReplace("foo", "o", "b", true));
  EXPECT_EQ("fbo", string_util::StringReplace("foo", "o", "b", false));
  EXPECT_EQ("abcDEf", string_util::StringReplace("abcdef", "de", "DE", true));
  EXPECT_EQ("abcf", string_util::StringReplace("abcdef", "de", "", true));
  EXPECT_EQ("aBCaBC", string_util::StringReplace("abcabc", "bc", "BC", true));
  EXPECT_EQ("aBCabc", string_util::StringReplace("abcabc", "bc", "BC", false));
  EXPECT_EQ("", string_util::StringReplace("", "bc", "BC", false));
  EXPECT_EQ("", string_util::StringReplace("", "bc", "", false));
  EXPECT_EQ("", string_util::StringReplace("", "", "", false));
  EXPECT_EQ("abc", string_util::StringReplace("abc", "", "b", false));
}

TEST(UtilTest, EncodePODTet) {
  std::string tmp;
  {
    float v = 0.0;
    tmp = string_util::EncodePOD<float>(10.0);
    EXPECT_TRUE(string_util::DecodePOD<float>(tmp, &v));
    EXPECT_EQ(10.0, v);
  }

  {
    double v = 0.0;
    tmp = string_util::EncodePOD<double>(10.0);
    EXPECT_TRUE(string_util::DecodePOD<double>(tmp, &v));
    EXPECT_EQ(10.0, v);
  }

  {
    int32 v = 0;
    tmp = string_util::EncodePOD<int32>(10);
    EXPECT_TRUE(string_util::DecodePOD<int32>(tmp, &v));
    EXPECT_EQ(10, v);
  }

  {
    int16 v = 0;
    tmp = string_util::EncodePOD<int16>(10);
    EXPECT_TRUE(string_util::DecodePOD<int16>(tmp, &v));
    EXPECT_EQ(10, v);
  }

  {
    int64 v = 0;
    tmp = string_util::EncodePOD<int64>(10);
    EXPECT_TRUE(string_util::DecodePOD<int64>(tmp, &v));
    EXPECT_EQ(10, v);
  }

  // Invalid data
  {
    int32 v = 0;
    tmp = string_util::EncodePOD<int64>(10);
    EXPECT_FALSE(string_util::DecodePOD<int32>(tmp, &v));
  }
}

TEST(UtilTest, ItoaTest) {
  auto Itoa = [](int v) {
    char buf[16];
    string_util::Itoa(v, buf);
    return std::string(buf);
  };

  EXPECT_EQ("0", Itoa(0));
  EXPECT_EQ("10", Itoa(10));
  EXPECT_EQ("-10", Itoa(-10));
  EXPECT_EQ("718", Itoa(718));
  EXPECT_EQ("-522", Itoa(-522));
}

TEST(UtilTest, OneCharLenTest) {
  EXPECT_EQ(1, string_util::OneCharLen("abc"));
  EXPECT_EQ(3, string_util::OneCharLen("テスト"));
}

TEST(UtilTest, DecodeUTF8Test) {
  size_t mblen = 0;

  {
    const std::string input = "";
    EXPECT_EQ(0, string_util::DecodeUTF8(input.data(),
                                         input.data() + input.size(), &mblen));
    EXPECT_EQ(1, mblen);  // mblen always returns >= 1
  }

  {
    const std::string input = "\x01";
    EXPECT_EQ(1, string_util::DecodeUTF8(input.data(),
                                         input.data() + input.size(), &mblen));
    EXPECT_EQ(1, mblen);
  }

  {
    const std::string input = "\x7F";
    EXPECT_EQ(0x7F, string_util::DecodeUTF8(
                        input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(1, mblen);
  }

  {
    const std::string input = "\xC2\x80 ";
    EXPECT_EQ(0x80, string_util::DecodeUTF8(
                        input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(2, mblen);
  }

  {
    const std::string input = "\xDF\xBF ";
    EXPECT_EQ(0x7FF, string_util::DecodeUTF8(
                         input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(2, mblen);
  }

  {
    const std::string input = "\xE0\xA0\x80 ";
    EXPECT_EQ(0x800, string_util::DecodeUTF8(
                         input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(3, mblen);
  }

  {
    const std::string input = "\xF0\x90\x80\x80 ";
    EXPECT_EQ(0x10000, string_util::DecodeUTF8(
                           input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(4, mblen);
  }

  {
    const std::string input = "\xF7\xBF\xBF\xBF ";
    EXPECT_EQ(0x1FFFFF, string_util::DecodeUTF8(
                            input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(4, mblen);
  }

  {
    const std::string input = "\xF8\x88\x80\x80\x80 ";
    EXPECT_EQ(0x200000, string_util::DecodeUTF8(
                            input.data(), input.data() + input.size(), &mblen));
    EXPECT_EQ(5, mblen);
  }

  {
    const std::string input = "\xFC\x84\x80\x80\x80\x80 ";
    EXPECT_EQ(0x4000000,
              string_util::DecodeUTF8(input.data(), input.data() + input.size(),
                                      &mblen));
    EXPECT_EQ(6, mblen);
  }

  {
    const char *kInvalidData[] = {
        "\xC2",      // must be 2byte.
        "\xE0\xE0",  // must be 3byte.
        "\xFF",      // BOM
        "\xFE"       // BOM
    };

    for (size_t i = 0; i < 4; ++i) {
      // return values of string_util::DecodeUTF8 is not defined.
      // TODO(taku) implement an workaround.
      string_util::DecodeUTF8(
          kInvalidData[i], kInvalidData[i] + strlen(kInvalidData[i]), &mblen);
      EXPECT_EQ(1, mblen);
    }
  }
}

TEST(UtilTest, EncodeUTF8Test) {
  constexpr int kMaxUnicode = 0x110000;
  char buf[16];
  for (char32 cp = 1; cp <= kMaxUnicode; ++cp) {
    const size_t mblen = string_util::EncodeUTF8(cp, buf);
    size_t mblen2;
    char32 c = string_util::DecodeUTF8(buf, buf + 16, &mblen2);
    EXPECT_EQ(mblen2, mblen);
    EXPECT_EQ(cp, c);
  }

  EXPECT_EQ(0, string_util::EncodeUTF8(0, buf));
  EXPECT_EQ('\0', buf[0]);

  // non UCS4
  size_t mblen;
  EXPECT_EQ(5, string_util::EncodeUTF8(0x7000000, buf));
  string_util::DecodeUTF8(buf, buf + 16, &mblen);
  EXPECT_EQ(5, mblen);

  EXPECT_EQ(6, string_util::EncodeUTF8(0x8000001, buf));
  string_util::DecodeUTF8(buf, buf + 16, &mblen);
  EXPECT_EQ(6, mblen);
}

TEST(UtilTest, UnicodeCharToUTF8Test) {
  constexpr int kMaxUnicode = 0x110000;
  for (char32 cp = 1; cp <= kMaxUnicode; ++cp) {
    const auto s = string_util::UnicodeCharToUTF8(cp);
    const auto ut = string_util::UTF8ToUnicodeText(s);
    EXPECT_EQ(1, ut.size());
    EXPECT_EQ(cp, ut[0]);
  }
}

TEST(UtilTest, UnicodeTextToUTF8Test) {
  string_util::UnicodeText ut;

  ut = string_util::UTF8ToUnicodeText("test");
  EXPECT_EQ("test", string_util::UnicodeTextToUTF8(ut));

  ut = string_util::UTF8ToUnicodeText("テスト");
  EXPECT_EQ("テスト", string_util::UnicodeTextToUTF8(ut));

  ut = string_util::UTF8ToUnicodeText("これはtest");
  EXPECT_EQ("これはtest", string_util::UnicodeTextToUTF8(ut));
}

TEST(UtilTest, MapUtilTest) {
  const std::map<std::string, std::string> kMap = {
      {"a", "A"}, {"b", "B"}, {"c", "C"}};

  EXPECT_TRUE(port::ContainsKey(kMap, "a"));
  EXPECT_TRUE(port::ContainsKey(kMap, "b"));
  EXPECT_FALSE(port::ContainsKey(kMap, ""));
  EXPECT_FALSE(port::ContainsKey(kMap, "x"));

  EXPECT_EQ("A", port::FindOrDie(kMap, "a"));
  EXPECT_EQ("B", port::FindOrDie(kMap, "b"));
  EXPECT_DEATH(port::FindOrDie(kMap, "x"));

  EXPECT_EQ("A", port::FindWithDefault(kMap, "a", "x"));
  EXPECT_EQ("B", port::FindWithDefault(kMap, "b", "x"));
  EXPECT_EQ("x", port::FindWithDefault(kMap, "d", "x"));

  EXPECT_EQ("A", port::FindOrDie(kMap, "a"));
  EXPECT_DEATH(port::FindOrDie(kMap, "d"));
}

TEST(UtilTest, MapUtilVecTest) {
  const std::map<std::vector<int>, std::string> kMap = {{{0, 1}, "A"}};
  EXPECT_DEATH(port::FindOrDie(kMap, {0, 2}));
}

TEST(UtilTest, InputOutputBufferTest) {
  test::ScopedTempFile sf("test_file");

  const char *kData[] = {
      "This"
      "is"
      "a"
      "test"};

  {
    io::OutputBuffer output(sf.filename());
    for (size_t i = 0; i < arraysize(kData); ++i) {
      output.WriteLine(kData[i]);
    }
  }

  {
    io::InputBuffer input(sf.filename());
    std::string line;
    for (size_t i = 0; i < arraysize(kData); ++i) {
      EXPECT_TRUE(input.ReadLine(&line));
      EXPECT_EQ(kData[i], line);
    }
    EXPECT_FALSE(input.ReadLine(&line));
  }
}

TEST(UtilTest, InputOutputBufferInvalidFileTest) {
  EXPECT_DEATH(io::InputBuffer input("__UNKNOWN__FILE__"));
}

TEST(UtilTest, STLDeleteELementsTest) {
  class Item {
   public:
    explicit Item(int *counter) : counter_(counter) {}
    ~Item() { ++*counter_; }

   private:
    int *counter_;
  };

  std::vector<Item *> data;
  int counter = 0;
  for (int i = 0; i < 10; ++i) {
    data.push_back(new Item(&counter));
  }
  port::STLDeleteElements(&data);
  CHECK_EQ(10, counter);
  EXPECT_EQ(0, data.size());
}

TEST(UtilTest, StatusTest) {
  const util::Status ok;
  EXPECT_TRUE(ok.ok());
  EXPECT_EQ(util::error::OK, ok.code());
  EXPECT_EQ(std::string(""), ok.error_message());

  const util::Status s1(util::error::UNKNOWN, "unknown");
  const util::Status s2(util::error::UNKNOWN, std::string("unknown"));

  EXPECT_EQ(util::error::UNKNOWN, s1.code());
  EXPECT_EQ(util::error::UNKNOWN, s2.code());
  EXPECT_EQ(std::string("unknown"), s1.error_message());
  EXPECT_EQ(std::string("unknown"), s2.error_message());

  auto ok2 = util::OkStatus();
  EXPECT_TRUE(ok2.ok());
  EXPECT_EQ(util::error::OK, ok2.code());
  EXPECT_EQ(std::string(""), ok2.error_message());

  util::OkStatus().IgnoreError();
  for (int i = 0; i <= 16; ++i) {
    util::Status s(static_cast<util::error::Code>(i), "message");
    EXPECT_TRUE(s.ToString().find("message") != std::string::npos);
  }
}
}  // namespace sentencepiece
