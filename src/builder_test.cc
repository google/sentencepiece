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

#include "builder.h"
#include "common.h"
#include "normalizer.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(BuilderTest, RemoveRedundantMapTest) {
  Builder::CharsMap chars_map;

  // ab => AB, a => A, b => B, abc => BCA
  chars_map[{0x0061}] = {0x0041};
  chars_map[{0x0062}] = {0x0042};
  chars_map[{0x0061, 0x0062}] = {0x0041, 0x0042};
  chars_map[{0x0061, 0x0062, 0x0063}] = {0x0043, 0x0042, 0x0041};

  const auto new_chars_map = Builder::RemoveRedundantMap(chars_map);
  EXPECT_EQ(3, new_chars_map.size());
  EXPECT_EQ(new_chars_map.end(), new_chars_map.find({0x0061, 0x0062}));
  EXPECT_NE(new_chars_map.end(), new_chars_map.find({0x0061}));
  EXPECT_NE(new_chars_map.end(), new_chars_map.find({0x0062}));
  EXPECT_NE(new_chars_map.end(), new_chars_map.find({0x0061, 0x0062, 0x0063}));
}

TEST(BuilderTest, GetPrecompiledCharsMapWithInvalidNameTest) {
  std::string output;
  EXPECT_NOT_OK(Builder::GetPrecompiledCharsMap("", &output));
  EXPECT_NOT_OK(Builder::GetPrecompiledCharsMap("__UNKNOWN__", &output));
}

TEST(BuilderTest, BuildIdentityMapTest) {
  const auto m = Builder::BuildIdentityMap();
  EXPECT_EQ(1, m.size());
}

TEST(BuilderTest, BuildNFKCMapTest) {
#ifdef ENABLE_NFKC_COMPILE
  const auto m = Builder::BuildNFKCMap();
  EXPECT_TRUE(!m.empty());
#else
  EXPECT_DEATH(Builder::BuildNFKCMap());
#endif
}

TEST(BuilderTest, GetPrecompiledCharsMapTest) {
  {
    NormalizerSpec spec;
    spec.set_name("nfkc");
    EXPECT_OK(Builder::PopulateNormalizationSpec(&spec));
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "ABC", normalizer.Normalize("ＡＢＣ"));
    EXPECT_EQ(WS "(株)", normalizer.Normalize("㈱"));
    EXPECT_EQ(WS "グーグル", normalizer.Normalize("ｸﾞｰｸﾞﾙ"));
  }

  {
    NormalizerSpec spec;
    spec.set_name("identity");
    EXPECT_OK(Builder::PopulateNormalizationSpec(&spec));
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "ＡＢＣ", normalizer.Normalize("ＡＢＣ"));
    EXPECT_EQ(WS "㈱", normalizer.Normalize("㈱"));
    EXPECT_EQ(WS "ｸﾞｰｸﾞﾙ", normalizer.Normalize("ｸﾞｰｸﾞﾙ"));
  }
}

TEST(BuilderTest, CompileCharsMap) {
  Builder::CharsMap chars_map;

  // Lowercase => Uppercase
  for (char32 lc = static_cast<char32>('a'); lc <= static_cast<char32>('z');
       ++lc) {
    const char32 uc = lc + 'A' - 'a';
    chars_map[{lc}] = {uc};
  }

  // あいう => abc
  chars_map[{0x3042, 0x3044, 0x3046}] = {0x0061, 0x0062, 0x0063};

  NormalizerSpec spec;
  EXPECT_OK(
      Builder::CompileCharsMap(chars_map, spec.mutable_precompiled_charsmap()));
  spec.set_add_dummy_prefix(false);
  const Normalizer normalizer(spec);

  EXPECT_EQ("ABC", normalizer.Normalize("abc"));
  EXPECT_EQ("ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ("XY" WS "Z", normalizer.Normalize("xy z"));

  EXPECT_EQ("あ", normalizer.Normalize("あ"));
  EXPECT_EQ("abc", normalizer.Normalize("あいう"));
  EXPECT_EQ("abcえ", normalizer.Normalize("あいうえ"));
  EXPECT_EQ("ABCabcD", normalizer.Normalize("abcあいうd"));
}

TEST(BuilderTest, BuildMapFromFileTest) {
  const auto cmap = Builder::BuildMapFromFile("../data/nfkc.tsv");
  std::string expected, precompiled;
  EXPECT_OK(Builder::CompileCharsMap(cmap, &precompiled));
  EXPECT_OK(Builder::GetPrecompiledCharsMap("nfkc", &expected));
  EXPECT_EQ(expected, precompiled);
}

TEST(BuilderTest, ContainsTooManySharedPrefixTest) {
  Builder::CharsMap chars_map;
  std::vector<char32> keys;
  // chars_map contains too many shared prefix ("aaaa...");
  for (int i = 0; i < 100; ++i) {
    keys.push_back('a');
    chars_map[keys] = {'b'};
  }
  std::string output;
  EXPECT_NOT_OK(Builder::CompileCharsMap(chars_map, &output));
}

}  // namespace normalizer
}  // namespace sentencepiece
