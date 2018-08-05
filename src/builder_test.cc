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
#include "flags.h"
#include "normalizer.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "util.h"

DECLARE_string(data_dir);

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

  EXPECT_OK(Builder::RemoveRedundantMap(&chars_map));
  EXPECT_EQ(3, chars_map.size());
  EXPECT_EQ(chars_map.end(), chars_map.find({0x0061, 0x0062}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0061}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0062}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0061, 0x0062, 0x0063}));
}

TEST(BuilderTest, GetPrecompiledCharsMapWithInvalidNameTest) {
  std::string output;
  EXPECT_NOT_OK(Builder::GetPrecompiledCharsMap("", &output));
  EXPECT_NOT_OK(Builder::GetPrecompiledCharsMap("__UNKNOWN__", &output));
}

TEST(BuilderTest, BuildNFKCMapTest) {
  Builder::CharsMap chars_map;
#ifdef ENABLE_NFKC_COMPILE
  EXPECT_OK(Builder::BuildNFKCMap(&chars_map));
  EXPECT_TRUE(!chars_map.empty());
#else
  EXPECT_OK(Builder::BuildNFKCMap(&chars_map));
#endif
}

TEST(BuilderTest, GetPrecompiledCharsMapTest) {
  {
    const NormalizerSpec spec =
        SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc");
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "ABC", normalizer.Normalize("ＡＢＣ"));
    EXPECT_EQ(WS "(株)", normalizer.Normalize("㈱"));
    EXPECT_EQ(WS "グーグル", normalizer.Normalize("ｸﾞｰｸﾞﾙ"));
  }

  {
    const NormalizerSpec spec =
        SentencePieceTrainer::GetNormalizerSpec("nfkc_cf");
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "abc", normalizer.Normalize("ＡＢＣ"));
    EXPECT_EQ(WS "abc", normalizer.Normalize("ABC"));
  }

  {
    const NormalizerSpec spec =
        SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc_cf");
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "abc", normalizer.Normalize("ＡＢＣ"));
    EXPECT_EQ(WS "abc", normalizer.Normalize("ABC"));
  }

  {
    const NormalizerSpec spec =
        SentencePieceTrainer::GetNormalizerSpec("identity");
    EXPECT_TRUE(spec.precompiled_charsmap().empty());
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

  // えお => remove
  chars_map[{0x3048, 0x304A}] = {};

  NormalizerSpec spec;
  EXPECT_OK(
      Builder::CompileCharsMap(chars_map, spec.mutable_precompiled_charsmap()));
  Builder::CharsMap decompiled_chars_map;
  EXPECT_OK(Builder::DecompileCharsMap(spec.precompiled_charsmap(),
                                       &decompiled_chars_map));
  EXPECT_EQ(chars_map, decompiled_chars_map);

  spec.set_add_dummy_prefix(false);
  const Normalizer normalizer(spec);

  EXPECT_EQ("ABC", normalizer.Normalize("abc"));
  EXPECT_EQ("ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ("XY" WS "Z", normalizer.Normalize("xy z"));

  EXPECT_EQ("あ", normalizer.Normalize("あ"));
  EXPECT_EQ("abc", normalizer.Normalize("あいう"));
  EXPECT_EQ("abcえ", normalizer.Normalize("あいうえ"));
  EXPECT_EQ("ABCabcD", normalizer.Normalize("abcあいうd"));
  EXPECT_EQ("abcか", normalizer.Normalize("あいうえおか"));
}

TEST(BuilderTest, LoadCharsMapTest) {
  Builder::CharsMap chars_map;
  EXPECT_OK(Builder::LoadCharsMap(util::JoinPath(FLAGS_data_dir, "nfkc.tsv"),
                                  &chars_map));

  std::string precompiled, expected;
  EXPECT_OK(Builder::CompileCharsMap(chars_map, &precompiled));

  // Round-trip.
  Builder::CharsMap decompiled_chars_map;
  EXPECT_OK(Builder::DecompileCharsMap(precompiled, &decompiled_chars_map));
  EXPECT_EQ(chars_map, decompiled_chars_map);

  test::ScopedTempFile output_tsv("output.tsv");
  EXPECT_OK(Builder::SaveCharsMap(output_tsv.filename(), chars_map));

  Builder::CharsMap saved_chars_map;
  EXPECT_OK(Builder::LoadCharsMap(output_tsv.filename(), &saved_chars_map));
  EXPECT_EQ(chars_map, saved_chars_map);

#ifdef ENABLE_NFKC_COMPILE
  Builder::CharsMap nfkc_map;
  EXPECT_OK(Builder::BuildNFKCMap(&nfkc_map));
  EXPECT_OK(Builder::CompileCharsMap(nfkc_map, &expected));
#endif
}

TEST(BuilderTest, LoadCharsMapWithEmptyeTest) {
  test::ScopedTempFile test_tsv("test.tsv");
  test::ScopedTempFile test_out_tsv("test_out.tsv");
  {
    io::OutputBuffer output(test_tsv.filename());
    output.WriteLine("0061\t0041");
    output.WriteLine("0062");
    output.WriteLine("0063\t\t#foo=>bar");
  }

  Builder::CharsMap chars_map;
  EXPECT_OK(Builder::LoadCharsMap(test_tsv.filename(), &chars_map));

  EXPECT_EQ(3, chars_map.size());
  EXPECT_EQ(std::vector<char32>({0x0041}), chars_map[{0x0061}]);
  EXPECT_EQ(std::vector<char32>({}), chars_map[{0x0062}]);
  EXPECT_EQ(std::vector<char32>({}), chars_map[{0x0063}]);

  EXPECT_OK(Builder::SaveCharsMap(test_out_tsv.filename(), chars_map));

  Builder::CharsMap new_chars_map;
  EXPECT_OK(Builder::LoadCharsMap(test_out_tsv.filename(), &new_chars_map));
  EXPECT_EQ(chars_map, new_chars_map);
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
  EXPECT_FALSE(Builder::CompileCharsMap(chars_map, &output).ok());
}

}  // namespace normalizer
}  // namespace sentencepiece
