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
#include "filesystem.h"
#include "normalizer.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
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

  EXPECT_TRUE(Builder::RemoveRedundantMap(&chars_map).ok());
  EXPECT_EQ(3, chars_map.size());
  EXPECT_EQ(chars_map.end(), chars_map.find({0x0061, 0x0062}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0061}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0062}));
  EXPECT_NE(chars_map.end(), chars_map.find({0x0061, 0x0062, 0x0063}));
}

TEST(BuilderTest, GetPrecompiledCharsMapWithInvalidNameTest) {
  std::string output;
  EXPECT_FALSE(Builder::GetPrecompiledCharsMap("", &output).ok());
  EXPECT_FALSE(Builder::GetPrecompiledCharsMap("__UNKNOWN__", &output).ok());
}

TEST(BuilderTest, BuildNFKCMapTest) {
  Builder::CharsMap chars_map;
#ifdef ENABLE_NFKC_COMPILE
  EXPECT_TRUE(Builder::BuildNFKCMap(&chars_map).ok());
  EXPECT_TRUE(!chars_map.empty());
#else
  EXPECT_TRUE(Builder::BuildNFKCMap(&chars_map).ok());
#endif
}

TEST(BuilderTest, GetPrecompiledCharsMapTest) {
  SetDataDir(::testing::SrcDir());

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
  for (char32_t lc = static_cast<char32_t>('a');
       lc <= static_cast<char32_t>('z'); ++lc) {
    const char32_t uc = lc + 'A' - 'a';
    chars_map[{lc}] = {uc};
  }

  // あいう => abc
  chars_map[{0x3042, 0x3044, 0x3046}] = {0x0061, 0x0062, 0x0063};

  // えお => remove
  chars_map[{0x3048, 0x304A}] = {};

  NormalizerSpec spec;
  EXPECT_TRUE(
      Builder::CompileCharsMap(chars_map, spec.mutable_precompiled_charsmap())
          .ok());
  Builder::CharsMap decompiled_chars_map;
  EXPECT_TRUE(Builder::DecompileCharsMap(spec.precompiled_charsmap(),
                                         &decompiled_chars_map)
                  .ok());
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

TEST(BuilderTest, DecompileMalformedCharsMapTest) {
  // Assembles a precompiled charsmap from raw darts units and a normalized
  // block, matching the on-disk <size><trie><normalized> layout.
  auto make_blob = [](const std::vector<uint32_t>& units,
                      absl::string_view normalized) {
    std::string trie_blob;
    for (const uint32_t u : units)
      trie_blob += string_util::EncodePOD<uint32_t>(u);
    std::string blob = string_util::EncodePOD<uint32_t>(
        static_cast<uint32_t>(trie_blob.size()));
    blob += trie_blob;
    blob.append(normalized.data(), normalized.size());
    return blob;
  };

  // A leaf stores an offset into the normalized block. Point it one past the
  // block: the value must be rejected, not dereferenced.
  {
    std::vector<uint32_t> units(256, 0);
    units[0] = 0x05;                       // root: label 5, no leaf.
    units[1] = 0x01 | 0x100 | (3u << 10);  // byte 1: label 1, leaf, offset 3.
    const std::string normalized("abc\0", 4);
    units[2] = 0x80000000u | static_cast<uint32_t>(normalized.size());
    Builder::CharsMap chars_map;
    EXPECT_FALSE(
        Builder::DecompileCharsMap(make_blob(units, normalized), &chars_map)
            .ok());
  }

  // A node whose child base lies outside the array must be rejected before it
  // is walked.
  {
    std::vector<uint32_t> units(256, 0);
    units[5] = 0x3FFu << 10;  // non-leaf unit with an out-of-range offset.
    Builder::CharsMap chars_map;
    EXPECT_FALSE(Builder::DecompileCharsMap(
                     make_blob(units, std::string("x\0", 2)), &chars_map)
                     .ok());
  }
}

static constexpr char kTestInputData[] = "nfkc.tsv";

TEST(BuilderTest, LoadCharsMapTest) {
  Builder::CharsMap chars_map;
  ASSERT_TRUE(
      Builder::LoadCharsMap(util::JoinPath(::testing::SrcDir(), kTestInputData),
                            &chars_map)
          .ok());

  std::string precompiled, expected;
  ASSERT_TRUE(Builder::CompileCharsMap(chars_map, &precompiled).ok());

  // Round-trip.
  Builder::CharsMap decompiled_chars_map;
  ASSERT_TRUE(
      Builder::DecompileCharsMap(precompiled, &decompiled_chars_map).ok());
  EXPECT_EQ(chars_map, decompiled_chars_map);

  ASSERT_TRUE(Builder::SaveCharsMap(
                  util::JoinPath(::testing::TempDir(), "output.tsv"), chars_map)
                  .ok());

  Builder::CharsMap saved_chars_map;
  ASSERT_TRUE(
      Builder::LoadCharsMap(util::JoinPath(::testing::TempDir(), "output.tsv"),
                            &saved_chars_map)
          .ok());
  EXPECT_EQ(chars_map, saved_chars_map);

#ifdef ENABLE_NFKC_COMPILE
  Builder::CharsMap nfkc_map;
  ASSERT_TRUE(Builder::BuildNFKCMap(&nfkc_map).ok());
  ASSERT_TRUE(Builder::CompileCharsMap(nfkc_map, &expected).ok());
#endif
}

TEST(BuilderTest, LoadCharsMapWithEmptyeTest) {
  {
    auto output = filesystem::NewWritableFile(
        util::JoinPath(::testing::TempDir(), "test.tsv"));
    output->WriteLine("0061\t0041");
    output->WriteLine("0062");
    output->WriteLine("0063\t\t#foo=>bar");
  }

  Builder::CharsMap chars_map;
  EXPECT_TRUE(Builder::LoadCharsMap(
                  util::JoinPath(::testing::TempDir(), "test.tsv"), &chars_map)
                  .ok());

  EXPECT_EQ(3, chars_map.size());
  EXPECT_EQ(std::vector<char32_t>({0x0041}), chars_map[{0x0061}]);
  EXPECT_EQ(std::vector<char32_t>({}), chars_map[{0x0062}]);
  EXPECT_EQ(std::vector<char32_t>({}), chars_map[{0x0063}]);

  EXPECT_TRUE(
      Builder::SaveCharsMap(
          util::JoinPath(::testing::TempDir(), "test_out.tsv"), chars_map)
          .ok());

  Builder::CharsMap new_chars_map;
  EXPECT_TRUE(
      Builder::LoadCharsMap(
          util::JoinPath(::testing::TempDir(), "test_out.tsv"), &new_chars_map)
          .ok());
  EXPECT_EQ(chars_map, new_chars_map);
}

TEST(BuilderTest, ContainsTooManySharedPrefixTest) {
  Builder::CharsMap chars_map;
  std::vector<char32_t> keys;
  // chars_map contains too many shared prefix ("aaaa...");
  for (int i = 0; i < 100; ++i) {
    keys.push_back('a');
    chars_map[keys] = {'b'};
  }
  std::string output;
  EXPECT_FALSE(Builder::CompileCharsMap(chars_map, &output).ok());
}

TEST(BuilderTest, DecompileDeepCharsMapTest) {
  Builder::CharsMap chars_map;
  std::vector<char32_t> key(1001, 'a');
  chars_map[key] = {'b'};

  std::string blob;
  ASSERT_TRUE(Builder::CompileCharsMap(chars_map, &blob).ok());

  Builder::CharsMap decompiled_map;
  absl::Status status = Builder::DecompileCharsMap(blob, &decompiled_map);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "Max recursion depth exceeded in decompile.");
}

}  // namespace normalizer
}  // namespace sentencepiece
