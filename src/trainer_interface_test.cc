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

#include "trainer_interface.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(TrainerInterfaceTest, IsValidSentencePieceTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;

  auto IsValid = [&trainer_spec, &normalizer_spec](const std::string &str) {
    TrainerInterface trainer(trainer_spec, normalizer_spec);
    const string_util::UnicodeText text = string_util::UTF8ToUnicodeText(str);
    return trainer.IsValidSentencePiece(text);
  };

  // Default trainer spec.
  EXPECT_FALSE(IsValid(""));
  EXPECT_FALSE(IsValid("12345678912345678"));  // too long
  EXPECT_TRUE(IsValid("a"));
  EXPECT_TRUE(IsValid(WS));
  EXPECT_TRUE(IsValid(WS "a"));
  EXPECT_FALSE(IsValid("a" WS));
  EXPECT_FALSE(IsValid(WS "a" WS));
  EXPECT_FALSE(IsValid("a" WS "b"));
  EXPECT_FALSE(IsValid("a" WS "b" WS));
  EXPECT_TRUE(IsValid("あいう"));
  EXPECT_TRUE(IsValid("グーグル"));  // "ー" is a part of Katakana
  EXPECT_TRUE(IsValid("食べる"));
  EXPECT_FALSE(IsValid("漢字ABC"));  // mixed CJK scripts
  EXPECT_FALSE(IsValid("F1"));
  EXPECT_TRUE(IsValid("$10"));  // $ and 1 are both "common" script.
  EXPECT_FALSE(IsValid("$ABC"));
  EXPECT_FALSE(IsValid("ab\tbc"));  // "\t" is UPP boundary.

  trainer_spec.set_split_by_whitespace(false);
  EXPECT_TRUE(IsValid(WS));
  EXPECT_TRUE(IsValid(WS "a"));
  EXPECT_FALSE(IsValid("a" WS));
  EXPECT_FALSE(IsValid(WS "a" WS));
  EXPECT_TRUE(IsValid("a" WS "b"));  // "a b" is a valid piece.
  EXPECT_TRUE(IsValid(WS "a" WS "b"));
  EXPECT_TRUE(IsValid(WS "a" WS "b" WS "c"));
  EXPECT_FALSE(IsValid("a" WS "b" WS));

  trainer_spec.set_split_by_unicode_script(false);
  EXPECT_TRUE(IsValid("あいう"));
  EXPECT_TRUE(IsValid("グーグル"));
  EXPECT_TRUE(IsValid("食べる"));
  EXPECT_TRUE(IsValid("漢字ABC"));
  EXPECT_TRUE(IsValid("F1"));
  EXPECT_TRUE(IsValid("$10"));
  EXPECT_TRUE(IsValid("$ABC"));

  trainer_spec.set_max_sentencepiece_length(4);
  EXPECT_TRUE(IsValid("1234"));
  EXPECT_FALSE(IsValid("12345"));
}
}  // namespace sentencepiece
