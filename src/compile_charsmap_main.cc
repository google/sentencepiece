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

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include "builder.h"
#include "flags.h"
#include "sentencepiece_processor.h"
#include "stringpiece.h"
#include "util.h"

using sentencepiece::normalizer::Builder;
using sentencepiece::util::Status;

DEFINE_bool(output_precompiled_header, false, "make normalization_rule.h file");

namespace sentencepiece {
namespace {

std::string ToHexData(StringPiece data) {
  const char *begin = data.data();
  const char *end = data.data() + data.size();
  constexpr char kHex[] = "0123456789ABCDEF";
  constexpr size_t kNumOfBytesOnOneLine = 20;

  size_t output_count = 0;
  std::stringstream os;
  while (begin < end) {
    const size_t bucket_size =
        std::min<size_t>(end - begin, kNumOfBytesOnOneLine -
                                          output_count % kNumOfBytesOnOneLine);
    if (output_count % kNumOfBytesOnOneLine == 0) {
      os << "\"";
    }
    for (size_t i = 0; i < bucket_size; ++i) {
      os << "\\x" << kHex[(*begin & 0xF0) >> 4] << kHex[(*begin & 0x0F) >> 0];
      ++begin;
    }
    output_count += bucket_size;
    if (output_count % kNumOfBytesOnOneLine == 0) {
      os << "\"\n";
    }
  }
  os << "\"\n";

  return os.str();
}
}  // namespace
}  // namespace sentencepiece

int main(int argc, char **argv) {
  sentencepiece::flags::ParseCommandLineFlags(argc, argv);

  const std::vector<
      std::pair<std::string, std::function<Status(Builder::CharsMap *)>>>
      kRuleList = {{"nfkc", Builder::BuildNFKCMap}};

  constexpr char kHeader[] =
      R"(#ifndef NORMALIZATION_RULE_H_
#define NORMALIZATION_RULE_H_
#include <cstdio>
namespace sentencepiece {
namespace {
struct BinaryBlob {
 const char *name;
 size_t size;
 const char *data;
};
constexpr BinaryBlob kNormalizationRules_blob[] = {)";

  constexpr char kFooter[] = R"(
}  // namespace
}  // namespace sentencepiece
#endif  // NORMALIZATION_RULE_H_)";

  std::stringstream os;
  os << kHeader;

  for (const auto &p : kRuleList) {
    Builder::CharsMap normalized_map;
    CHECK_OK(p.second(&normalized_map));

    // Write Header.
    std::string index;
    CHECK_OK(Builder::CompileCharsMap(normalized_map, &index));
    os << "{ \"" << p.first << "\", " << index.size() << ",\n";
    os << sentencepiece::ToHexData(index);
    os << " },";

    // Write TSV file.
    CHECK_OK(Builder::SaveCharsMap(p.first + ".tsv", normalized_map));
  }

  os << "};\n";
  os << "constexpr size_t kNormalizationRules_size = " << kRuleList.size()
     << ";\n";
  os << kFooter;

  if (FLAGS_output_precompiled_header) {
    constexpr char kPrecompiledHeaderFileName[] = "normalization_rule.h";
    sentencepiece::io::OutputBuffer output(kPrecompiledHeaderFileName);
    CHECK_OK(output.status());
    output.Write(os.str());
  }

  return 0;
}
