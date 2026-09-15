// Copyright 2026 Google LLC
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
// limitations under the License.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"

ABSL_FLAG(std::string, model, "",
          "Path to FlatBuffers (*.spm.fb) or original Protobuf (*.model) "
          "SentencePiece model.");
ABSL_FLAG(bool, encode, false, "Run in tokenization/encoding mode (default).");
ABSL_FLAG(bool, decode, false, "Run in detokenization/decoding mode.");
ABSL_FLAG(bool, normalize, false, "Run in text normalization mode.");
ABSL_FLAG(std::string, input, "",
          "Path to input file. If empty, reads from stdin.");
ABSL_FLAG(std::string, output, "",
          "Path to output file. If empty, writes to stdout.");

namespace {

using sentencepiece::lite::SentencePieceLiteProcessor;
using sentencepiece::lite::StatusCode;

std::string ReadFileToString(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "Failed to open file: " << path << "\n";
    std::exit(1);
  }
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    std::cerr << "The --model flag is required.\n";
    return 1;
  }

  const std::string input_path = absl::GetFlag(FLAGS_input);
  const std::string output_path = absl::GetFlag(FLAGS_output);

  int modes = 0;
  if (absl::GetFlag(FLAGS_encode)) ++modes;
  if (absl::GetFlag(FLAGS_decode)) ++modes;
  if (absl::GetFlag(FLAGS_normalize)) ++modes;

  if (modes > 1) {
    std::cerr << "Only one of --encode, --decode, or --normalize can be "
                 "specified.\n";
    return 1;
  }

  bool encode = absl::GetFlag(FLAGS_encode);
  bool decode = absl::GetFlag(FLAGS_decode);
  bool normalize = absl::GetFlag(FLAGS_normalize);

  if (modes == 0) {
    encode = true;  // Default mode is encode.
  }

  std::string model_bytes = ReadFileToString(model_path);
  std::unique_ptr<SentencePieceLiteProcessor> processor;

  // On-the-fly conversion for original SentencePiece model files (*.model)
  if (model_path.size() >= 6 &&
      model_path.compare(model_path.size() - 6, 6, ".model") == 0) {
    sentencepiece::ModelProto proto;
    if (!proto.ParseFromString(model_bytes)) {
      std::cerr << "Failed to parse original SentencePiece model from "
                << model_path << "\n";
      return 1;
    }
    auto fbs_bytes_or = sentencepiece::lite::ToFlatbuffer(proto);
    if (!fbs_bytes_or.ok()) {
      std::cerr << "Model conversion failed: "
                << fbs_bytes_or.status().message() << "\n";
      return 1;
    }
    processor = std::make_unique<SentencePieceLiteProcessor>(
        std::make_shared<std::string>(std::move(*fbs_bytes_or)));
  } else {
    processor = std::make_unique<SentencePieceLiteProcessor>(
        std::make_shared<std::string>(std::move(model_bytes)));
  }

  if (processor->status() != StatusCode::kOk) {
    std::cerr << "Failed to initialize SentencePieceLiteProcessor: "
              << static_cast<int>(processor->status()) << "\n";
    return 1;
  }

  std::istream* input_stream = &std::cin;
  std::ifstream input_file;
  if (!input_path.empty()) {
    input_file.open(input_path);
    if (!input_file) {
      std::cerr << "Failed to open input file: " << input_path << "\n";
      return 1;
    }
    input_stream = &input_file;
  }

  std::ostream* output_stream = &std::cout;
  std::ofstream output_file;
  if (!output_path.empty()) {
    output_file.open(output_path);
    if (!output_file) {
      std::cerr << "Failed to open output file: " << output_path << "\n";
      return 1;
    }
    output_stream = &output_file;
  }

  std::string line;
  while (std::getline(*input_stream, line)) {
    if (encode) {
      std::vector<int> ids;
      if (processor->Encode(line, &ids) == StatusCode::kOk) {
        *output_stream << absl::StrJoin(ids, " ") << "\n";
      } else {
        std::cerr << "Failed to encode line: " << line << "\n";
      }
    } else if (decode) {
      std::vector<std::string_view> id_strs =
          absl::StrSplit(line, ' ', absl::SkipEmpty());
      std::vector<int> ids;
      bool parse_ok = true;
      for (const auto& id_str : id_strs) {
        int id;
        if (absl::SimpleAtoi(id_str, &id)) {
          ids.push_back(id);
        } else {
          std::cerr << "Invalid token ID: " << id_str << "\n";
          parse_ok = false;
          break;
        }
      }
      if (parse_ok) {
        std::string output;
        if (processor->Decode(ids, &output) == StatusCode::kOk) {
          *output_stream << output << "\n";
        } else {
          std::cerr << "Failed to decode IDs: " << line << "\n";
        }
      }
    } else if (normalize) {
      std::string output;
      if (processor->Normalize(line, &output) == StatusCode::kOk) {
        *output_stream << output << "\n";
      } else {
        std::cerr << "Failed to normalize line: " << line << "\n";
      }
    }
  }

  return 0;
}
