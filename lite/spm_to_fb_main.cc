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

// Standalone offline model conversion tool for SentencePiece Lite Runtime.

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"

ABSL_FLAG(std::string, model, "",
          "Path to the input SentencePiece Protobuf model (*.model)");
ABSL_FLAG(std::string, output, "",
          "Path to the output FlatBuffers model (*.spm.fb)");
ABSL_FLAG(bool, treat_null_byte_as_unused, false,
          "If true, automatically converts pieces containing null bytes "
          "('\\0') into UNUSED pieces (requires byte_fallback = true).");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model);
  const std::string output_path = absl::GetFlag(FLAGS_output);

  if (model_path.empty() || output_path.empty()) {
    std::cerr << "Both --model and --output flags are required.\n";
    return 1;
  }

  std::ifstream input(model_path, std::ios::binary);
  if (!input) {
    std::cerr << "Failed to open input model file: " << model_path << "\n";
    return 1;
  }

  sentencepiece::ModelProto proto;
  if (!proto.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse Protobuf model.\n";
    return 1;
  }

  sentencepiece::lite::ConverterOptions options;
  options.treat_null_byte_as_unused =
      absl::GetFlag(FLAGS_treat_null_byte_as_unused);

  auto fbs_bytes_or = sentencepiece::lite::ToFlatbuffer(proto, options);
  if (!fbs_bytes_or.ok()) {
    std::cerr << "Conversion failed: " << fbs_bytes_or.status().message()
              << "\n";
    return 1;
  }

  // Verify the converted FlatBuffers model by loading it into the Lite
  // processor.
  sentencepiece::lite::SentencePieceLiteProcessor processor(*fbs_bytes_or);
  if (processor.status() != sentencepiece::lite::StatusCode::kOk) {
    std::cerr << "Model verification failed: Converted FlatBuffers model is "
                 "invalid or corrupted.\n";
    return 1;
  }

  std::ofstream output_file(output_path, std::ios::binary);
  if (!output_file) {
    std::cerr << "Failed to open output file: " << output_path << "\n";
    return 1;
  }

  output_file.write(fbs_bytes_or->data(), fbs_bytes_or->size());

  std::cout << "Successfully converted " << model_path << " to " << output_path
            << "\n";
  return 0;
}
