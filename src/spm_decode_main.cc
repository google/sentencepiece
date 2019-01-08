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
#include "common.h"
#include "filesystem.h"
#include "flags.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

DEFINE_string(model, "", "model file name");
DEFINE_string(output, "", "output filename");
DEFINE_string(input_format, "piece", "choose from piece or id");
DEFINE_string(output_format, "string", "choose from string or proto");
DEFINE_string(extra_options, "",
              "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");

int main(int argc, char *argv[]) {
  std::vector<std::string> rest_args;
  sentencepiece::flags::ParseCommandLineFlags(argc, argv, &rest_args);

  CHECK_OR_HELP(model);

  sentencepiece::SentencePieceProcessor sp;
  CHECK_OK(sp.Load(FLAGS_model));
  CHECK_OK(sp.SetDecodeExtraOptions(FLAGS_extra_options));

  auto output = sentencepiece::filesystem::NewWritableFile(FLAGS_output);
  CHECK_OK(output->status());

  if (rest_args.empty()) {
    rest_args.push_back("");  // empty means that reading from stdin.
  }

  std::string detok, line;
  sentencepiece::SentencePieceText spt;
  std::function<void(const std::vector<std::string> &pieces)> process;

  auto ToIds = [&](const std::vector<std::string> &pieces) {
    std::vector<int> ids;
    for (const auto &s : pieces) {
      ids.push_back(atoi(s.c_str()));
    }
    return ids;
  };

  if (FLAGS_input_format == "piece") {
    if (FLAGS_output_format == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &detok));
        output->WriteLine(detok);
      };
    } else if (FLAGS_output_format == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &spt));
        //        output->WriteLine(spt.Utf8DebugString());
      };
    } else {
      LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
    }
  } else if (FLAGS_input_format == "id") {
    if (FLAGS_output_format == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &detok));
        output->WriteLine(detok);
      };
    } else if (FLAGS_output_format == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &spt));
        //        output->WriteLine(spt.Utf8DebugString());
      };
    } else {
      LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
    }
  } else {
    LOG(FATAL) << "Unknown input format: " << FLAGS_input_format;
  }

  for (const auto &filename : rest_args) {
    auto input = sentencepiece::filesystem::NewReadableFile(filename);
    CHECK_OK(input->status());
    while (input->ReadLine(&line)) {
      const auto pieces = sentencepiece::string_util::Split(line, " ");
      process(pieces);
    }
  }

  return 0;
}
