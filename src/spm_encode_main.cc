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

#include "common.h"
#include "flags.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

DEFINE_string(model, "", "model file name");
DEFINE_string(output_format, "piece", "choose from piece, id, or proto");
DEFINE_string(output, "", "output filename");
DEFINE_string(extra_options, "",
              "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");

int main(int argc, char *argv[]) {
  std::vector<std::string> rest_args;
  sentencepiece::flags::ParseCommandLineFlags(argc, argv, &rest_args);

  CHECK_OR_HELP(model);

  sentencepiece::SentencePieceProcessor sp;
  sp.LoadOrDie(FLAGS_model);
  sp.SetEncodeExtraOptions(FLAGS_extra_options);

  sentencepiece::io::OutputBuffer output(FLAGS_output);

  if (rest_args.empty()) {
    rest_args.push_back("");  // empty means that reading from stdin.
  }

  std::string line;
  std::vector<std::string> sps;
  std::vector<int> ids;
  sentencepiece::SentencePieceText spt;
  std::function<void(const std::string &line)> process;

  if (FLAGS_output_format == "piece") {
    process = [&](const std::string &line) {
      sp.Encode(line, &sps);
      output.WriteLine(sentencepiece::string_util::Join(sps, " "));
    };
  } else if (FLAGS_output_format == "id") {
    process = [&](const std::string &line) {
      sp.Encode(line, &ids);
      output.WriteLine(sentencepiece::string_util::Join(ids, " "));
    };
  } else if (FLAGS_output_format == "proto") {
    process = [&](const std::string &line) {
      sp.Encode(line, &spt);
      output.WriteLine(spt.Utf8DebugString());
    };
  } else {
    LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
  }

  for (const auto &filename : rest_args) {
    sentencepiece::io::InputBuffer input(filename);
    while (input.ReadLine(&line)) {
      if (line.empty()) {
	output.WriteLine("");
        continue;
      }
      process(line);
    }
  }

  return 0;
}
