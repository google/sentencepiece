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

#include <sstream>
#include "common.h"
#include "flags.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

DEFINE_string(output, "", "Output filename");
DEFINE_string(model, "", "input model file name");
DEFINE_string(output_format, "txt", "output format. choose from txt or proto");

int main(int argc, char *argv[]) {
  sentencepiece::flags::ParseCommandLineFlags(argc, argv);
  sentencepiece::SentencePieceProcessor sp;
  CHECK_OK(sp.Load(FLAGS_model));

  sentencepiece::io::OutputBuffer output(FLAGS_output);
  CHECK_OK(output.status());

  if (FLAGS_output_format == "txt") {
    for (const auto &piece : sp.model_proto().pieces()) {
      std::ostringstream os;
      os << piece.piece() << "\t" << piece.score();
      output.WriteLine(os.str());
    }
  } else if (FLAGS_output_format == "proto") {
    output.Write(sp.model_proto().Utf8DebugString());
  }

  return 0;
}
