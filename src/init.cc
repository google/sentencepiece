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

#include "init.h"

#include "common.h"
#include "config.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_cat.h"
#include "util.h"

#ifdef _USE_EXTERNAL_PROTOBUF
#include "google/protobuf/message_lite.h"
#else
#include "third_party/protobuf-lite/google/protobuf/message_lite.h"
#endif

ABSL_FLAG(bool, quiet, false, "Suppress logging message.");

namespace sentencepiece {
void ParseCommandLineFlags(const char *usage, int *argc, char ***argv,
                           bool remove_arg) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::SetProgramUsageMessage(absl::StrCat(PACKAGE_STRING, " ", VERSION,
                                            "\n\n", "Usage: ", *argv[0],
                                            " [options] files"));

  absl::FlagsUsageConfig usage_config;
  usage_config.version_string = [&]() {
    return absl::StrCat(PACKAGE_STRING, " ", VERSION, "\n");
  };
  absl::SetFlagsUsageConfig(usage_config);

  const auto unused_args = absl::ParseCommandLine(*argc, *argv);

  if (remove_arg) {
    char **argv_val = *argv;
    *argv = argv_val = argv_val + *argc - unused_args.size();
    std::copy(unused_args.begin(), unused_args.end(), argv_val);
    *argc = static_cast<int>(unused_args.size());
  }

  if (absl::GetFlag(FLAGS_quiet)) {
    absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(100));
  }
}

void ShutdownLibrary() { google::protobuf::ShutdownProtobufLibrary(); }

}  // namespace sentencepiece
