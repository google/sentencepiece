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

#ifndef FILESYSTEM_H_
#define FILESYSTEM_H_

#include <stdio.h>

#include <fstream>
#include <memory>
#include <string>
#include <variant>

#include "common.h"
#include "sentencepiece_processor.h"
#include "absl/strings/string_view.h"

namespace sentencepiece {
namespace filesystem {

using ps_string = std::variant<absl::string_view, std::shared_ptr<std::string>>;
class ReadableFile {
 public:
  ReadableFile() {}
  explicit ReadableFile(
      absl::string_view filename,
      bool is_binary = false,
      char delim = '\n') {}
  virtual ~ReadableFile() {}

  virtual util::Status status() const = 0;
  virtual bool ReadBuffer(std::string *buffer) = 0;
  virtual bool ReadLine(absl::string_view *line) = 0;
  // TODO: Fix ReadLine instead of adding ReadLineStdin.
  virtual bool ReadLineStdin(ps_string *line) = 0;
  virtual bool ReadAll(absl::string_view *line) = 0;
};

class WritableFile {
 public:
  WritableFile() {}
  explicit WritableFile(absl::string_view filename, bool is_binary = false) {}
  virtual ~WritableFile() {}

  virtual util::Status status() const = 0;
  virtual bool Write(absl::string_view text) = 0;
  virtual bool WriteLine(absl::string_view text) = 0;
};

std::unique_ptr<ReadableFile> NewReadableFile(absl::string_view filename,
                                              bool is_binary = false,
                                              char delim = '\n');
std::unique_ptr<WritableFile> NewWritableFile(absl::string_view filename,
                                              bool is_binary = false);

}  // namespace filesystem
}  // namespace sentencepiece
#endif  // FILESYSTEM_H_
