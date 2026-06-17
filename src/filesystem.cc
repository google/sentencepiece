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

#include "filesystem.h"

#include <fstream>
#include <iostream>
#include <memory>

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "util.h"

#if defined(OS_WIN) && defined(UNICODE) && defined(_UNICODE)
#define WPATH(path) (::sentencepiece::util::Utf8ToWide(path).c_str())
#else
#define WPATH(path) (path.data())
#endif

namespace sentencepiece {
namespace filesystem {

class PosixReadableFile : public ReadableFile {
 public:
  explicit PosixReadableFile(absl::string_view filename,
                             bool is_binary = false) {
    if (filename.empty()) {
      is_ = &std::cin;
    } else {
      file_.open(WPATH(filename),
                 is_binary ? std::ios::binary | std::ios::in : std::ios::in);
      is_ = &file_;
    }
    if (!*is_ || ((is_->peek() != 0) && is_->fail())) {
      status_ = absl::StatusBuilder(absl::StatusCode::kNotFound)
                << "\"" << filename.data() << "\": " << util::StrError(errno);
    }
  }

  ~PosixReadableFile() override = default;

  absl::Status status() const override { return status_; }

  bool ReadLine(std::string* line) override {
    return static_cast<bool>(std::getline(*is_, *line));
  }

  bool ReadAll(std::string* line) override {
    if (is_ == &std::cin) {
      LOG(ERROR) << "ReadAll is not supported for stdin.";
      return false;
    }
    line->assign(std::istreambuf_iterator<char>(*is_),
                 std::istreambuf_iterator<char>());
    return true;
  }

 private:
  absl::Status status_;
  std::ifstream file_;
  std::istream* is_;
};

class PosixWritableFile : public WritableFile {
 public:
  explicit PosixWritableFile(absl::string_view filename,
                             bool is_binary = false) {
    if (filename.empty()) {
      os_ = &std::cout;
    } else {
      file_.open(WPATH(filename),
                 is_binary ? std::ios::binary | std::ios::out : std::ios::out);
      os_ = &file_;
    }
    if (!*os_) {
      status_ = absl::StatusBuilder(absl::StatusCode::kPermissionDenied)
                << "\"" << filename.data() << "\": " << util::StrError(errno);
    }
  }

  ~PosixWritableFile() override = default;

  absl::Status status() const override { return status_; }

  bool Write(absl::string_view text) override {
    os_->write(text.data(), text.size());
    return os_->good();
  }

  bool WriteLine(absl::string_view text) override {
    return Write(text) && Write("\n");
  }

 private:
  absl::Status status_;
  std::ofstream file_;
  std::ostream* os_;
};

using DefaultReadableFile = PosixReadableFile;
using DefaultWritableFile = PosixWritableFile;

std::unique_ptr<ReadableFile> NewReadableFile(absl::string_view filename,
                                              bool is_binary) {
  return std::make_unique<DefaultReadableFile>(filename, is_binary);
}

std::unique_ptr<WritableFile> NewWritableFile(absl::string_view filename,
                                              bool is_binary) {
  return std::make_unique<DefaultWritableFile>(filename, is_binary);
}

}  // namespace filesystem
}  // namespace sentencepiece
