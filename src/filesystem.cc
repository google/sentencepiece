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

#include <iostream>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "filesystem.h"
#include "third_party/absl/memory/memory.h"
#include "util.h"

#if defined(OS_WIN) && defined(UNICODE) && defined(_UNICODE)
#define WPATH(path) (::sentencepiece::win32::Utf8ToWide(path).c_str())
#else
#define WPATH(path) (path)
#endif

namespace sentencepiece {
namespace filesystem {

class PosixReadableFile : public ReadableFile {
 public:
  PosixReadableFile(absl::string_view filename, bool is_binary = false, char delim = '\n')
      : delim_(delim), mem_(nullptr) {
    if (!filename.empty()) {
      int fd = open(filename.data(), O_RDONLY);
      if (fd < 0) {
        SetErrorStatus(util::StatusCode::kNotFound, filename);
        return;
      }
      struct stat st;
      if (fstat(fd, &st) < 0) {
        SetErrorStatus(util::StatusCode::kInternal, filename);
        return;
      }
      file_size_ = st.st_size;
      mem_ = reinterpret_cast<char *>(mmap(
          NULL, file_size_, PROT_READ, MAP_SHARED, fd, 0));
      if (mem_ == MAP_FAILED) {
        SetErrorStatus(util::StatusCode::kInternal, filename);
        close(fd);
        return;
      }
      close(fd);
      head_ = mem_;
    }
  }

  ~PosixReadableFile() {
    if (mem_ != nullptr) {
      munmap(mem_, file_size_);
    }
  }

  util::Status status() const { return status_; }
  
  bool ReadLine(absl::string_view *line) {
    if (mem_ == nullptr) {
      return static_cast<bool>(std::getline(std::cin, lines_.emplace_back(), delim_));
    }
    size_t size_left = file_size_ - (head_ - mem_);
    if (size_left == 0) {
      return false;
    }
    auto ptr = reinterpret_cast<char *>(memchr(head_, delim_, size_left));
    if (ptr == nullptr) {
      *line = absl::string_view(head_, size_left);
      head_ = mem_ + file_size_;
    } else {
      *line = absl::string_view(head_, ptr - head_);
      head_ = ptr + 1;
    }
    return true;
  }

  bool ReadAll(absl::string_view *line) {
    if (mem_ == nullptr) {
      LOG(ERROR) << "ReadAll is not supported for stdin.";
      return false;
    }
    *line = absl::string_view(mem_, file_size_);
    return true;
  }

 private:
  char delim_;
  util::Status status_;
  char *mem_;
  char *head_;
  size_t file_size_;
  std::vector<std::string> lines_;

  void SetErrorStatus(util::StatusCode status, absl::string_view filename) {
    status_ = util::StatusBuilder(status, GTL_LOC)
              << "\"" << filename.data() << "\": " << util::StrError(errno);
  }
};

class PosixWritableFile : public WritableFile {
 public:
  PosixWritableFile(absl::string_view filename, bool is_binary = false)
      : os_(filename.empty()
                ? &std::cout
                : new std::ofstream(WPATH(filename.data()),
                                    is_binary ? std::ios::binary | std::ios::out
                                              : std::ios::out)) {
    if (!*os_)
      status_ =
          util::StatusBuilder(util::StatusCode::kPermissionDenied, GTL_LOC)
          << "\"" << filename.data() << "\": " << util::StrError(errno);
  }

  ~PosixWritableFile() {
    if (os_ != &std::cout) delete os_;
  }

  util::Status status() const { return status_; }

  bool Write(absl::string_view text) {
    os_->write(text.data(), text.size());
    return os_->good();
  }

  bool WriteLine(absl::string_view text) { return Write(text) && Write("\n"); }

 private:
  util::Status status_;
  std::ostream *os_;
};

using DefaultReadableFile = PosixReadableFile;
using DefaultWritableFile = PosixWritableFile;

std::unique_ptr<ReadableFile> NewReadableFile(absl::string_view filename,
                                              bool is_binary,
                                              char delim) {
  return absl::make_unique<DefaultReadableFile>(filename, is_binary, delim);
}

std::unique_ptr<WritableFile> NewWritableFile(absl::string_view filename,
                                              bool is_binary) {
  return absl::make_unique<DefaultWritableFile>(filename, is_binary);
}

}  // namespace filesystem
}  // namespace sentencepiece
