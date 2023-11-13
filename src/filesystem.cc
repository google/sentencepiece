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
#include "absl/memory/memory.h"
#include "util.h"

#if defined(OS_WIN) && defined(UNICODE) && defined(_UNICODE)
#define WPATH(path) (::sentencepiece::win32::Utf8ToWide(path).c_str())
#else
#define WPATH(path) (path)
#endif

#define STDIN_BLOCK_SIZE (1 << 20)

namespace sentencepiece {
namespace filesystem {

class PosixReadableFile : public ReadableFile {
 public:
  PosixReadableFile(absl::string_view filename, bool is_binary = false, char delim = '\n')
      : delim_(delim), mem_(nullptr), head_(nullptr), stdin_(filename.empty()) {
    if (!stdin_) {
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
    } else {
      mem_ = new char[STDIN_BLOCK_SIZE];
      head_ = mem_;
      file_size_ = 0;
    }
  }

  ~PosixReadableFile() {
    if (mem_ != nullptr) {
      if (stdin_) {
        delete[] mem_;
      } else {
        munmap(mem_, file_size_);
      }
    }
  }

  util::Status status() const { return status_; }

  bool ReadBuffer(std::string *buffer) {
     if (mem_ == nullptr && stdin_) {
       std::cin.read(buffer->data(), buffer->size());
       if (std::cin.fail()) {
         SetErrorStatus(util::StatusCode::kOutOfRange, "stdin");
         return false;
       }
       return true;
     }
     size_t size_left = file_size_ - (head_ - mem_);
     if (size_left < buffer->size()) {
       SetErrorStatus(util::StatusCode::kOutOfRange, "N/A");
       return false;
     }
     memcpy(buffer->data(), head_, buffer->size());
     head_ += buffer->size();
     return true;
  }
  
  bool ReadLine(absl::string_view *line) {
    size_t size_left = file_size_ - (head_ - mem_);
    if (size_left == 0) {
      if (stdin_) {
        auto bytes_read = read(STDIN_FILENO, mem_, STDIN_BLOCK_SIZE);
        if (bytes_read == -1) {
          // Error happened when reading from stdin
          SetErrorStatus(util::StatusCode::kUnavailable, strerror(errno));
          return false;
        }
        if (bytes_read == 0) {
          // EOF reached on stdin
          return false;
        }
        head_ = mem_;
        file_size_ = bytes_read;
        size_left = file_size_;
      } else {
        return false;
      }
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

  bool ReadLineStdin(ps_string *line) {
    if (stdin_) {
      std::string tmp;
      auto worked = static_cast<bool>(std::getline(std::cin, tmp, delim_));
      if (worked) {
        *line = tmp;
      }
      return worked;
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
  bool stdin_;

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
    if (!os_->good()) {
      status_ =
          util::StatusBuilder(util::StatusCode::kDataLoss, GTL_LOC)
          << util::StrError(errno);
      return false;
    }
    return true;
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
