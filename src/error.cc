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

#include <cstring>
#include "common.h"
#include "sentencepiece_processor.h"

namespace sentencepiece {
namespace error {
int gTestCounter = 0;

void Abort() {
  if (GetTestCounter() == 1) {
    SetTestCounter(2);
  } else {
    std::cerr << "Program terminated with an unrecoverable error." << std::endl;
    exit(-1);
  }
}

void Exit(int code) {
  if (GetTestCounter() == 1) {
    SetTestCounter(2);
  } else {
    exit(code);
  }
}

void SetTestCounter(int c) { gTestCounter = c; }
bool GetTestCounter() { return gTestCounter; }
}  // namespace error

namespace util {

Status::Status() {}
Status::~Status() {}

struct Status::Rep {
  error::Code code;
  std::string error_message;
};

Status::Status(error::Code code, const char* error_message) : rep_(new Rep) {
  rep_->code = code;
  rep_->error_message = error_message;
}

Status::Status(error::Code code, const std::string& error_message)
    : rep_(new Rep) {
  rep_->code = code;
  rep_->error_message = error_message;
}

Status::Status(const Status& s)
    : rep_((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_)) {}

void Status::operator=(const Status& s) {
  if (rep_ != s.rep_)
    rep_.reset((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_));
}

bool Status::operator==(const Status& s) const { return (rep_ == s.rep_); }

bool Status::operator!=(const Status& s) const { return (rep_ != s.rep_); }

const char* Status::error_message() const {
  return ok() ? "" : rep_->error_message.c_str();
}

void Status::set_error_message(const char* str) {
  if (rep_ == nullptr) rep_.reset(new Rep);
  rep_->error_message = str;
}

error::Code Status::code() const { return ok() ? error::OK : rep_->code; }

std::string Status::ToString() const {
  if (rep_ == nullptr) return "OK";

  std::string result;
  switch (code()) {
    case error::CANCELLED:
      result = "Cancelled";
      break;
    case error::UNKNOWN:
      result = "Unknown";
      break;
    case error::INVALID_ARGUMENT:
      result = "Invalid argument";
      break;
    case error::DEADLINE_EXCEEDED:
      result = "Deadline exceeded";
      break;
    case error::NOT_FOUND:
      result = "Not found";
      break;
    case error::ALREADY_EXISTS:
      result = "Already exists";
      break;
    case error::PERMISSION_DENIED:
      result = "Permission denied";
      break;
    case error::UNAUTHENTICATED:
      result = "Unauthenticated";
      break;
    case error::RESOURCE_EXHAUSTED:
      result = "Resource exhausted";
      break;
    case error::FAILED_PRECONDITION:
      result = "Failed precondition";
      break;
    case error::ABORTED:
      result = "Aborted";
      break;
    case error::OUT_OF_RANGE:
      result = "Out of range";
      break;
    case error::UNIMPLEMENTED:
      result = "Unimplemented";
      break;
    case error::INTERNAL:
      result = "Internal";
      break;
    case error::UNAVAILABLE:
      result = "Unavailable";
      break;
    case error::DATA_LOSS:
      result = "Data loss";
      break;
    default:
      result = "Unknown code:";
      break;
  }

  result += ": ";
  result += rep_->error_message;
  return result;
}

void Status::IgnoreError() {}

}  // namespace util
}  // namespace sentencepiece
