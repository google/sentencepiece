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

#ifndef FLAGS_H_
#define FLAGS_H_

#include <memory>
#include <string>
#include <vector>

namespace sentencepiece {
namespace flags {

enum { I, B, I64, U64, D, S };

struct Flag;

class FlagRegister {
 public:
  FlagRegister(const char *name, void *storage, const void *default_storage,
               int shorttpe, const char *help);
  ~FlagRegister();

 private:
  std::unique_ptr<Flag> flag_;
};

std::string PrintHelp(const char *programname);

void ParseCommandLineFlags(int argc, char **argv,
                           std::vector<std::string> *rest_args = nullptr);
}  // namespace flags
}  // namespace sentencepiece

#define DEFINE_VARIABLE(type, shorttype, name, value, help)               \
  namespace sentencepiece_flags_fL##shorttype {                           \
    using namespace sentencepiece::flags;                                 \
    type FLAGS_##name = value;                                            \
    static const type FLAGS_DEFAULT_##name = value;                       \
    static const sentencepiece::flags::FlagRegister fL##name(             \
        #name, reinterpret_cast<void *>(&FLAGS_##name),                   \
        reinterpret_cast<const void *>(&FLAGS_DEFAULT_##name), shorttype, \
        help);                                                            \
  }                                                                       \
  using sentencepiece_flags_fL##shorttype::FLAGS_##name

#define DECLARE_VARIABLE(type, shorttype, name) \
  namespace sentencepiece_flags_fL##shorttype { \
    extern type FLAGS_##name;                   \
  }                                             \
  using sentencepiece_flags_fL##shorttype::FLAGS_##name

#define DEFINE_int32(name, value, help) \
  DEFINE_VARIABLE(int32, I, name, value, help)
#define DECLARE_int32(name) DECLARE_VARIABLE(int32, I, name)

#define DEFINE_int64(name, value, help) \
  DEFINE_VARIABLE(int64, I64, name, value, help)
#define DECLARE_int64(name) DECLARE_VARIABLE(int64, I64, name)

#define DEFINE_uint64(name, value, help) \
  DEFINE_VARIABLE(uint64, U64, name, value, help)
#define DECLARE_uint64(name) DECLARE_VARIABLE(uint64, U64, name)

#define DEFINE_double(name, value, help) \
  DEFINE_VARIABLE(double, D, name, value, help)
#define DECLARE_double(name) DECLARE_VARIABLE(double, D, name)

#define DEFINE_bool(name, value, help) \
  DEFINE_VARIABLE(bool, B, name, value, help)
#define DECLARE_bool(name) DECLARE_VARIABLE(bool, B, name)

#define DEFINE_string(name, value, help) \
  DEFINE_VARIABLE(std::string, S, name, value, help)
#define DECLARE_string(name) DECLARE_VARIABLE(std::string, S, name)

#define CHECK_OR_HELP(flag)                                        \
  if (FLAGS_##flag.empty()) {                                      \
    std::cout << "ERROR: --" << #flag << " must not be empty\n\n"; \
    std::cout << sentencepiece::flags::PrintHelp(PACKAGE_STRING);  \
    sentencepiece::error::Exit(0);                                 \
  }

#endif  // FLAGS_H_
