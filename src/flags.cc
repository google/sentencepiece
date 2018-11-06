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

#include "flags.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "common.h"
#include "config.h"
#include "util.h"

namespace sentencepiece {
namespace flags {

struct Flag {
  int type;
  void *storage;
  const void *default_storage;
  std::string help;
};

static int32 g_minloglevel = 0;

int GetMinLogLevel() { return g_minloglevel; }
void SetMinLogLevel(int minloglevel) { g_minloglevel = minloglevel; }

namespace {
using FlagMap = std::map<std::string, Flag *>;

FlagMap *GetFlagMap() {
  static FlagMap flag_map;
  return &flag_map;
}

bool SetFlag(const std::string &name, const std::string &value) {
  auto it = GetFlagMap()->find(name);
  if (it == GetFlagMap()->end()) {
    return false;
  }

  std::string v = value;
  Flag *flag = it->second;

  // If empty value is set, we assume true or emtpy string is set
  // for boolean or string option. With other types, setting fails.
  if (value.empty()) {
    switch (flag->type) {
      case B:
        v = "true";
        break;
      case S:
        v = "";
        break;
      default:
        return false;
    }
  }

#define DEFINE_ARG(FLAG_TYPE, CPP_TYPE)                        \
  case FLAG_TYPE: {                                            \
    CPP_TYPE *s = reinterpret_cast<CPP_TYPE *>(flag->storage); \
    CHECK(string_util::lexical_cast<CPP_TYPE>(v, s));          \
    break;                                                     \
  }

  switch (flag->type) {
    DEFINE_ARG(I, int32);
    DEFINE_ARG(B, bool);
    DEFINE_ARG(I64, int64);
    DEFINE_ARG(U64, uint64);
    DEFINE_ARG(D, double);
    DEFINE_ARG(S, std::string);
    default:
      break;
  }

  return true;
}  // namespace

bool CommandLineGetFlag(int argc, char **argv, std::string *key,
                        std::string *value, int *used_args) {
  key->clear();
  value->clear();

  *used_args = 1;
  const char *start = argv[0];
  if (start[0] != '-') {
    return false;
  }

  ++start;
  if (start[0] == '-') ++start;
  const std::string arg = start;
  const size_t n = arg.find("=");
  if (n != std::string::npos) {
    *key = arg.substr(0, n);
    *value = arg.substr(n + 1, arg.size() - n);
    return true;
  }

  key->assign(arg);
  value->clear();

  if (argc == 1) {
    return true;
  }
  start = argv[1];
  if (start[0] == '-') {
    return true;
  }

  *used_args = 2;
  value->assign(start);
  return true;
}
}  // namespace

FlagRegister::FlagRegister(const char *name, void *storage,
                           const void *default_storage, int shortype,
                           const char *help)
    : flag_(new Flag) {
  flag_->type = shortype;
  flag_->storage = storage;
  flag_->default_storage = default_storage;
  flag_->help = help;
  GetFlagMap()->insert(std::make_pair(std::string(name), flag_.get()));
}

FlagRegister::~FlagRegister() {}

std::string PrintHelp(const char *programname) {
  std::ostringstream os;
  os << PACKAGE_STRING << "\n\n";
  os << "Usage: " << programname << " [options] files\n\n";

  for (const auto &it : *GetFlagMap()) {
    os << "   --" << it.first << " (" << it.second->help << ")";
    const Flag *flag = it.second;
    switch (flag->type) {
      case I:
        os << "  type: int32  default: "
           << *(reinterpret_cast<const int *>(flag->default_storage)) << '\n';
        break;
      case B:
        os << "  type: bool  default: "
           << (*(reinterpret_cast<const bool *>(flag->default_storage))
                   ? "true"
                   : "false")
           << '\n';
        break;
      case I64:
        os << "  type: int64 default: "
           << *(reinterpret_cast<const int64 *>(flag->default_storage)) << '\n';
        break;
      case U64:
        os << "  type: uint64  default: "
           << *(reinterpret_cast<const uint64 *>(flag->default_storage))
           << '\n';
        break;
      case D:
        os << "  type: double  default: "
           << *(reinterpret_cast<const double *>(flag->default_storage))
           << '\n';
        break;
      case S:
        os << "  type: string  default: "
           << *(reinterpret_cast<const std::string *>(flag->default_storage))
           << '\n';
        break;
      default:
        break;
    }
  }

  os << "\n\n";

  return os.str();
}

void ParseCommandLineFlags(int argc, char **argv,
                           std::vector<std::string> *rest_flags) {
  int used_argc = 0;
  std::string key, value;

  for (int i = 1; i < argc; i += used_argc) {
    if (!CommandLineGetFlag(argc - i, argv + i, &key, &value, &used_argc)) {
      if (rest_flags) rest_flags->push_back(std::string(argv[i]));
      continue;
    }
    if (key == "help") {
      std::cout << PrintHelp(argv[0]);
      error::Exit(0);
    } else if (key == "version") {
      std::cout << PACKAGE_STRING << " " << VERSION << std::endl;
      error::Exit(0);
    } else if (key == "minloglevel") {
      flags::SetMinLogLevel(atoi(value.c_str()));
    } else if (!SetFlag(key, value)) {
      std::cerr << "Unknown/Invalid flag " << key << "\n\n"
                << PrintHelp(argv[0]);
      error::Exit(1);
    }
  }
}
}  // namespace flags
}  // namespace sentencepiece
