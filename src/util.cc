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

#include "util.h"
#include <iostream>

namespace sentencepiece {
namespace string_util {

template <typename T>
std::vector<T> SplitInternal(const T &str, const T &delim) {
  std::vector<T> result;
  size_t current_pos = 0;
  size_t found_pos = 0;
  while ((found_pos = str.find_first_of(delim, current_pos)) != T::npos) {
    if (found_pos > current_pos) {
      result.push_back(str.substr(current_pos, found_pos - current_pos));
    }
    current_pos = found_pos + 1;
  }
  if (str.size() > current_pos) {
    result.push_back(str.substr(current_pos, str.size() - current_pos));
  }
  return result;
}

std::vector<std::string> Split(const std::string &str,
                               const std::string &delim) {
  return SplitInternal<std::string>(str, delim);
}

std::vector<StringPiece> SplitPiece(StringPiece str, StringPiece delim) {
  return SplitInternal<StringPiece>(str, delim);
}

std::string Join(const std::vector<std::string> &tokens, StringPiece delim) {
  std::string result;
  if (!tokens.empty()) {
    result.append(tokens[0]);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    result.append(tokens[i]);
  }
  return result;
}

std::string Join(const std::vector<int> &tokens, StringPiece delim) {
  std::string result;
  char buf[32];
  if (!tokens.empty()) {
    const size_t len = Itoa(tokens[0], buf);
    result.append(buf, len);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    const size_t len = Itoa(tokens[i], buf);
    result.append(buf, len);
  }
  return result;
}

std::string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                          bool replace_all) {
  std::string ret;
  StringReplace(s, oldsub, newsub, replace_all, &ret);
  return ret;
}

void StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                   bool replace_all, std::string *res) {
  if (oldsub.empty()) {
    res->append(s.data(), s.size());
    return;
  }

  StringPiece::size_type start_pos = 0;
  do {
    const StringPiece::size_type pos = s.find(oldsub, start_pos);
    if (pos == StringPiece::npos) {
      break;
    }
    res->append(s.data() + start_pos, pos - start_pos);
    res->append(newsub.data(), newsub.size());
    start_pos = pos + oldsub.size();
  } while (replace_all);
  res->append(s.data() + start_pos, s.size() - start_pos);
}

// mblen sotres the number of bytes consumed after decoding.
// decoder_utf8 is optimized for speed. It doesn't check
// the following malformed UTF8:
// 1) Redundant UTF8
// 2) BOM (returns value is undefined).
// 3) Trailing byte after leading byte (c & 0xc0 == 0x80)
char32 DecodeUTF8(const char *begin, const char *end, size_t *mblen) {
  const size_t len = end - begin;
  if (len >= 3 && (begin[0] & 0xf0) == 0xe0) {
    *mblen = 3;
    return (((begin[0] & 0x0f) << 12) | ((begin[1] & 0x3f) << 6) |
            ((begin[2] & 0x3f)));
  } else if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xe0) == 0xc0) {
    *mblen = 2;
    return (((begin[0] & 0x1f) << 6) | ((begin[1] & 0x3f)));
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xf0) {
    *mblen = 4;
    return (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3f) << 12) |
            ((begin[2] & 0x3f) << 6) | ((begin[3] & 0x3f)));
  } else if (len >= 5 && (begin[0] & 0xfc) == 0xf8) {
    *mblen = 5;
    return (((begin[0] & 0x03) << 24) | ((begin[1] & 0x3f) << 18) |
            ((begin[2] & 0x3f) << 12) | ((begin[3] & 0x3f) << 6) |
            ((begin[4] & 0x3f)));
  } else if (len >= 6 && (begin[0] & 0xfe) == 0xfc) {
    *mblen = 6;
    return (((begin[0] & 0x01) << 30) | ((begin[1] & 0x3f) << 24) |
            ((begin[2] & 0x3f) << 18) | ((begin[3] & 0x3f) << 12) |
            ((begin[4] & 0x3f) << 6) | ((begin[5] & 0x3f)));
  }

  *mblen = 1;
  return 0;
}

size_t EncodeUTF8(char32 c, char *output) {
  if (c == 0) {
    // Do nothing if |c| is NUL. Previous implementation of UCS4ToUTF8Append
    // worked like this.
    output[0] = '\0';
    return 0;
  }
  if (c < 0x00080) {
    output[0] = static_cast<char>(c & 0xFF);
    output[1] = '\0';
    return 1;
  }
  if (c < 0x00800) {
    output[0] = static_cast<char>(0xC0 + ((c >> 6) & 0x1F));
    output[1] = static_cast<char>(0x80 + (c & 0x3F));
    output[2] = '\0';
    return 2;
  }
  if (c < 0x10000) {
    output[0] = static_cast<char>(0xE0 + ((c >> 12) & 0x0F));
    output[1] = static_cast<char>(0x80 + ((c >> 6) & 0x3F));
    output[2] = static_cast<char>(0x80 + (c & 0x3F));
    output[3] = '\0';
    return 3;
  }
  if (c < 0x200000) {
    output[0] = static_cast<char>(0xF0 + ((c >> 18) & 0x07));
    output[1] = static_cast<char>(0x80 + ((c >> 12) & 0x3F));
    output[2] = static_cast<char>(0x80 + ((c >> 6) & 0x3F));
    output[3] = static_cast<char>(0x80 + (c & 0x3F));
    output[4] = '\0';
    return 4;
  }
  // below is not in UCS4 but in 32bit int.
  if (c < 0x8000000) {
    output[0] = static_cast<char>(0xF8 + ((c >> 24) & 0x03));
    output[1] = static_cast<char>(0x80 + ((c >> 18) & 0x3F));
    output[2] = static_cast<char>(0x80 + ((c >> 12) & 0x3F));
    output[3] = static_cast<char>(0x80 + ((c >> 6) & 0x3F));
    output[4] = static_cast<char>(0x80 + (c & 0x3F));
    output[5] = '\0';
    return 5;
  }
  output[0] = static_cast<char>(0xFC + ((c >> 30) & 0x01));
  output[1] = static_cast<char>(0x80 + ((c >> 24) & 0x3F));
  output[2] = static_cast<char>(0x80 + ((c >> 18) & 0x3F));
  output[3] = static_cast<char>(0x80 + ((c >> 12) & 0x3F));
  output[4] = static_cast<char>(0x80 + ((c >> 6) & 0x3F));
  output[5] = static_cast<char>(0x80 + (c & 0x3F));
  output[6] = '\0';
  return 6;
}

std::string UnicodeCharToUTF8(const char32 c) { return UnicodeTextToUTF8({c}); }

UnicodeText UTF8ToUnicodeText(StringPiece utf8) {
  UnicodeText uc;
  const char *begin = utf8.data();
  const char *end = utf8.data() + utf8.size();
  while (begin < end) {
    size_t mblen;
    const char32 c = DecodeUTF8(begin, end, &mblen);
    uc.push_back(c);
    begin += mblen;
  }
  return uc;
}

std::string UnicodeTextToUTF8(const UnicodeText &utext) {
  char buf[8];
  std::string result;
  for (const char32 c : utext) {
    const size_t mblen = EncodeUTF8(c, buf);
    result.append(buf, mblen);
  }
  return result;
}
}  // namespace string_util

namespace io {

InputBuffer::InputBuffer(StringPiece filename)
    : is_(filename.empty() ? &std::cin
                           : new std::ifstream(WPATH(filename.data()))) {
  CHECK_IFS(*is_, filename.data());
}

InputBuffer::~InputBuffer() {
  if (is_ != &std::cin) {
    delete is_;
  }
}

bool InputBuffer::ReadLine(std::string *line) {
  return static_cast<bool>(std::getline(*is_, *line));
}

OutputBuffer::OutputBuffer(StringPiece filename)
    : os_(filename.empty()
              ? &std::cout
              : new std::ofstream(WPATH(filename.data()), OUTPUT_MODE)) {
  CHECK_OFS(*os_, filename.data());
}

OutputBuffer::~OutputBuffer() {
  if (os_ != &std::cout) {
    delete os_;
  }
}

void OutputBuffer::Write(StringPiece text) {
  os_->write(text.data(), text.size());
}

void OutputBuffer::WriteLine(StringPiece text) {
  Write(text);
  Write("\n");
}
}  // namespace io
}  // namespace sentencepiece
