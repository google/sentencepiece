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

#ifndef STRINGPIECE_H_
#define STRINGPIECE_H_

#include <cstring>
#include <string>

namespace sentencepiece {

class StringPiece {
 public:
  typedef size_t size_type;

  // Create an empty slice.
  StringPiece() : data_(""), size_(0) {}

  // Create a slice that refers to d[0,n-1].
  StringPiece(const char *d, size_t n) : data_(d), size_(n) {}

  // Create a slice that refers to the contents of "s"
  StringPiece(const std::string &s) : data_(s.data()), size_(s.size()) {}

  // Create a slice that refers to s[0,strlen(s)-1]
  StringPiece(const char *s) : data_(s), size_(strlen(s)) {}

  void set(const void *data, size_t len) {
    data_ = reinterpret_cast<const char *>(data);
    size_ = len;
  }

  void set(const char *data) {
    data_ = data;
    size_ = strlen(data);
  }

  // Return a pointer to the beginning of the referenced data
  const char *data() const { return data_; }

  // Return the length (in bytes) of the referenced data
  size_t size() const { return size_; }

  // Return true iff the length of the referenced data is zero
  bool empty() const { return size_ == 0; }

  typedef const char *const_iterator;
  typedef const char *iterator;
  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  static const size_type npos = static_cast<size_type>(-1);

  char operator[](size_t n) const { return data_[n]; }

  // Change this slice to refer to an empty array
  void clear() {
    data_ = "";
    size_ = 0;
  }

  // Drop the first "n" bytes from this slice.
  void remove_prefix(size_t n) {
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) { size_ -= n; }

  size_type find(StringPiece s, size_type pos = 0) const {
    if (size_ <= 0 || pos > static_cast<size_type>(size_)) {
      if (size_ == 0 && pos == 0 && s.size_ == 0) {
        return 0;
      }
      return npos;
    }
    const char *result = memmatch(data_ + pos, size_ - pos, s.data_, s.size_);
    return result ? result - data_ : npos;
  }

  size_type find(char c, size_type pos) const {
    if (size_ <= 0 || pos >= static_cast<size_type>(size_)) {
      return npos;
    }
    const char *result =
        static_cast<const char *>(memchr(data_ + pos, c, size_ - pos));
    return result != nullptr ? result - data_ : npos;
  }

  size_type find_first_of(char c, size_type pos = 0) const {
    return find(c, pos);
  }

  size_type find_first_of(StringPiece s, size_type pos = 0) const {
    if (size_ <= 0 || s.size_ <= 0) {
      return npos;
    }

    if (s.size_ == 1) {
      return find_first_of(s.data_[0], pos);
    }

    bool lookup[256] = {false};
    for (size_t i = 0; i < s.size_; ++i) {
      lookup[static_cast<unsigned char>(s.data_[i])] = true;
    }
    for (size_t i = pos; i < size_; ++i) {
      if (lookup[static_cast<unsigned char>(data_[i])]) {
        return i;
      }
    }

    return npos;
  }

  bool Consume(StringPiece x) {
    if (starts_with(x)) {
      remove_prefix(x.size_);
      return true;
    }
    return false;
  }

  StringPiece substr(size_type pos, size_type n = npos) const {
    size_type size = static_cast<size_type>(size_);
    if (pos > size) pos = size;
    if (n > size - pos) n = size - pos;
    return StringPiece(data_ + pos, n);
  }

  // Return a string that contains the copy of the referenced data.
  std::string ToString() const { return std::string(data_, size_); }
  std::string to_string() const { return std::string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(StringPiece b) const;

  // Return true iff "x" is a prefix of "*this"
  bool starts_with(StringPiece x) const {
    return ((size_ >= x.size_) && (memcmp(data_, x.data_, x.size_) == 0));
  }
  // Return true iff "x" is a suffix of "*this"
  bool ends_with(StringPiece x) const {
    return ((size_ >= x.size_) &&
            (memcmp(data_ + (size_ - x.size_), x.data_, x.size_) == 0));
  }

 private:
  static const char *memmatch(const char *phaystack, size_t haylen,
                              const char *pneedle, size_t neelen) {
    if (0 == neelen) {
      return phaystack;  // even if haylen is 0
    }
    if (haylen < neelen) {
      return nullptr;
    }
    const char *match;
    const char *hayend = phaystack + haylen - neelen + 1;
    while ((match = (const char *)(memchr(phaystack, pneedle[0],
                                          hayend - phaystack)))) {
      if (memcmp(match, pneedle, neelen) == 0) {
        return match;
      } else {
        phaystack = match + 1;
      }
    }
    return nullptr;
  }

  const char *data_;
  size_t size_;
};

inline bool operator==(StringPiece x, StringPiece y) {
  return ((x.size() == y.size()) &&
          (memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(StringPiece x, StringPiece y) { return !(x == y); }

inline bool operator<(StringPiece x, StringPiece y) { return x.compare(y) < 0; }
inline bool operator>(StringPiece x, StringPiece y) { return x.compare(y) > 0; }
inline bool operator<=(StringPiece x, StringPiece y) {
  return x.compare(y) <= 0;
}
inline bool operator>=(StringPiece x, StringPiece y) {
  return x.compare(y) >= 0;
}

inline int StringPiece::compare(StringPiece b) const {
  const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
  int r = memcmp(data_, b.data_, min_len);
  if (r == 0) {
    if (size_ < b.size_) {
      r = -1;
    } else if (size_ > b.size_) {
      r = +1;
    }
  }
  return r;
}

inline std::ostream &operator<<(std::ostream &o, StringPiece piece) {
  o << piece.data();
  return o;
}

struct StringPieceHash {
  // DJB hash function.
  inline size_t operator()(const StringPiece &sp) const {
    size_t hash = 5381;
    for (size_t i = 0; i < sp.size(); ++i) {
      hash = ((hash << 5) + hash) + sp[i];
    }
    return hash;
  }
};
}  // namespace sentencepiece

#endif  // STRINGPIECE_H_
