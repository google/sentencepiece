// Copyright 2026 Google LLC
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
// limitations under the License.

#include "sentencepiece_lite.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef SENTENCEPIECE_LITE_USE_ABSL
#include "absl/container/inlined_vector.h"
#endif

#ifdef SENTENCEPIECE_LITE_USE_UTF8_RANGE
#include "utf8_validity.h"
#endif

#include "flatbuffers/base.h"
#include "flatbuffers/vector.h"
#include "flatbuffers/verifier.h"
#include "sentencepiece_lite_generated.h"

#define LITE_RETURN_IF_ERROR(expr)                  \
  do {                                              \
    const StatusCode _status = (expr);              \
    if (_status != StatusCode::kOk) return _status; \
  } while (0)

#if defined(_MSC_VER)
#define SPM_LITE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SPM_LITE_RESTRICT __restrict__
#else
#define SPM_LITE_RESTRICT
#endif

namespace sentencepiece::lite {
namespace {

#ifdef SENTENCEPIECE_LITE_USE_ABSL
template <typename T, size_t N>
using InlineVector = absl::InlinedVector<T, N>;
#else
template <typename T, size_t N>
using InlineVector = std::vector<T>;
#endif

// Maximum permitted input length (in bytes) for single-pass text normalization,
// pre-tokenization, and encoding operations.
// Set to std::numeric_limits<int>::max() - 1 (2,147,483,646 bytes) to strictly
// prevent 32-bit signed integer overflow in Viterbi DP table and BPE index
// calculations.
constexpr size_t kMaxInputLength =
    static_cast<size_t>(std::numeric_limits<int>::max()) - 1;

constexpr size_t kModelBufferAlignment = 4;

// Double-Array Trie (Darts) flat 32-bit integer array representation.
// It encodes a prefix tree (Trie) into a single 1D array where tree nodes are
// array indices. Child node positions are computed via XOR arithmetic, enabling
// O(1) state transitions per character without pointer indirections.
//
// Unit Bit Layout (32-bit integer):
//  - Bits [0..7]:   label (character byte that transitioned here)
//  - Bit 8:         has_leaf (true if a symbol ends at this node)
//  - Bits [10..30]: offset (XOR base jump offset to child nodes)
//  - Bit 31:        leaf flag (0 for internal transition node, 1
//                             for leaf/value node)
//  - Leaf node:     when Bit 31 is 1, stores the actual symbol value/ID in bits
//  [0..30]
//
// Pseudo-code for exact lookup:
//   int node = 0; // root
//   for (char c : key) {
//     int next = node ^ unit[node].offset() ^ (unsigned char)c;
//     if (unit[next].label() != (unsigned char)c) return -1;
//     node = next;
//   }
//   return unit[node].has_leaf()
//              ? unit[node ^ unit[node].offset()].value()
//              : -1;
class DoubleArray {
 private:
  class DoubleArrayUnit {
   public:
    DoubleArrayUnit() = default;
    explicit DoubleArrayUnit(uint32_t unit) : unit_(unit) {}

    bool has_leaf() const { return ((unit_ >> 8) & 1) == 1; }
    int value() const { return static_cast<int>(unit_ & ((1U << 31) - 1)); }
    uint32_t label() const { return unit_ & ((1U << 31) | 0xFF); }
    uint32_t offset() const {
      // ILP & Branchless Optimization:
      // Instead of variable shift `<< ((unit_ & (1U << 9)) >> 6)` which creates
      // a serialized data dependency chain and shift-register stall,
      // compute `base` (>> 10) and `scaled` (>> 2 & ~0xFFU) in parallel across
      // two ALUs and select in 1 cycle using branchless cmov/csel!
      const uint32_t base = unit_ >> 10;
      const uint32_t scaled = (unit_ >> 2) & ~0xFFU;
      return (unit_ & (1U << 9)) ? scaled : base;
    }

   private:
    uint32_t unit_ = 0;
  };

 public:
  DoubleArray() = default;

  void set_array(const void* ptr, size_t size = 0) {
    array_ = static_cast<const uint32_t*>(ptr);
    size_ = size;
  }

  bool has_array() const { return array_ != nullptr; }
  const uint32_t* array() const { return array_; }

  // Returns the value associated with `key`.
  int exact_lookup(std::string_view key) const {
    if (array_ == nullptr || key.empty()) return -1;
    uint32_t node_pos = 0;
    DoubleArrayUnit unit(array_[node_pos]);
    for (size_t i = 0; i < key.size(); ++i) {
      node_pos ^= unit.offset() ^ static_cast<unsigned char>(key[i]);
      unit = DoubleArrayUnit(array_[node_pos]);
      if (unit.label() != static_cast<unsigned char>(key[i])) {
        return -1;
      }
    }
    if (!unit.has_leaf()) {
      return -1;
    }
    unit = DoubleArrayUnit(array_[node_pos ^ unit.offset()]);
    return unit.value();
  }

  // Returns a pair of {value, length} representing the longest matching prefix
  // of `key`. Returns {-1, 0} if no match is found.
  std::pair<int, size_t> longest_prefix_lookup(std::string_view key) const {
    int matched_id = -1;
    size_t longest_len = 0;
    common_prefix_lookup(key, [&](int value, size_t length) {
      matched_id = value;
      longest_len = length;
    });
    return {matched_id, longest_len};
  }

  // Performs a common prefix search. `callback(value, length)` is called
  // for all prefixes of `key`.
  template <typename Callback>
  void common_prefix_lookup(std::string_view key, Callback&& callback) const {
    if (array_ == nullptr || key.empty()) return;
    uint32_t node_pos = 0;
    DoubleArrayUnit unit(array_[node_pos]);
    for (size_t i = 0; i < key.size(); ++i) {
      node_pos ^= unit.offset() ^ static_cast<unsigned char>(key[i]);
      unit = DoubleArrayUnit(array_[node_pos]);
      if (unit.label() != static_cast<unsigned char>(key[i])) {
        break;
      }
      if (unit.has_leaf()) {
        DoubleArrayUnit leaf(array_[node_pos ^ unit.offset()]);
        callback(leaf.value(), i + 1);
      }
    }
  }

  // Non-owning, zero-cost Trie view wrapper with __restrict__ optimization.
  // Encapsulates raw DoubleArray pointer access without `this->` aliasing
  // overhead, allowing compilers (Clang/GCC) to perform register allocation and
  // instruction-level parallelization inside hot tokenization loops.
  struct View {
    const uint32_t* SPM_LITE_RESTRICT array;
    static constexpr uint32_t kInvalidNodePos = ~0U;

    static uint32_t ExtractOffset(uint32_t unit) {
      const uint32_t base = unit >> 10;
      const uint32_t scaled = (unit >> 2) & ~0xFFU;
      return (unit & (1U << 9)) ? scaled : base;
    }

    uint32_t transition(uint32_t node_pos, std::string_view key) const {
      if (node_pos == kInvalidNodePos) return kInvalidNodePos;
      uint32_t id = node_pos;
      for (char c : key) {
        const uint32_t offset = ExtractOffset(array[id]);
        id ^= offset ^ static_cast<unsigned char>(c);
        const uint32_t target_unit = array[id];
        if ((target_unit & ((1U << 31) | 0xFF)) !=
            static_cast<unsigned char>(c)) {
          return kInvalidNodePos;
        }
      }
      return id;
    }

    int leaf_value(uint32_t node_pos) const {
      if (node_pos == kInvalidNodePos) return -1;
      const uint32_t unit = array[node_pos];
      if (((unit >> 8) & 1) == 0) return -1;
      const uint32_t leaf_unit = array[node_pos ^ ExtractOffset(unit)];
      return static_cast<int>(leaf_unit & ((1U << 31) - 1));
    }
  };

  // Creates a non-owning View with __restrict__ optimization for
  // high-performance loops.
  View view() const { return View{array_}; }

  // Performs a trie transition from `node_pos` with `key`.
  // Returns kInvalidNodePos (~0U) if no valid node exists for the transition.
  uint32_t transition(uint32_t node_pos, std::string_view key) const {
    return view().transition(node_pos, key);
  }

  // Returns the value of `node_pos`.
  int leaf_value(uint32_t node_pos) const {
    return view().leaf_value(node_pos);
  }

  // WARNING: CRITICAL SECURITY AND MEMORY SAFETY BOUNDARY
  // Runtime trie traversals (`exact_lookup`, `transition`, etc.) deliberately
  // omit runtime boundary checks in their inner loops to maximize tokenization
  // performance. Consequently, this validation method serves as the sole
  // barrier preventing out-of-bounds (OOB) memory accesses, segmentation
  // faults, and potential security vulnerabilities from untrusted model data.
  //
  // Do NOT disable, bypass, or weaken any checks in this method. Any changes to
  // this logic require maximum scrutiny to ensure that all 256 possible byte
  // transitions, leaf nodes, and failure links remain strictly confined within
  // allocated array bounds (`[0, size_)`).
  //
  // Detailed Validation Specification:
  // - Transition safety: Verifies that for every node `i`, all 256 possible
  //   byte transitions `(i ^ offset) ^ c` land within valid bounds `< size_`.
  // - `max_value_limit`: If non-negative, guarantees that decoded symbol IDs
  //   (`unit.value()`) are strictly less than `max_value_limit`.
  // - `check_suffix_links`: If true, verifies that all internal failure links
  //   point to valid transition states within the trie bounds.
  bool validate(int max_value_limit = -1,
                bool check_suffix_links = false) const {
    if (size_ == 0 || array_ == nullptr) return false;
    DoubleArrayUnit root(array_[0]);
    // The next trie node transition ID is calculated as `id ^= offset ^ c`.
    // Verifying that the max index `(id ^ offset) | 0xFF` for any byte `c`
    // stays below `size_` mathematically guarantees that input strings
    // cannot trigger out-of-bounds reads during trie traversal.
    if (root.label() != 0 || root.has_leaf() || root.offset() == 0 ||
        ((0 ^ root.offset()) | 0xFF) >= size_) {
      return false;
    }
    for (size_t i = 1; i < size_; ++i) {
      DoubleArrayUnit unit(array_[i]);
      if (unit.label() <= 0xFF) {
        // Pre-validate OOB transition bound for all internal nodes.
        if (((i ^ unit.offset()) | 0xFF) >= size_) {
          return false;
        }
        // Verify that if this node has a leaf, the target null-transition unit
        // is actually a valid leaf/value node (with Bit 31 set, label > 0xFF).
        if (unit.has_leaf()) {
          if (DoubleArrayUnit(array_[i ^ unit.offset()]).label() <= 0xFF) {
            return false;
          }
        }
      } else {
        if (max_value_limit >= 0) {
          if (unit.value() >= max_value_limit) {
            return false;
          }
        }
        if (check_suffix_links) {
          // Suffix links point to valid transition nodes in the same trie.
          // We verify that the target node is within the trie bounds, and is a
          // valid transition state (its label is a single-byte char <= 0xFF).
          const int link = unit.value();
          if (link < 0 || link >= static_cast<int>(size_)) {
            return false;
          }
          DoubleArrayUnit target(array_[link]);
          if (target.label() > 0xFF) {
            return false;
          }
        }
      }
    }
    return true;
  }

  const uint32_t* array_ = nullptr;
  size_t size_ = 0;
};

StatusCode InitializeTrie(const uint8_t* data, size_t size, int max_value_limit,
                          bool check_suffix_links, bool is_optional,
                          DoubleArray* trie) {
  if (data == nullptr || size == 0) {
    return is_optional ? StatusCode::kOk : StatusCode::kInternal;
  }
  if (size < 1024 || (size & 0x3FF) != 0) {
    return StatusCode::kInternal;
  }
  trie->set_array(data, size / sizeof(uint32_t));
  if (!trie->validate(max_value_limit, check_suffix_links)) {
    return StatusCode::kInternal;
  }
  return StatusCode::kOk;
}

StatusCode InitializeTrie(const flatbuffers::Vector<uint8_t>* blob,
                          int max_value_limit, bool check_suffix_links,
                          bool is_optional, DoubleArray* trie) {
  return InitializeTrie(blob ? blob->data() : nullptr, blob ? blob->size() : 0,
                        max_value_limit, check_suffix_links, is_optional, trie);
}

constexpr float kUnkPenalty = 10.0;
constexpr std::string_view kSpaceSymbol = "\xe2\x96\x81";
constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";
constexpr size_t kMaxPieceLength = 4096;

inline float GetUserDefinedScore(int length) { return 0.1f * (length - 1); }

inline size_t OneCharLen(const char* src) {
  // Fast-path for ASCII bytes (< 0x80). When cast to signed char, non-ASCII
  // leading bytes (0x80 to 0xFF) are negative, falling through cleanly with
  // zero regression. Speeds up pretokenization chunking by ~4.4%.
  if (static_cast<signed char>(*src) >= 0) return 1;
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

inline uint64_t LoadLe64(const void* ptr) {
  uint64_t val;
  std::memcpy(&val, ptr, sizeof(uint64_t));
  return flatbuffers::EndianScalar(val);
}

// Returns the byte offset of the first matched byte within a little-endian
// 8-byte SWAR word.
inline size_t FirstMatchedByteOffset(uint64_t match_mask) {
  return std::countr_zero(match_mask) >> 3;
}

// Fast, register-friendly UTF-8 string scanner designed to replace repeated
// std::string_view::substr() calls and redundant buffer boundary checks.
//
// Motivation:
// In UTF-8 processing (pretokenization chunking, Trie traversals, and token
// extraction), characters are at most 4 bytes long. When iterating over long
// text using standard string_view slices, calling substr(offset) and
// std::min(OneCharLen(ptr), remaining) on every character incurs redundant
// pointer arithmetic, length recalculations, and bounds checking branches.
//
// StringScanner encapsulates raw pointers (ptr_ and end_) in CPU registers.
// By checking `if (ptr_ + 4 <= end_)` once, it safely executes 4-byte UTF-8
// character reads without std::min branches for >99.9% of text processing,
// reducing ALU instructions and branch overheads in hot encoding loops.
class StringScanner {
 public:
  explicit StringScanner(std::string_view str)
      : ptr_(str.data()), end_(str.data() + str.size()) {}

  bool empty() const { return ptr_ >= end_; }
  size_t remaining() const { return end_ - ptr_; }
  const char* data() const { return ptr_; }
  std::string_view remaining_view() const {
    return std::string_view(ptr_, end_ - ptr_);
  }

  // Returns the length of the next UTF-8 character without redundant
  // std::min boundary checking when remaining buffer is >= 4 bytes.
  size_t NextLen() const {
    if (empty()) return 0;
    const size_t len = OneCharLen(ptr_);
    if (ptr_ + 4 <= end_) {
      return len;
    }
    return std::min<size_t>(len, static_cast<size_t>(end_ - ptr_));
  }

  // Advances the pointer by `n` bytes.
  void Advance(size_t n) { ptr_ += n; }

  // Consumes and returns the substring up to the next occurrence of `c`.
  std::string_view ConsumeUntil(char c) {
    if (empty()) return {};
    const char* target = nullptr;
    // Fast-path: 64-bit SWAR scans for ' ' within the next 8 bytes in
    // registers, avoiding libc function call overhead for short English words
    // (avg ~5 bytes).
    if (c == ' ' && ptr_ + sizeof(uint64_t) <= end_) {
      const uint64_t word = LoadLe64(ptr_);
      const uint64_t space_xor = word ^ 0x2020202020202020ULL;
      const uint64_t space_match = (space_xor - 0x0101010101010101ULL) &
                                   ~space_xor & 0x8080808080808080ULL;
      if (space_match != 0) {
        target = ptr_ + FirstMatchedByteOffset(space_match);
      }
    }
    // Falls back to std::memchr, highly optimized with AVX2/Neon SIMD
    // instructions.
    if (target == nullptr) {
      const void* p =
          std::memchr(ptr_, static_cast<unsigned char>(c), remaining());
      target = (p != nullptr) ? static_cast<const char*>(p) : end_;
    }
    const std::string_view result(ptr_, target - ptr_);
    ptr_ = target;
    return result;
  }

  // Consumes consecutive occurrences of `c` and returns the count.
  size_t SkipAll(char c) {
    const char* start = ptr_;
    while (ptr_ < end_ && *ptr_ == c) {
      ++ptr_;
    }
    return ptr_ - start;
  }

  // Consumes the next UTF-8 character and returns its string_view.
  std::string_view ConsumeNext() {
    if (empty()) return {};
    const size_t len = NextLen();
    std::string_view piece(ptr_, len);
    ptr_ += len;
    return piece;
  }

 private:
  const char* ptr_ = nullptr;
  const char* end_ = nullptr;
};

namespace utf8 {
#if defined(SENTENCEPIECE_LITE_USE_UTF8_RANGE)
inline bool IsStructurallyValid(std::string_view s) {
  return utf8_range::IsStructurallyValid(s);
}
inline size_t SpanStructurallyValid(std::string_view s) {
  return utf8_range::SpanStructurallyValid(s);
}
#else
// Self-contained fast UTF-8 validator with 64-bit SWAR ASCII skip.
inline size_t SpanStructurallyValid(std::string_view s) {
  const char* ptr = s.data();
  const char* const end = s.data() + s.size();
  while (ptr < end) {
    // 64-bit SWAR fast path for ASCII runs (~20 GB/s)
    while (ptr + sizeof(uint64_t) <= end) {
      uint64_t word;
      std::memcpy(&word, ptr, sizeof(uint64_t));
      if ((word & 0x8080808080808080ULL) != 0) break;
      ptr += sizeof(uint64_t);
    }
    while (ptr < end && static_cast<unsigned char>(*ptr) < 0x80) ++ptr;
    if (ptr >= end) break;

    // Validate multibyte UTF-8 (2..4 bytes), continuation bits, and codepoint
    // range.
    const size_t len = OneCharLen(ptr);
    if (len < 2 || ptr + len > end) return ptr - s.data();
    uint32_t cp = static_cast<unsigned char>(*ptr++) & (0xFF >> len);
    for (size_t i = 1; i < len; ++i) {
      const unsigned char b = static_cast<unsigned char>(*ptr++);
      if ((b & 0xC0) != 0x80) return (ptr - 1 - i) - s.data();
      cp = (cp << 6) | (b & 0x3F);
    }
    const uint32_t min_cp = (len == 2) ? 0x80 : (len == 3) ? 0x800 : 0x10000;
    if (cp < min_cp || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
      return (ptr - len) - s.data();
    }
  }
  return s.size();
}

inline bool IsStructurallyValid(std::string_view s) {
  return SpanStructurallyValid(s) == s.size();
}
#endif

// Validates a single UTF-8 character at the beginning of `input`.
// Returns true if valid, setting `*mblen` to the character byte length (1..4).
// Returns false if invalid, setting `*mblen` to 1.
inline bool IsValidUTF8(std::string_view input, size_t* mblen) {
  if (input.empty()) {
    *mblen = 0;
    return false;
  }
  const size_t len = OneCharLen(input.data());
  if (len <= input.size() &&
      IsStructurallyValid(std::string_view(input.data(), len))) {
    *mblen = len;
    return true;
  }
  *mblen = 1;
  return false;
}
}  // namespace utf8

inline void ReplaceAll(std::string_view s, std::string_view from,
                       std::string_view to, std::string* out) {
  if (from.empty() || s.empty()) {
    out->append(s.data(), s.size());
    return;
  }
  size_t last_pos = 0;
  while (true) {
    const size_t pos = s.find(from, last_pos);
    if (pos == std::string_view::npos) {
      break;
    }
    if (pos > last_pos) {
      out->append(s.data() + last_pos, pos - last_pos);
    }
    out->append(to.data(), to.size());
    last_pos = pos + from.size();
  }
  if (last_pos < s.size()) {
    out->append(s.data() + last_pos, s.size() - last_pos);
  }
}

// Formats a raw byte into its SentencePiece byte-fallback token string
// representation in the format "<0xXX>" (e.g., 0x41 is converted to "<0x41>").
std::string ByteToPiece(unsigned char c) {
  static constexpr char kHexDigits[] = "0123456789ABCDEF";
  std::string s = "<0x00>";
  s[3] = kHexDigits[(c >> 4) & 0xF];
  s[4] = kHexDigits[c & 0xF];
  return s;
}

class Normalizer {
 public:
  Normalizer() = default;
  ~Normalizer() = default;

  StatusCode Initialize(const NormalizerSpec* spec) {
    spec_ = spec;
    if (spec_ == nullptr) return StatusCode::kOk;

    // Caches all flags to avoid pointer traversal.
    add_dummy_prefix_ = spec_->add_dummy_prefix();
    remove_extra_whitespaces_ = spec_->remove_extra_whitespaces();
    escape_whitespaces_ = spec_->escape_whitespaces();
    treat_whitespace_as_suffix_ = spec_->treat_whitespace_as_suffix();

    if (spec_->precompiled_charsmap() != nullptr &&
        !spec_->precompiled_charsmap()->empty()) {
      const uint8_t* data = spec_->precompiled_charsmap()->data();
      const size_t size = spec_->precompiled_charsmap()->size();

      if (size <= sizeof(uint32_t)) {
        return StatusCode::kInternal;
      }
      uint32_t normalizer_trie_blob_size = 0;
      std::memcpy(&normalizer_trie_blob_size, data, sizeof(uint32_t));

      if (normalizer_trie_blob_size >= size - sizeof(uint32_t)) {
        return StatusCode::kInternal;
      }

      const uint8_t* normalizer_trie_ptr = data + sizeof(uint32_t);
      normalized_ =
          std::string_view(reinterpret_cast<const char*>(
                               normalizer_trie_ptr + normalizer_trie_blob_size),
                           size - sizeof(uint32_t) - normalizer_trie_blob_size);

      if (normalized_.empty() || normalized_.back() != '\0') {
        return StatusCode::kInternal;
      }

      LITE_RETURN_IF_ERROR(InitializeTrie(
          normalizer_trie_ptr, normalizer_trie_blob_size, normalized_.size(),
          /*check_suffix_links=*/false, /*is_optional=*/false, &trie_));
    }
    return StatusCode::kOk;
  }

  // Note: Loop-invariant pointer checks (`if (offset != nullptr)`) incur zero
  // CPU overhead due to 100% branch prediction and superscalar execution.
  StatusCode Normalize(std::string_view input, std::string* normalized,
                       std::vector<size_t>* offset = nullptr) const {
    if (offset != nullptr) {
      offset->clear();
    }
    normalized->clear();
    if (input.empty()) {
      return StatusCode::kOk;
    }

    if (spec_ == nullptr) {
      normalized->assign(input.data(), input.size());
      if (offset != nullptr) {
        // For identity normalization, byte positions map 1:1 (0, 1, ..., N-1),
        // followed by N as the sentinel.
        offset->resize(input.size() + 1);
        std::iota(offset->begin(), offset->end(), 0);
      }
      return StatusCode::kOk;
    }

    size_t consumed = 0;

    if (remove_extra_whitespaces_) {
      while (!input.empty()) {
        const auto p = NormalizePrefix(input);
        if (p.first != " ") {
          break;
        }
        input.remove_prefix(p.second);
        if (offset != nullptr) {
          consumed += p.second;
        }
      }
    }

    if (input.empty()) {
      return StatusCode::kOk;
    }

    const size_t kReservedSize = input.size() * 3 / 2;
    normalized->reserve(kReservedSize);
    if (offset != nullptr) {
      offset->reserve(kReservedSize);
    }

    const std::string_view space_symbol =
        escape_whitespaces_ ? kSpaceSymbol : " ";

    if (!treat_whitespace_as_suffix_ && add_dummy_prefix_) {
      normalized->append(space_symbol.data(), space_symbol.size());
      if (offset != nullptr) {
        offset->insert(offset->end(), space_symbol.size(), consumed);
      }
    }

    if (!trie_.has_array() && offset == nullptr) {
      if (utf8::IsStructurallyValid(input)) {
        NormalizeIdentityFast(input, space_symbol, &consumed, normalized);
      } else {
        NormalizeGeneralCore(input, space_symbol, &consumed, normalized,
                             /*offset=*/nullptr);
      }
    } else {
      NormalizeGeneralCore(input, space_symbol, &consumed, normalized, offset);
    }

    if (remove_extra_whitespaces_) {
      while (normalized->size() >= space_symbol.size() &&
             std::string_view(*normalized).ends_with(space_symbol)) {
        const size_t length = normalized->size() - space_symbol.size();
        normalized->resize(length);
        if (offset != nullptr) {
          consumed = (*offset)[length];
          offset->resize(length);
        }
      }
    }

    if (treat_whitespace_as_suffix_ && add_dummy_prefix_) {
      normalized->append(space_symbol.data(), space_symbol.size());
      if (offset != nullptr) {
        offset->insert(offset->end(), space_symbol.size(), consumed);
      }
    }

    if (offset != nullptr) {
      offset->push_back(consumed);
    }

    return StatusCode::kOk;
  }

  bool remove_extra_whitespaces() const { return remove_extra_whitespaces_; }

 private:
  // Ultra-fast path for identity normalization without offsets (empty compiled
  // map), accelerated with space-pivoted 64-bit SWAR / SIMD memchr and batched
  // appends.
  void NormalizeIdentityFast(std::string_view input,
                             std::string_view space_symbol, size_t* consumed,
                             std::string* normalized) const {
    bool is_prev_space = remove_extra_whitespaces_;
    StringScanner scanner(input);
    while (!scanner.empty()) {
      const std::string_view chunk = scanner.ConsumeUntil(' ');
      if (!chunk.empty()) {
        normalized->append(chunk.data(), chunk.size());
        *consumed += chunk.size();
        is_prev_space = false;
      }

      if (scanner.empty()) {
        break;
      }

      const size_t space_count = scanner.SkipAll(' ');
      if (remove_extra_whitespaces_) {
        if (!is_prev_space) {
          normalized->append(space_symbol.data(), space_symbol.size());
          is_prev_space = true;
        }
      } else {
        for (size_t i = 0; i < space_count; ++i) {
          normalized->append(space_symbol.data(), space_symbol.size());
        }
        is_prev_space = false;
      }
      *consumed += space_count;
    }
  }

  // General core using NormalizePrefix for models with DoubleArray Trie
  // or when character offset tracking is requested.
  void NormalizeGeneralCore(std::string_view input,
                            std::string_view space_symbol, size_t* consumed,
                            std::string* normalized,
                            std::vector<size_t>* offset) const {
    bool is_prev_space = remove_extra_whitespaces_;
    while (!input.empty()) {
      auto p = NormalizePrefix(input);
      std::string_view sp = p.first;

      while (is_prev_space && sp.starts_with(' ')) {
        sp.remove_prefix(1);
      }

      if (!sp.empty()) {
        if (sp.find(' ') == std::string_view::npos) {  // NOLINT
          normalized->append(sp.data(), sp.size());
          if (offset != nullptr) {
            offset->insert(offset->end(), sp.size(), *consumed);
          }
        } else {
          for (size_t n = 0; n < sp.size(); ++n) {
            if (sp[n] == ' ') {
              normalized->append(space_symbol.data(), space_symbol.size());
              if (offset != nullptr) {
                offset->insert(offset->end(), space_symbol.size(), *consumed);
              }
            } else {
              *normalized += sp[n];  // NOLINT
              if (offset != nullptr) {
                offset->push_back(*consumed);
              }
            }
          }
        }
        is_prev_space = sp.ends_with(' ');
      }

      if (offset != nullptr) {
        *consumed += p.second;
      }
      input.remove_prefix(p.second);
      if (!remove_extra_whitespaces_) {
        is_prev_space = false;
      }
    }
  }

 private:
  // Normalizes the longest matching prefix of `input`.
  // Returns a pair {normalized_piece, consumed_bytes}.
  std::pair<std::string_view, int> NormalizePrefix(
      std::string_view input) const {
    std::pair<std::string_view, int> result;
    if (input.empty()) return result;

    size_t longest_length = 0;
    int longest_value = 0;

    if (trie_.has_array()) {
      std::tie(longest_value, longest_length) =
          trie_.longest_prefix_lookup(input);
    }

    if (longest_length == 0 || longest_length > input.size() ||
        static_cast<size_t>(longest_value) >= normalized_.size()) {
      // Fast-path for ASCII bytes without normalization rules avoids variable
      // initialization and function call overhead.
      if (static_cast<unsigned char>(input[0]) < 0x80) {
        result.second = 1;
        result.first = std::string_view(input.data(), 1);
      } else {
        size_t length = 0;
        if (!utf8::IsValidUTF8(input, &length)) {
          result.second = 1;
          result.first = kReplacementChar;
        } else {
          result.second = length;
          result.first = std::string_view(input.data(), result.second);
        }
      }
    } else {
      result.second = longest_length;
      result.first = std::string_view(normalized_.data() + longest_value);
    }

    return result;
  }

  const NormalizerSpec* spec_ = nullptr;
  DoubleArray trie_;
  std::string_view normalized_;

  // Caching these specification flags during initialization avoids repetitive
  // FlatBuffer vtable dereferences inside tight character normalization loops,
  // improving text normalization performance by ~3.5% to 6.4%.
  bool add_dummy_prefix_ = true;
  bool remove_extra_whitespaces_ = true;
  bool escape_whitespaces_ = true;
  bool treat_whitespace_as_suffix_ = false;
};
}  // namespace

class Model {
 public:
  explicit Model(std::string_view model_buffer);
  ~Model();

  StatusCode status() const { return status_; }

  void Encode(std::string_view normalized, std::vector<int>* ids,
              std::vector<std::string_view>* pieces = nullptr,
              float alpha = 0.0f,
              const FunctionRef<float()>* uniform_sampler = nullptr) const;

  int PieceToId(std::string_view piece) const;
  std::string_view IdToPiece(int id) const;
  float GetScore(int id) const { return scores_[id]; }
  int32_t GetIntScore(int id) const { return int_scores_[id]; }
  bool IsUnknown(int id) const { return types_[id] == PieceType_UNKNOWN; }
  bool IsControl(int id) const { return types_[id] == PieceType_CONTROL; }
  bool IsByte(int id) const { return types_[id] == PieceType_BYTE; }

  bool IsUserDefined(int id) const {
    return types_[id] == PieceType_USER_DEFINED;
  }

  // Returns true if `id` corresponds to a non-surface or special vocabulary
  // token (`UNKNOWN`, `CONTROL`, `UNUSED`, or `BYTE`).
  // This is used during Unigram lattice construction and BPE symbol merging to
  // filter out special tokens so regular text substrings cannot be encoded as
  // control tags or reserved IDs.
  bool IsInvisible(int id) const {
    if (id < 0) return true;
    const auto type = types_[id];
    // Fast-path inverted check: Over 99.9% of tokens in any vocabulary are
    // PieceType_NORMAL (1) or PieceType_USER_DEFINED (4). By checking against
    // these two surface types first, `type != PieceType_NORMAL` immediately
    // evaluates to false and short-circuits in 1 ALU instruction, avoiding
    // 4 sequential equality checks against special token types!
    return type != PieceType_NORMAL && type != PieceType_USER_DEFINED;
  }

  int piece_type(int id) const { return static_cast<int>(types_[id]); }
  int vocab_size() const { return model_proto_->pieces()->size(); }
  int model_type() const { return static_cast<int>(model_type_); }
  int unk_id() const { return unk_id_; }
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
  int pad_id() const { return pad_id_; }
  float unk_score() const { return unk_score_; }

  std::string_view unk_piece() const {
    return unk_id_ < 0 ? "<unk>" : IdToPiece(unk_id_);
  }
  std::string_view unk_surface() const;

  bool ByteFallbackEnabled() const { return byte_fallback_start_id_ != -1; }
  bool add_dummy_prefix() const { return add_dummy_prefix_; }

  bool has_normalizer_spec() const {
    return model_proto_->normalizer_spec() != nullptr;
  }

  bool treat_whitespace_as_suffix() const {
    return treat_whitespace_as_suffix_;
  }

  bool remove_extra_whitespaces() const {
    return normalizer_.remove_extra_whitespaces();
  }

  const ModelProto& model_proto() const { return *model_proto_; }

  bool has_direct_mappings() const {
    return model_proto_->has_direct_mappings();
  }

  // Returns true if `id` corresponds to a "direct mapping" token.
  // In BPE tokenization, a direct mapping indicates that a regular surface
  // token is mathematically guaranteed to encode as a single standalone token
  // whenever it appears as an isolated pre-split chunk. When true, the encoder
  // can bypass the iterative BPE symbol merge loop and emit the ID directly in
  // O(L) time.
  bool IsDirectMapping(int id) const {
    if (id >= vocab_size() || IsInvisible(id)) return false;
    const auto* vector = model_proto_->is_direct_mapping();
    if (vector == nullptr) {
      return model_proto_->has_direct_mappings();
    }
    return vector->Get(id) != 0;
  }

  bool has_non_null_vector() const {
    return model_proto_->is_direct_mapping() != nullptr;
  }
  void SetScoreResetThresholdForTesting(float threshold) {
    score_reset_threshold_ = threshold;
  }

  int ByteToId(unsigned char b) const { return byte_to_id_[b]; }

  int IdToByte(int id) const {
    if (byte_fallback_start_id_ == -1) return -1;
    if (id >= byte_fallback_start_id_ && id < byte_fallback_start_id_ + 256) {
      return id - byte_fallback_start_id_;
    }
    return -1;
  }

  StatusCode PretokenizeAtSafeBoundaries(
      std::string_view normalized,
      FunctionRef<void(std::string_view)> receiver) const;
  StatusCode Normalize(std::string_view input, std::string* normalized,
                       std::vector<size_t>* offset = nullptr) const;
  void EncodeChunk(std::string_view chunk, std::vector<int>* ids,
                   std::vector<std::string_view>* pieces = nullptr,
                   float alpha = 0.0f,
                   const FunctionRef<float()>* uniform_sampler = nullptr) const;

 private:
  void EncodeUnigram(
      std::string_view normalized, std::vector<int>* ids,
      std::vector<std::string_view>* pieces = nullptr, float alpha = 0.0f,
      const FunctionRef<float()>* uniform_sampler = nullptr) const;
  void EncodeBPE(std::string_view normalized, std::vector<int>* ids,
                 std::vector<std::string_view>* pieces = nullptr,
                 float alpha = 0.0f,
                 const FunctionRef<float()>* uniform_sampler = nullptr) const;
  template <bool kReverse = false>
  void EmitPiece(int id, std::string_view piece, std::vector<int>* ids,
                 std::vector<std::string_view>* pieces = nullptr) const {
    if (id == unk_id_ && byte_fallback_start_id_ != -1) {
      const size_t len = piece.size();
      for (size_t i = 0; i < len; ++i) {
        ids->push_back(ByteToId(kReverse ? piece[len - 1 - i] : piece[i]));
        if (pieces != nullptr) {
          pieces->push_back((i == len - 1) ? piece : piece.substr(0, 0));
        }
      }
      return;
    }

    if (id != unk_id_ || ids->empty() || ids->back() != unk_id_) {
      ids->push_back(id);
      if (pieces != nullptr) pieces->push_back(piece);
      return;
    }

    if (pieces != nullptr && !pieces->empty()) {
      auto& back = pieces->back();
      back = kReverse
                 ? std::string_view(piece.data(), piece.size() + back.size())
                 : std::string_view(back.data(), back.size() + piece.size());
    }
  }

  StatusCode Initialize(std::string_view model_buffer);

  Normalizer normalizer_;
  const ModelProto* model_proto_ = nullptr;
  StatusCode status_ = StatusCode::kFailedPrecondition;
  DoubleArray pieces_trie_;
  DoubleArray prefix_matcher_trie_;
  DoubleArray char_bigram_trie_;

  // Caching these flags avoids per-token FlatBuffer vtable pointer lookups
  // during decoding and normalization checks.
  int unk_id_ = 0;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int pad_id_ = -1;
  bool add_dummy_prefix_ = true;
  bool treat_whitespace_as_suffix_ = false;
  float unk_score_ = 0.0;
  ModelType model_type_ = ModelType_UNIGRAM;
  const float* scores_ = nullptr;
  const int32_t* int_scores_ = nullptr;
  const signed char* types_ = nullptr;

  // `byte_to_id_[x]` (which equals `byte_fallback_start_id_ + x`) stores the
  // actual vocabulary symbol ID for raw byte `x` (0 <= x < 256).
  // If byte fallback is not enabled or not found in the vocabulary,
  // `byte_fallback_start_id_` is set to -1.
  int byte_to_id_[256];
  // Pre-computed root-node Trie transition positions for ASCII bytes (0..255).
  uint32_t root_ascii_table_[256];
  int byte_fallback_start_id_ = -1;
  float score_reset_threshold_ = 100000.0f;
};

Model::Model(std::string_view model_buffer)
    : status_(StatusCode::kFailedPrecondition) {
  status_ = Initialize(model_buffer);
}

Model::~Model() = default;

StatusCode Model::Initialize(std::string_view model_buffer) {
  if (model_buffer.empty() ||
      model_buffer.size() >= static_cast<size_t>(FLATBUFFERS_MAX_BUFFER_SIZE) ||
      (reinterpret_cast<uintptr_t>(model_buffer.data()) %
       kModelBufferAlignment) != 0) {
    return StatusCode::kInvalidArgument;
  }

  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(model_buffer.data()),
      model_buffer.size());
  if (!VerifyModelProtoBuffer(verifier)) {
    return StatusCode::kInvalidArgument;
  }

  model_proto_ = GetModelProto(model_buffer.data());
  if (model_proto_ == nullptr || model_proto_->pieces() == nullptr) {
    return StatusCode::kInvalidArgument;
  }

  const size_t vocab_size = model_proto_->pieces()->size();

  unk_id_ = model_proto_->unk_id();
  if (unk_id_ < -1 || unk_id_ >= static_cast<int>(vocab_size)) {
    return StatusCode::kInternal;
  }

  bos_id_ = model_proto_->bos_id();
  if (bos_id_ < -1 || bos_id_ >= static_cast<int>(vocab_size)) {
    return StatusCode::kInternal;
  }

  eos_id_ = model_proto_->eos_id();
  if (eos_id_ < -1 || eos_id_ >= static_cast<int>(vocab_size)) {
    return StatusCode::kInternal;
  }

  pad_id_ = model_proto_->pad_id();
  if (pad_id_ < -1 || pad_id_ >= static_cast<int>(vocab_size)) {
    return StatusCode::kInternal;
  }

  if (has_normalizer_spec()) {
    add_dummy_prefix_ = model_proto_->normalizer_spec()->add_dummy_prefix();
    treat_whitespace_as_suffix_ =
        model_proto_->normalizer_spec()->treat_whitespace_as_suffix();
    LITE_RETURN_IF_ERROR(
        normalizer_.Initialize(model_proto_->normalizer_spec()));
  } else {
    add_dummy_prefix_ = true;
    treat_whitespace_as_suffix_ = false;
  }

  model_type_ = model_proto_->model_type();

  LITE_RETURN_IF_ERROR(InitializeTrie(
      model_proto_->pieces_trie_blob(), vocab_size,
      /*check_suffix_links=*/false, /*is_optional=*/false, &pieces_trie_));
  LITE_RETURN_IF_ERROR(
      InitializeTrie(model_proto_->prefix_matcher_trie_blob(), vocab_size,
                     /*check_suffix_links=*/false, /*is_optional=*/true,
                     &prefix_matcher_trie_));
  LITE_RETURN_IF_ERROR(InitializeTrie(
      model_proto_->char_bigram_trie_blob(), -1,
      /*check_suffix_links=*/true, /*is_optional=*/true, &char_bigram_trie_));

  if (char_bigram_trie_.has_array()) {
    const auto trie_view = char_bigram_trie_.view();
    for (int i = 0; i < 256; ++i) {
      const char c = static_cast<char>(i);
      root_ascii_table_[i] = trie_view.transition(0U, std::string_view(&c, 1));
    }
  } else {
    for (int i = 0; i < 256; ++i) {
      root_ascii_table_[i] = DoubleArray::View::kInvalidNodePos;
    }
  }

  if (model_type_ == ModelType_BPE) {
    if (model_proto_->int_scores() == nullptr ||
        model_proto_->int_scores()->size() != vocab_size) {
      return StatusCode::kInternal;
    }
  } else {
    if (model_proto_->scores() == nullptr ||
        model_proto_->scores()->size() != vocab_size) {
      return StatusCode::kInternal;
    }
  }

  if (model_type_ == ModelType_BPE) {
    // BPE models store merge priority scores as integers.
    int_scores_ = model_proto_->int_scores()->data();
  } else {
    // Unigram models store scores as floating-point log-probabilities.
    scores_ = model_proto_->scores()->data();
  }

  if (model_proto_->types() == nullptr ||
      model_proto_->types()->size() != vocab_size) {
    return StatusCode::kInternal;
  }
  types_ = model_proto_->types()->data();

  if (model_proto_->has_direct_mappings() &&
      model_proto_->is_direct_mapping() != nullptr &&
      model_proto_->is_direct_mapping()->size() != vocab_size) {
    return StatusCode::kInternal;
  }

  // Validate that the offline compiled Double-Array Trie is fully consistent
  // with the FlatBuffers `pieces` vector. Specifically, we verify:
  // 1. Every piece string is non-null and within valid bounds
  //    (`kMaxPieceLength`).
  // 2. Exact lookup of every piece string in `pieces_trie_` returns a valid ID.
  // 3. The returned trie lookup ID exactly matches the piece index `i`,
  //    guaranteeing bidirectional 1-to-1 mapping consistency.
  for (size_t i = 0; i < vocab_size; ++i) {
    const auto* piece_obj = model_proto_->pieces()->Get(i);
    if (piece_obj == nullptr || piece_obj->size() > kMaxPieceLength) {
      return StatusCode::kInternal;
    }
    const std::string_view piece(piece_obj->c_str(), piece_obj->size());
    // Verify exact lookup returns the expected vocabulary index i.
    const int id = pieces_trie_.exact_lookup(piece);
    if (id < 0 || id >= static_cast<int>(vocab_size) ||
        id != static_cast<int>(i)) {
      return StatusCode::kInternal;
    }
  }

  if (model_proto_->unk_surface() != nullptr &&
      model_proto_->unk_surface()->size() > kMaxPieceLength) {
    return StatusCode::kInternal;
  }

  // Calculate unk_score and verify against NaN / Inf score corruption.
  // Rejecting NaN or Inf during model initialization guarantees that Viterbi
  // DP operates strictly on valid real numbers, preventing silent arithmetic
  // corruption.
  float min_score = FLT_MAX;
  if (model_type_ != ModelType_BPE) {
    for (int i = 0; i < static_cast<int>(vocab_size); ++i) {
      float score = model_proto_->scores()->Get(i);
      if (std::isnan(score) || std::isinf(score)) {
        return StatusCode::kInternal;
      }
      const auto type = model_proto_->types()->Get(i);
      if (type == PieceType_DEFAULT || type == PieceType_NORMAL) {
        min_score = std::min(min_score, score);
      }
    }
  }
  unk_score_ = min_score - kUnkPenalty;

  // Initialize byte/ID caches.
  // We dynamically detect if byte fallback is supported by looking up the
  // first byte piece "<0x00>" in the vocabulary Trie. If present, we populate
  // the direct mapping cache and verify that all 256 byte pieces are
  // stored contiguously in the vocabulary.
  std::fill(std::begin(byte_to_id_), std::end(byte_to_id_), -1);
  const std::string first_byte_piece = ByteToPiece(0);
  const int first_byte_id = pieces_trie_.exact_lookup(first_byte_piece);

  if (first_byte_id >= 0 && first_byte_id < static_cast<int>(vocab_size)) {
    byte_fallback_start_id_ = first_byte_id;
    for (int i = 0; i < 256; ++i) {
      const std::string piece = ByteToPiece(i);
      const int id = pieces_trie_.exact_lookup(piece);
      if (id == -1 || id != byte_fallback_start_id_ + i) {
        return StatusCode::kInternal;
      }
      byte_to_id_[i] = id;
    }
  } else {
    byte_fallback_start_id_ = -1;
  }

  if (unk_id_ == -1 && byte_fallback_start_id_ == -1) {
    return StatusCode::kInternal;
  }

  return StatusCode::kOk;
}

std::string_view Model::unk_surface() const {
  if (model_proto_ == nullptr || model_proto_->unk_surface() == nullptr) {
    return " \xe2\x81\x87 ";
  }
  const auto* s = model_proto_->unk_surface();
  return std::string_view(s->c_str(), s->size());
}

int Model::PieceToId(std::string_view piece) const {
  return pieces_trie_.exact_lookup(piece);
}

std::string_view Model::IdToPiece(int id) const {
  assert(id >= 0 && id < vocab_size());
  const auto* s = model_proto_->pieces()->Get(id);
  return std::string_view(s->c_str(), s->size());
}

void Model::EncodeChunk(std::string_view chunk, std::vector<int>* ids,
                        std::vector<std::string_view>* pieces, float alpha,
                        const FunctionRef<float()>* uniform_sampler) const {
  if (model_type_ == ModelType_BPE) {
    // Shortcut: If the entire pre-split chunk is already a normal token in the
    // vocabulary and marked as a direct mapping, we can output its ID directly
    // and return immediately.
    // Because the chunk is isolated and BPE is a greedy merge process, this is
    // mathematically guaranteed to yield a theoretically identical result to
    // running the full BPE merge loop, but runs in O(L) trie lookup time.
    if (uniform_sampler == nullptr) {
      if (const int id = PieceToId(chunk); IsDirectMapping(id)) {
        ids->push_back(id);
        if (pieces != nullptr) {
          pieces->push_back(chunk);
        }
        return;
      }
    }
    EncodeBPE(chunk, ids, pieces, alpha, uniform_sampler);
  } else {
    EncodeUnigram(chunk, ids, pieces, alpha, uniform_sampler);
  }
}

void Model::Encode(std::string_view normalized, std::vector<int>* ids,
                   std::vector<std::string_view>* pieces, float alpha,
                   const FunctionRef<float()>* uniform_sampler) const {
  if (status_ != StatusCode::kOk || normalized.empty()) {
    return;
  }

  // DESIGN NOTE: We only apply strict character-bigram pre-tokenization for
  // BPE models.
  //
  // 1. BPE Greedy Merging: Using a heap/priority-queue, BPE's merge loop has a
  //    worst-case O(N log N) complexity on a contiguous string of length N.
  //    Pre-tokenization splits the input into small independent chunks at safe
  //    boundaries. Splitting a string of length N into k chunks of length N/k
  //    reduces the heap operations from O(N log N) to O(N log (N/k)). This
  //    substantially decreases constant-factor overhead, reduces memory
  //    footprint, and yields a practical 2x speedup.
  // 2. Unigram Viterbi DP: Unigram tokenization uses the Viterbi algorithm,
  //    which is mathematically O(N) where N is the string length. The constant
  //    overhead of pre-tokenization (allocating and traversing chunks)
  //    outweighs any DP search-space reduction, making it slower for Unigram.
  //
  // Note: The character bigram trie (`char_bigram_trie_`) is still compiled
  // and loaded unconditionally for all models (including Unigram) to maintain
  // flatbuffer schema uniformity and support potential future optimizations.
  if (model_type_ != ModelType_BPE || !char_bigram_trie_.has_array()) {
    EncodeChunk(normalized, ids, pieces, alpha, uniform_sampler);
    return;
  }

  PretokenizeAtSafeBoundaries(normalized, [&](std::string_view chunk) {
    if (!chunk.empty()) {
      EncodeChunk(chunk, ids, pieces, alpha, uniform_sampler);
    }
  });
}

StatusCode Model::Normalize(std::string_view input, std::string* normalized,
                            std::vector<size_t>* offset) const {
  if (status_ != StatusCode::kOk) return status_;
  if (!has_normalizer_spec()) {
    normalized->assign(input.data(), input.size());
    if (offset != nullptr) {
      // For identity normalization, byte positions map 1:1 (0, 1, ..., N-1),
      // followed by N as the sentinel.
      offset->resize(input.size() + 1);
      std::iota(offset->begin(), offset->end(), 0);
    }
    return StatusCode::kOk;
  }
  return normalizer_.Normalize(input, normalized, offset);
}

StatusCode Model::PretokenizeAtSafeBoundaries(
    std::string_view normalized,
    FunctionRef<void(std::string_view)> receiver) const {
  if (normalized.empty()) {
    return StatusCode::kOk;
  }
  if (!char_bigram_trie_.has_array()) {
    receiver(normalized);
    return StatusCode::kOk;
  }

  auto trie = char_bigram_trie_.view();

  auto get_root_node_pos = [this, &trie](std::string_view piece) -> uint32_t {
    if (piece.size() == 1) {
      return root_ascii_table_[static_cast<unsigned char>(piece[0])];
    }
    return trie.transition(0U, piece);
  };

  StringScanner scanner(normalized);
  const char* chunk_start = scanner.data();
  std::string_view prev_char = scanner.ConsumeNext();
  uint32_t prev_node_pos = get_root_node_pos(prev_char);

  while (!scanner.empty()) {
    if (scanner.remaining_view().size() >= 8) {
      // SWAR (SIMD Within A Register) 64-bit fast-forward for 8 ASCII letters.
      // Checks 8 contiguous bytes in a single word without data-dependent
      // branches, replacing the scalar loop:
      //   while (count < 8 && is_ascii_alpha(scanner.data()[count])) ++count;
      // In English documents (e.g., ice_long_doc.txt, 1.91 MB), this
      // accelerates pre-tokenization throughput by ~3.44x (52.5 MB/s -> 180.8
      // MB/s).
      const uint64_t val = LoadLe64(scanner.data());
      const uint64_t lower = val | 0x2020202020202020ULL;
      const uint64_t sub = lower - 0x6161616161616161ULL;
      const uint64_t bad_mask =
          (val | sub | (0x7A7A7A7A7A7A7A7AULL - lower)) & 0x8080808080808080ULL;
      const int count =
          (bad_mask == 0) ? 8
                          : static_cast<int>(FirstMatchedByteOffset(bad_mask));
      if (count > 0) {
        scanner.Advance(count);
        prev_node_pos =
            root_ascii_table_[static_cast<unsigned char>(scanner.data()[-1])];
        continue;
      }
    }

    const char* curr_pos = scanner.data();
    std::string_view curr_char = scanner.ConsumeNext();

    int next_node_pos = -1;
    if (prev_node_pos != DoubleArray::View::kInvalidNodePos) {
      uint32_t curr_node = trie.transition(prev_node_pos, curr_char);
      next_node_pos = trie.leaf_value(curr_node);
    }

    if (next_node_pos >= 0) {
      prev_node_pos = static_cast<uint32_t>(next_node_pos);
    } else {
      receiver(std::string_view(chunk_start, curr_pos - chunk_start));
      chunk_start = curr_pos;
      prev_node_pos = get_root_node_pos(curr_char);
    }
  }

  if (chunk_start < scanner.data()) {
    receiver(std::string_view(chunk_start, scanner.data() - chunk_start));
  }

  return StatusCode::kOk;
}

void Model::EncodeUnigram(std::string_view normalized, std::vector<int>* ids,
                          std::vector<std::string_view>* pieces, float alpha,
                          const FunctionRef<float()>* uniform_sampler) const {
  struct BestPathNode {
    int id = -1;
    float best_path_score = 0;
    int starts_at = -1;
  };

  const bool has_sampler = (uniform_sampler != nullptr);

  // Inline Gumbel-Max noise generator for stochastic Unigram sampling.
  // When alpha > 0.0f and a non-null sampler is provided, perturb scores with
  // Gumbel(0, 1) noise scaled by `alpha` temperature: s_new = s + alpha * G.
  // Zero overhead when alpha <= 0.0f or uniform_sampler is null (standard
  // Viterbi). Uses std::clamp to prevent log(0) domain errors while
  // maintaining exact Boltzmann distribution sampling dynamics.
  // See Gumbel-Max Trick / Gumbel-Softmax (Jang et al. 2016 / Maddison et al.
  // 2016) (https://arxiv.org/abs/1903.06059) for numerical stability and math.
  auto add_gumbel = [alpha, uniform_sampler, has_sampler](float s) {
    if (!has_sampler) return s;
    const float u = std::clamp((*uniform_sampler)(), 1e-6f, 1.0f - 1e-6f);
    return s + alpha * (-std::log(-std::log(u)));
  };

  const int size = normalized.size();
  const float unk_penalty_score = unk_score();
  std::vector<BestPathNode> best_path_ends_at(size + 1);
  int starts_at = 0;
  // Bidirectional threshold-triggered Viterbi score re-centering:
  // In IEEE 754 single-precision float (24-bit significand, ~7 decimal digits),
  // accumulating large log-probability or reward scores over long text causes
  // significand degradation (precision loss), eroding path tiebreaking.
  // When the accumulated score exceeds kScoreResetThreshold in either
  // direction (< -100000.0f or > +100000.0f), we subtract the current offset
  // from all active future path nodes, resetting the accumulator to 0.0f
  // without altering relative path differences or requiring prior validation.
  const float kScoreResetThreshold = score_reset_threshold_;
  int max_frontier = 0;
  StringScanner scanner(normalized);
  while (starts_at < size) {
    float best_path_score_till_here =
        best_path_ends_at[starts_at].best_path_score;
    if (best_path_score_till_here < -kScoreResetThreshold ||
        best_path_score_till_here > kScoreResetThreshold) {
      const float offset = best_path_score_till_here;
      for (int i = starts_at; i <= max_frontier; ++i) {
        if (i == starts_at || best_path_ends_at[i].starts_at != -1) {
          best_path_ends_at[i].best_path_score -= offset;
        }
      }
      best_path_score_till_here = 0.0f;
    }
    bool has_single_node = false;
    const int mblen = static_cast<int>(scanner.NextLen());
    pieces_trie_.common_prefix_lookup(
        scanner.remaining_view(), [&](int ret, size_t length) {
          if (IsInvisible(ret)) return;
          const int end_pos = starts_at + static_cast<int>(length);
          max_frontier = std::max(max_frontier, end_pos);
          auto& target_node = best_path_ends_at[end_pos];
          const auto score =
              IsUserDefined(ret) ? GetUserDefinedScore(length) : GetScore(ret);
          const auto candidate_best_path_score =
              add_gumbel(score) + best_path_score_till_here;
          if (target_node.starts_at == -1 ||
              candidate_best_path_score > target_node.best_path_score) {
            target_node.best_path_score = candidate_best_path_score;
            target_node.starts_at = starts_at;
            target_node.id = ret;
          }
          if (!has_single_node && length == static_cast<size_t>(mblen)) {
            has_single_node = true;
          }
        });
    if (!has_single_node) {
      const int end_pos = starts_at + mblen;
      max_frontier = std::max(max_frontier, end_pos);
      auto& target_node = best_path_ends_at[end_pos];
      const auto candidate_best_path_score =
          add_gumbel(unk_penalty_score) + best_path_score_till_here;
      if (target_node.starts_at == -1 ||
          candidate_best_path_score > target_node.best_path_score) {
        target_node.best_path_score = candidate_best_path_score;
        target_node.starts_at = starts_at;
        target_node.id = unk_id_;
      }
    }
    starts_at += mblen;
    scanner.Advance(mblen);
  }

  const size_t start_ids_index = ids->size();
  const size_t start_pieces_index = pieces != nullptr ? pieces->size() : 0;
  int ends_at = size;
  while (ends_at > 0) {
    const auto& node = best_path_ends_at[ends_at];
    EmitPiece<true>(node.id,
                    normalized.substr(node.starts_at, ends_at - node.starts_at),
                    ids, pieces);
    ends_at = node.starts_at;
  }
  std::reverse(ids->begin() + start_ids_index, ids->end());
  if (pieces != nullptr) {
    std::reverse(pieces->begin() + start_pieces_index, pieces->end());
  }
}

namespace {

struct SymbolPair {
  int32_t int_score;  // BPE rank score. large is better.
  uint32_t left;      // left index of this pair
  int right;          // right index of this pair
  unsigned int size;  // length of this piece
  int id;             // vocabulary token ID of this pair
  // Cached Trie node position for the merged symbol piece.
  uint32_t trie_node_pos = DoubleArray::View::kInvalidNodePos;
};

struct SymbolPairComparator {
  bool operator()(const SymbolPair& h1, const SymbolPair& h2) const {
    if (h1.int_score != h2.int_score) {
      return h1.int_score < h2.int_score;
    }
    return h1.left > h2.left;
  }
};

struct Symbol {
  std::string_view piece;
  int prev;             // prev index of this symbol. -1 for BOS.
  int next;             // next index of this symbol. -1 for EOS.
  int32_t id : 31;      // vocab id of this symbol.
  uint32_t freeze : 1;  // Use 32-bit integer type so MSVC packs with `id`.
  // Cached Trie node position for `piece`.
  // Bypasses root-node re-lookups when merging adjacent symbols,
  // enabling direct 1-step Trie transitions from the left symbol's node.
  uint32_t trie_node_pos = DoubleArray::View::kInvalidNodePos;
};

static_assert(sizeof(SymbolPair) <= 24,
              "SymbolPair size should not exceed 24 bytes without padding.");
static_assert(sizeof(Symbol) <= 32, "Symbol size should not exceed 32 bytes.");
}  // namespace

void Model::EncodeBPE(std::string_view normalized, std::vector<int>* ids,
                      std::vector<std::string_view>* pieces, float alpha,
                      const FunctionRef<float()>* uniform_sampler) const {
  if (normalized.empty()) {
    return;
  }

  InlineVector<Symbol, 64> symbols;
  symbols.reserve(normalized.size());

  const auto trie_view = pieces_trie_.view();

  StringScanner scanner(normalized);
  int index = 0;
  while (!scanner.empty()) {
    Symbol s;
    size_t mblen = 0;
    int matched_id = -1;
    if (prefix_matcher_trie_.has_array()) {
      std::tie(matched_id, mblen) =
          prefix_matcher_trie_.longest_prefix_lookup(scanner.remaining_view());
    }
    if (mblen > 0 && IsUserDefined(matched_id)) {
      s.piece = std::string_view(scanner.data(), mblen);
      s.id = matched_id;
      s.freeze = true;
      s.trie_node_pos = DoubleArray::View::kInvalidNodePos;
    } else {
      mblen = scanner.NextLen();
      s.piece = std::string_view(scanner.data(), mblen);
      s.trie_node_pos = trie_view.transition(0U, s.piece);
      const int id = trie_view.leaf_value(s.trie_node_pos);
      s.id = id == -1 ? unk_id_ : id;
      s.freeze = false;
    }
    s.prev = index == 0 ? -1 : index - 1;
    scanner.Advance(mblen);
    s.next = scanner.empty() ? -1 : index + 1;
    ++index;
    symbols.emplace_back(s);
  }

  if (symbols.empty()) {
    return;
  }

  // BPE-Dropout (Provilkov et al., ACL 2020): `alpha` is the probability (p in
  // [0, 1]) of dropping a merge. When alpha >= 1.0f, 100% of merges are
  // skipped, so we bypass agenda creation and directly output initial symbols.
  if (alpha < 1.0f) {
    InlineVector<SymbolPair, 64> agenda_vec;
    agenda_vec.reserve(symbols.size());

    if (symbols.size() > 1) {
      for (size_t left = 0; left < symbols.size() - 1; ++left) {
        const size_t right = left + 1;
        if (symbols[left].freeze || symbols[right].freeze) continue;
        const uint32_t node_pos = trie_view.transition(
            symbols[left].trie_node_pos, symbols[right].piece);
        if (node_pos == DoubleArray::View::kInvalidNodePos) continue;
        const int id = trie_view.leaf_value(node_pos);
        if (id == -1 || IsInvisible(id)) continue;
        SymbolPair& h = agenda_vec.emplace_back();
        h.left = left;
        h.right = right;
        h.int_score = GetIntScore(id);
        h.size = symbols[left].piece.size() + symbols[right].piece.size();
        h.id = id;
        h.trie_node_pos = node_pos;
      }
    }

    if (agenda_vec.size() > 1) {
      std::make_heap(agenda_vec.begin(), agenda_vec.end(),  // NOLINT
                     SymbolPairComparator());
    }

    auto MaybeAddNewSymbolPair = [this, &trie_view, &symbols, &agenda_vec](
                                     int left, int right) {
      if (left == -1 || right == -1) return;
      const Symbol& left_symbol = symbols[left];
      const Symbol& right_symbol = symbols[right];
      if (left_symbol.freeze || right_symbol.freeze) return;
      const uint32_t node_pos =
          trie_view.transition(left_symbol.trie_node_pos, right_symbol.piece);
      if (node_pos == DoubleArray::View::kInvalidNodePos) return;
      const int id = trie_view.leaf_value(node_pos);
      if (id == -1 || IsInvisible(id)) return;
      SymbolPair h;
      h.left = left;
      h.right = right;
      h.int_score = GetIntScore(id);
      h.size = left_symbol.piece.size() + right_symbol.piece.size();
      h.id = id;
      h.trie_node_pos = node_pos;
      agenda_vec.push_back(h);
      std::push_heap(agenda_vec.begin(), agenda_vec.end(),  // NOLINT
                     SymbolPairComparator());
    };

    const bool has_sampler = (uniform_sampler != nullptr);

    while (!agenda_vec.empty()) {
      std::pop_heap(agenda_vec.begin(), agenda_vec.end(),  // NOLINT
                    SymbolPairComparator());
      SymbolPair top = agenda_vec.back();
      agenda_vec.pop_back();

      // O(1) lazy invalidation check: if the sum of piece sizes no longer
      // matches `top.size`, one of the constituent symbols was modified by a
      // prior merge step, so we discard this stale entry.
      if (symbols[top.left].piece.empty() || symbols[top.right].piece.empty() ||
          symbols[top.left].piece.size() + symbols[top.right].piece.size() !=
              top.size) {
        continue;
      }

      if (has_sampler && (*uniform_sampler)() < alpha) {
        continue;
      }

      const std::string_view piece(
          symbols[top.left].piece.data(),
          symbols[top.left].piece.size() + symbols[top.right].piece.size());
      symbols[top.left].piece = piece;
      symbols[top.left].id = top.id;
      symbols[top.left].trie_node_pos = top.trie_node_pos;
      symbols[top.right].piece = std::string_view();

      symbols[top.left].next = symbols[top.right].next;
      if (symbols[top.right].next != -1) {
        symbols[symbols[top.right].next].prev = top.left;
      }

      MaybeAddNewSymbolPair(symbols[top.left].prev, top.left);
      MaybeAddNewSymbolPair(top.left, symbols[top.left].next);
    }
  }

  int curr = 0;
  while (curr != -1) {
    if (!symbols[curr].piece.empty()) {
      EmitPiece(symbols[curr].id, symbols[curr].piece, ids, pieces);
    }
    curr = symbols[curr].next;
  }
}

SentencePieceLiteProcessor::SentencePieceLiteProcessor(
    std::string_view buffer) {
  model_ = std::make_unique<Model>(buffer);
  status_ = model_->status();
  if (status_ != StatusCode::kOk) {
    model_.reset();
  }
}

SentencePieceLiteProcessor::SentencePieceLiteProcessor(
    std::shared_ptr<std::string> shared_buffer)
    : shared_buffer_(std::move(shared_buffer)) {
  if (shared_buffer_ == nullptr) {
    status_ = StatusCode::kInvalidArgument;
    return;
  }
  model_ = std::make_unique<Model>(*shared_buffer_);
  status_ = model_->status();
  if (status_ != StatusCode::kOk) {
    model_.reset();
  }
}

SentencePieceLiteProcessor::~SentencePieceLiteProcessor() = default;

StatusCode SentencePieceLiteProcessor::status() const { return status_; }

StatusCode SentencePieceLiteProcessor::Normalize(
    std::string_view input, std::string* output,
    std::vector<size_t>* offset) const {
  if (model_ == nullptr) return StatusCode::kFailedPrecondition;
  if (output == nullptr || input.size() > kMaxInputLength) {
    return StatusCode::kInvalidArgument;
  }
  return model_->Normalize(input, output, offset);
}

StatusCode SentencePieceLiteProcessor::Encode(std::string_view input,
                                              std::vector<int>* ids) const {
  std::string normalized;
  LITE_RETURN_IF_ERROR(Normalize(input, &normalized));
  return EncodeNormalized(normalized, ids);
}

namespace {
template <typename EncodeFunc>
StatusCode EncodeNormalizedHelper(const Model* model,
                                  std::string_view normalized,
                                  std::vector<int>* ids,
                                  std::vector<std::string_view>* pieces,
                                  EncodeFunc&& encode_func) {
  if (model == nullptr) {
    return StatusCode::kFailedPrecondition;
  }
  if (normalized.size() > kMaxInputLength || ids == nullptr) {
    return StatusCode::kInvalidArgument;
  }
  ids->clear();
  if (pieces != nullptr) pieces->clear();
  if (normalized.empty()) return StatusCode::kOk;

  encode_func();
  return StatusCode::kOk;
}
}  // namespace

StatusCode SentencePieceLiteProcessor::EncodeNormalized(
    std::string_view normalized, std::vector<int>* ids,
    std::vector<std::string_view>* pieces) const {
  return EncodeNormalizedHelper(model_.get(), normalized, ids, pieces, [&] {
    model_->Encode(normalized, ids, pieces);
  });
}

StatusCode SentencePieceLiteProcessor::EncodeNormalizedChunk(
    std::string_view normalized_chunk, std::vector<int>* ids,
    std::vector<std::string_view>* pieces) const {
  return EncodeNormalizedHelper(
      model_.get(), normalized_chunk, ids, pieces,
      [&] { model_->EncodeChunk(normalized_chunk, ids, pieces); });
}

StatusCode SentencePieceLiteProcessor::SampleNormalized(
    std::string_view normalized, float alpha,
    FunctionRef<float()> uniform_sampler, std::vector<int>* ids,
    std::vector<std::string_view>* pieces) const {
  if (alpha < 0.0f || !std::isfinite(alpha)) {
    return StatusCode::kInvalidArgument;
  }
  return EncodeNormalizedHelper(model_.get(), normalized, ids, pieces, [&] {
    model_->Encode(normalized, ids, pieces, alpha, &uniform_sampler);
  });
}

StatusCode SentencePieceLiteProcessor::PretokenizeAtSafeBoundaries(
    std::string_view normalized,
    FunctionRef<void(std::string_view)> receiver) const {
  if (model_ == nullptr) {
    return StatusCode::kFailedPrecondition;
  }
  return model_->PretokenizeAtSafeBoundaries(normalized, receiver);
}

StatusCode SentencePieceLiteProcessor::Decode(
    Span<const int> ids, std::string* output,
    std::vector<std::string_view>* pieces) const {
  if (model_ == nullptr) {
    return StatusCode::kFailedPrecondition;
  }
  if (output == nullptr || ids.size() > kMaxInputLength) {
    return StatusCode::kInvalidArgument;
  }
  output->clear();
  if (pieces != nullptr) {
    pieces->clear();
    pieces->resize(ids.size(), std::string_view());
  }

  // Store temporary byte offsets in pieces->data() via reinterpret_cast to
  // avoid heap allocations and resolve them to string_views after decoding.
  // +1/-1 offset prevents 0x0 nullptr assertions in libc++ std::string_view
  // hardening.
  auto set_offset = [&](size_t idx, size_t begin, size_t end) {
    if (pieces != nullptr) {
      (*pieces)[idx] = std::string_view(
          reinterpret_cast<const char*>(static_cast<uintptr_t>(begin + 1)),
          end - begin);
    }
  };

  std::vector<unsigned char> accumulated_bytes;
  size_t first_byte_index = 0;

  auto flush_bytes = [&]() -> StatusCode {
    if (accumulated_bytes.empty()) return StatusCode::kOk;
    size_t offset = 0;
    while (offset < accumulated_bytes.size()) {
      std::string_view bytes_view(
          reinterpret_cast<const char*>(accumulated_bytes.data() + offset),
          accumulated_bytes.size() - offset);
      size_t consumed = 0;
      if (pieces == nullptr) {
        consumed = utf8::SpanStructurallyValid(bytes_view);
        if (consumed > 0) {
          output->append(bytes_view.data(), consumed);
        } else {
          output->append(kReplacementChar);
          consumed = 1;
        }
      } else {
        const bool is_valid = utf8::IsValidUTF8(bytes_view, &consumed);
        const size_t char_begin = output->size();
        if (!is_valid) {
          output->append(kReplacementChar);
        } else {
          output->append(bytes_view.data(), consumed);
        }
        const size_t char_end = output->size();
        for (size_t j = 0; j < consumed; ++j) {
          set_offset(first_byte_index + offset + j, char_begin,
                     (j == consumed - 1) ? char_end : char_begin);
        }
      }
      offset += consumed;
    }
    accumulated_bytes.clear();
    return StatusCode::kOk;
  };

  bool is_bos_ws = true;
  bool bos_ws_seen = false;
  const bool add_dummy_prefix = model_->add_dummy_prefix();
  const bool remove_extra_whitespaces = model_->remove_extra_whitespaces();

  const int vocab_size = model_->vocab_size();
  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    if (id < 0 || id >= vocab_size) {
      return StatusCode::kInvalidArgument;
    }
    if (model_->IsUnknown(id)) {
      LITE_RETURN_IF_ERROR(flush_bytes());
      const size_t begin_pos = output->size();
      output->append(model_->unk_surface());
      set_offset(i, begin_pos, output->size());
      is_bos_ws = false;
      continue;
    }

    const bool is_byte = model_->IsByte(id);

    if (is_byte) {
      const int byte = model_->IdToByte(id);
      if (byte >= 0) {
        if (accumulated_bytes.empty()) {
          first_byte_index = i;
        }
        accumulated_bytes.push_back(static_cast<unsigned char>(byte));
      }
    } else {
      LITE_RETURN_IF_ERROR(flush_bytes());

      if (model_->IsControl(id)) {
        set_offset(i, output->size(), output->size());
        continue;
      }

      // Match canonical: clear is_bos_ws BEFORE stripping, based on whether
      // we've already seen a bos_ws or any non-empty output.
      if (bos_ws_seen || !output->empty()) {
        is_bos_ws = false;
      }

      const std::string_view piece = model_->IdToPiece(id);
      std::string_view piece_view = piece;
      if (is_bos_ws && (add_dummy_prefix || remove_extra_whitespaces)) {
        const bool consumed = piece_view.starts_with(kSpaceSymbol);
        if (consumed) {
          piece_view.remove_prefix(kSpaceSymbol.size());
        }
        // Match canonical: if remove_extra_whitespaces, don't set bos_ws_seen
        // so is_bos_ws stays true for the next token (keeps stripping).
        // Otherwise set bos_ws_seen so is_bos_ws gets cleared next iteration.
        if (consumed && !remove_extra_whitespaces) {
          bos_ws_seen = true;
        }
      }

      const size_t begin_pos = output->size();
      ReplaceAll(piece_view, kSpaceSymbol, " ", output);
      set_offset(i, begin_pos, output->size());
    }
  }

  LITE_RETURN_IF_ERROR(flush_bytes());

  if (pieces != nullptr) {
    // TRICKY: In-place convert stored integer byte offsets to valid memory
    // addresses by subtracting the +1 dummy offset and adding base_data.
    const char* base_data = output->data();
    for (auto& piece : *pieces) {
      const size_t begin_offset =
          static_cast<size_t>(reinterpret_cast<uintptr_t>(piece.data())) - 1;
      piece = std::string_view(base_data + begin_offset, piece.size());
    }
  }

  return StatusCode::kOk;
}

size_t SentencePieceLiteProcessor::vocab_size() const {
  if (model_ == nullptr) return 0;
  return model_->vocab_size();
}

int SentencePieceLiteProcessor::PieceToId(std::string_view piece) const {
  if (model_ == nullptr) return -1;
  return model_->PieceToId(piece);
}

std::string_view SentencePieceLiteProcessor::IdToPiece(int id) const {
  if (model_ == nullptr) return "<unk>";
  if (id < 0 || id >= static_cast<int>(model_->vocab_size())) {
    return model_->unk_piece();
  }
  return model_->IdToPiece(id);
}

float SentencePieceLiteProcessor::GetScore(int id) const {
  if (model_ == nullptr) return 0.0f;
  if (id < 0 || id >= static_cast<int>(model_->vocab_size())) {
    return 0.0f;
  }
  if (model_->model_type() == ModelType_BPE) {  // BPE
    return static_cast<float>(model_->GetIntScore(id));
  }
  return model_->GetScore(id);
}

int SentencePieceLiteProcessor::unk_id() const {
  if (model_ == nullptr) return -1;
  return model_->unk_id();
}

int SentencePieceLiteProcessor::bos_id() const {
  if (model_ == nullptr) return -1;
  return model_->bos_id();
}

int SentencePieceLiteProcessor::eos_id() const {
  if (model_ == nullptr) return -1;
  return model_->eos_id();
}

int SentencePieceLiteProcessor::pad_id() const {
  if (model_ == nullptr) return -1;
  return model_->pad_id();
}

int SentencePieceLiteProcessor::piece_type(int id) const {
  if (model_ == nullptr) return -1;
  if (id < 0 || id >= static_cast<int>(model_->vocab_size())) {
    return -1;
  }
  return model_->piece_type(id);
}

bool SentencePieceLiteProcessor::HasNonNullDirectMappingVectorForTesting()
    const {
  return model_ != nullptr && model_->has_non_null_vector();
}

void SentencePieceLiteProcessor::SetScoreResetThresholdForTesting(
    float threshold) {
  if (model_) {
    model_->SetScoreResetThresholdForTesting(threshold);
  }
}

}  // namespace sentencepiece::lite
