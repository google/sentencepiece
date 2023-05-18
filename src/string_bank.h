#pragma once
#include <mutex>
#include <string>

#include "common.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/strings/string_view.h"

namespace sentencepiece {
namespace string_util {
class StringBank {
 public:
  StringBank() : hits_(0) {}
  StringBank(const StringBank &) = delete;
  StringBank& operator=(const StringBank &) = delete;
  virtual ~StringBank() {}

  template <typename T>
  inline absl::string_view View(const T& str) {
    const std::lock_guard lock(lock_);
    auto ep = bank_.emplace(str);
    hits_ += !ep.second;
    return *ep.first;
  }

  inline void Clear() { bank_.clear(); }

  inline uint64_t hits() const noexcept { return hits_; }

  inline size_t size() const { return bank_.size(); }

  inline size_t TotalSize() const {
    size_t size = 0;
    for (auto &s : bank_) {
      size += s.size();
    }
    return size;
  }

 private:
  absl::flat_hash_set<std::string> bank_;
  std::mutex lock_;
  uint64_t hits_;
};

}
}