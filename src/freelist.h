// Copyright 2018 Google Inc.
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

#ifndef FREELIST_H_
#define FREELIST_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace sentencepiece {
namespace model {

// Simple FreeList that allocates a chunk of T at once.
template <class T>
class FreeList {
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable.");
  static_assert(std::is_standard_layout_v<T>,
                "T must be a standard layout type.");

 public:
  FreeList() = delete;
  explicit FreeList(size_t chunk_size) : chunk_size_(chunk_size) {}
  virtual ~FreeList() = default;

  FreeList(const FreeList&) = delete;
  FreeList& operator=(const FreeList&) = delete;
  FreeList(FreeList&& other) noexcept = default;
  FreeList& operator=(FreeList&& other) noexcept = default;

  // `Free` doesn't free the object but reuse the allocated memory chunks.
  void Free() {
    for (auto& chunk : freelist_) {
      std::memset(static_cast<void*>(chunk.get()), 0, sizeof(T) * chunk_size_);
    }
    chunk_index_ = 0;
    element_index_ = 0;
  }

  // Returns the number of allocated elements.
  size_t size() const { return chunk_size_ * chunk_index_ + element_index_; }

  // Allocates new element.
  T* Allocate() {
    if (element_index_ >= chunk_size_) {
      ++chunk_index_;
      element_index_ = 0;
    }

    if (chunk_index_ == freelist_.size()) {
      auto chunk = std::make_unique<T[]>(chunk_size_);
      std::memset(static_cast<void*>(chunk.get()), 0, sizeof(T) * chunk_size_);
      freelist_.push_back(std::move(chunk));
    }

    T* result = freelist_[chunk_index_].get() + element_index_;
    ++element_index_;

    return result;
  }

 private:
  std::vector<std::unique_ptr<T[]>> freelist_;

  // The last element is stored at freelist_[chunk_index_][element_index_]
  size_t element_index_ = 0;
  size_t chunk_index_ = 0;
  size_t chunk_size_ = 0;  // Do not modify except in swap()
};
}  // namespace model
}  // namespace sentencepiece
#endif  // FREELIST_H_
