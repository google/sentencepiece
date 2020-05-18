//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: string_view.h
// -----------------------------------------------------------------------------
//
// This file contains the definition of the `absl::string_view` class. A
// `string_view` points to a contiguous span of characters, often part or all of
// another `std::string`, double-quoted std::string literal, character array, or
// even another `string_view`.
//
// This `absl::string_view` abstraction is designed to be a drop-in
// replacement for the C++17 `std::string_view` abstraction.
#ifndef ABSL_MEMORY_MEMORY_H_
#define ABSL_MEMORY_MEMORY_H_

#include <memory>

namespace absl {

// Trait to select overloads and return types for MakeUnique.
template <typename T>
struct MakeUniqueResult {
  using scalar = std::unique_ptr<T>;
};
template <typename T>
struct MakeUniqueResult<T[]> {
  using array = std::unique_ptr<T[]>;
};
template <typename T, size_t N>
struct MakeUniqueResult<T[N]> {
  using invalid = void;
};

// MakeUnique<T>(...) is an early implementation of C++14 std::make_unique.
// It is designed to be 100% compatible with std::make_unique so that the
// eventual switchover will be a simple renaming operation.
template <typename T, typename... Args>
typename MakeUniqueResult<T>::scalar make_unique(Args &&... args) {  // NOLINT
  return std::unique_ptr<T>(
      new T(std::forward<Args>(args)...));  // NOLINT(build/c++11)
}

// Overload for array of unknown bound.
// The allocation of arrays needs to use the array form of new,
// and cannot take element constructor arguments.
template <typename T>
typename MakeUniqueResult<T>::array make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Reject arrays of known bound.
template <typename T, typename... Args>
typename MakeUniqueResult<T>::invalid make_unique(Args &&... /* args */) =
    delete;  // NOLINT

}  // namespace absl
#endif  // ABSL_MEMORY_MEMORY_H_
