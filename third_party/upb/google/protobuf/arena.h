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
// limitations under the License.

#ifndef GOOGLE_PROTOBUF_ARENA_H_
#define GOOGLE_PROTOBUF_ARENA_H_

#include <vector>
#include <functional>
#include <utility>

namespace google {
namespace protobuf {

class Arena {
 public:
  Arena() {}
  ~Arena() {
    for (auto& cleanup : cleanups_) {
      cleanup();
    }
  }
  
  template <typename T, typename... Args>
  static T* Create(Arena* arena, Args&&... args) {
    if (arena) {
      T* ptr = new T(std::forward<Args>(args)...);
      arena->cleanups_.push_back([ptr]() { delete ptr; });
      return ptr;
    } else {
      return new T(std::forward<Args>(args)...);
    }
  }
  
  template <typename T, typename... Args>
  static T* CreateMaybeMessage(Arena* arena, Args&&... args) {
    return Create<T>(arena, std::forward<Args>(args)...);
  }
 private:
  std::vector<std::function<void()>> cleanups_;
};

} // namespace protobuf
} // namespace google

#endif // GOOGLE_PROTOBUF_ARENA_H_
