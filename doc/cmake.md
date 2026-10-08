# Building SentencePiece with CMake

This guide describes how to build, test, and install SentencePiece from source using [CMake](https://cmake.org/), as well as how to integrate SentencePiece into your own CMake project.

For Bazel instructions, see [Building with Bazel](bazel.md). For C++ API usage, see the [C++ API Reference](cpp.md).

---

## 1. Prerequisites

The following tools and libraries are required to build SentencePiece:

- **CMake** (3.14 or later; 3.15+ recommended for `FetchContent`)
- **C++20 compatible compiler** (e.g., GCC 11+, Clang 13+, or MSVC 2019+ with C++20 support)
- **gperftools** library (optional; provides a 10–40% runtime performance improvement via TCMalloc)

On Ubuntu/Debian-based systems, install the prerequisites using `apt-get`:
```bash
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
```

---

## 2. Building and Installing from Source

Clone the repository, configure the build with CMake, compile, and install:

```bash
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
sudo cmake --build build --target install
sudo ldconfig -v
```

*Note for macOS:* Replace `sudo ldconfig -v` with:
```bash
sudo update_dyld_shared_cache
```

### CMake Build Options

You can customize the build by passing `-D<OPTION>=<VALUE>` to `cmake -B build`:

| Option | Default | Description |
| :--- | :--- | :--- |
| `SPM_ENABLE_SHARED` | `ON` (`OFF` on Windows) | Build shared libraries (`libsentencepiece.so` / `.dylib`) in addition to static libraries. |
| `SPM_BUILD_TEST` | `OFF` | Build unit test binaries. |
| `SPM_ENABLE_BENCHMARK` | `OFF` | Build Google Benchmark suites. |
| `SPM_ENABLE_TCMALLOC` | `ON` | Link with TCMalloc (`gperftools`) if available. |
| `SPM_TCMALLOC_STATIC` | `OFF` | Link static library of TCMalloc. |
| `SPM_ENABLE_LITE` | `ON` | Build SentencePiece Lite targets (`sentencepiece_lite`, `spm_to_fb`, `spm_lite`). |
| `SPM_ABSL_PROVIDER` | `module` | Provider for Abseil (`module` fetches via CMake; `package` uses `find_package(absl)`). |
| `SPM_PROTOBUF_PROVIDER` | `module` | Provider for Protobuf (`module` fetches via CMake; `package` uses `find_package(Protobuf)`). |
| `SPM_ENABLE_NFKC_COMPILE` | `OFF` | Enable `compile_charsmap` compilation using system ICU. |
| `SPM_DISABLE_EMBEDDED_DATA` | `OFF` | Disable embedding pre-compiled normalization data into the binary. |
| `SPM_ENABLE_MSVC_MT_BUILD` | `OFF` | Use static MSVC runtime (`/MT`) on Windows. |

### Running Unit Tests

Enable `SPM_BUILD_TEST=ON` and run tests via `ctest`:

```bash
cmake -B build -DSPM_BUILD_TEST=ON
cmake --build build --config Release --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
ctest --test-dir build --output-on-failure
```

### Installing via vcpkg

You can also download and install SentencePiece using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install sentencepiece
```

---

## 3. Integrating SentencePiece into Your CMake Project

SentencePiece exports official CMake targets (`sentencepiece::sentencepiece` and `sentencepiece::sentencepiece_train`, as well as static variants `sentencepiece::sentencepiece-static` and `sentencepiece::sentencepiece_train-static`). Header search paths, transitive dependencies, and the required C++20 standard are automatically propagated to your targets.

### Option A: Using `find_package` (Installed Package)

When SentencePiece is installed on your system or in a custom prefix:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_project CXX)

# Find installed sentencepiece package
find_package(sentencepiece CONFIG REQUIRED)

add_executable(my_app main.cc)

# Link inference library
target_link_libraries(my_app PRIVATE sentencepiece::sentencepiece)

# Or link trainer library (if using SentencePieceTrainer)
# target_link_libraries(my_app PRIVATE sentencepiece::sentencepiece_train)
```

If SentencePiece is installed in a custom directory, pass `-DCMAKE_PREFIX_PATH`:
```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/sentencepiece_install
```

### Option B: Using `FetchContent` (Direct from Git)

To build SentencePiece automatically as part of your project without manual installation:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_project CXX)

include(FetchContent)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.2.3
)
FetchContent_MakeAvailable(sentencepiece)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE sentencepiece::sentencepiece)
```

### Option C: Using `add_subdirectory`

If you include SentencePiece as a Git submodule or vendored directory (e.g., `third_party/sentencepiece`):

```cmake
add_subdirectory(third_party/sentencepiece)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE sentencepiece::sentencepiece)
```
