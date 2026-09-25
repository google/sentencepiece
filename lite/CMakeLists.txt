# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Root directory of SentencePiece repository relative to lite
get_filename_component(SPM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Standalone build check when lite is configured independently
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(sentencepiece_lite VERSION 1.0.0 LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(SENTENCEPIECE_LITE_STANDALONE_BUILD ON)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
  endif()
endif()

if (MSVC)
  add_definitions("/wd4018 /wd4514")
else()
  add_definitions("-Wno-sign-compare -Wno-deprecated-declarations")
endif()

option(SPM_LITE_ENABLE_UTF8_RANGE "Enable utf8_range SIMD UTF-8 validation in SentencePiece Lite" ON)
option(SPM_LITE_ENABLE_ABSL "Enable Abseil in SentencePiece Lite" ON)

# ------------------------------------------------------------------------------
# FlatBuffers Dependency Management via FetchContent
# Downloads specified release tag dynamically if not already present.
# HEAD is intentionally avoided to ensure build reproducibility.
# ------------------------------------------------------------------------------
if (NOT TARGET flatbuffers OR NOT TARGET flatc)
  include(FetchContent)
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.12.19-2026-02-06-03fffb2
    GIT_SHALLOW    TRUE
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/third_party/flatbuffers-src"
  )
  FetchContent_MakeAvailable(flatbuffers)
  FetchContent_GetProperties(flatbuffers SOURCE_DIR flatbuffers_SOURCE_DIR)
  if (flatbuffers_SOURCE_DIR)
    file(REMOVE_RECURSE "${flatbuffers_SOURCE_DIR}/tests/fuzzer")
  endif()
endif()

# FlatBuffers Schema Compilation (flatc -> sentencepiece_lite_generated.h)
set(LITE_FBS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_lite.fbs")
set(LITE_FBS_GEN_HDR "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_lite_generated.h")

if (SPM_FLATC_EXECUTABLE)
  set(FLATC_CMD "${SPM_FLATC_EXECUTABLE}")
  set(FLATC_DEP "")
elseif (CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
  set(FLATC_CMD ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:flatc>)
  set(FLATC_DEP flatc)
else()
  set(FLATC_CMD $<TARGET_FILE:flatc>)
  set(FLATC_DEP flatc)
endif()

add_custom_command(
  OUTPUT "${LITE_FBS_GEN_HDR}"
  COMMAND ${FLATC_CMD} -c --gen-object-api -o "${CMAKE_CURRENT_BINARY_DIR}" "${LITE_FBS_SRC}"
  DEPENDS "${LITE_FBS_SRC}" ${FLATC_DEP}
  COMMENT "Generating FlatBuffers C++ header sentencepiece_lite_generated.h"
  VERBATIM
)

add_custom_target(generate_sentencepiece_lite_fbs ALL DEPENDS "${LITE_FBS_GEN_HDR}")

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${SPM_ROOT_DIR}
  ${SPM_ROOT_DIR}/src
  ${SPM_ROOT_DIR}/third_party
  ${SPM_ROOT_DIR}/third_party/darts_clone
  ${SPM_ROOT_DIR}/third_party/rapidhash
)

if (SENTENCEPIECE_LITE_STANDALONE_BUILD)
  if (NOT TARGET absl::status)
    include(FetchContent)
    FetchContent_Declare(
      abseil-cpp
      GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
      GIT_TAG 20260817.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(abseil-cpp)
    FetchContent_GetProperties(abseil-cpp SOURCE_DIR abseil_SOURCE_DIR)
    include_directories(${abseil_SOURCE_DIR})
  endif()
endif()

# ------------------------------------------------------------------------------
# Protobuf & GoogleTest Setup for Converter & Unit Tests
# ------------------------------------------------------------------------------
if (NOT TARGET protobuf::libprotobuf-lite AND NOT TARGET protobuf::libprotobuf)
  include(FetchContent)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
  set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v36.2
    GIT_SHALLOW TRUE
  )
  if (MSVC)
    set(CMAKE_CXX_STANDARD 17)
  endif()
  FetchContent_MakeAvailable(protobuf)
  if (MSVC)
    set(CMAKE_CXX_STANDARD 20)
  endif()
  FetchContent_GetProperties(protobuf SOURCE_DIR protobuf_SOURCE_DIR)
  include_directories(${protobuf_SOURCE_DIR}/src ${protobuf_SOURCE_DIR}/third_party/utf8_range)

  if (TARGET utf8_validity)
    if (MSVC)
      target_compile_options(utf8_validity PRIVATE /FIstring.h)
    else()
      target_compile_options(utf8_validity PRIVATE -include string.h)
    endif()
  endif()
else()
  FetchContent_GetProperties(protobuf SOURCE_DIR protobuf_SOURCE_DIR)
  if (protobuf_SOURCE_DIR)
    include_directories(${protobuf_SOURCE_DIR}/src ${protobuf_SOURCE_DIR}/third_party/utf8_range)
  endif()
endif()

if (TARGET protobuf::libprotobuf-lite)
  set(LITE_PROTOBUF_LIB protobuf::libprotobuf-lite)
elseif (TARGET protobuf::libprotobuf)
  set(LITE_PROTOBUF_LIB protobuf::libprotobuf)
endif()

if (NOT SPM_MODEL_PROTO_SRCS)
  if (SPM_PROTOC_EXECUTABLE)
    set(LITE_PROTOC_CMD "${SPM_PROTOC_EXECUTABLE}")
    set(LITE_PROTOC_DEP "")
  elseif (CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    set(LITE_PROTOC_CMD ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:protobuf::protoc>)
    set(LITE_PROTOC_DEP protobuf::protoc)
  else()
    set(LITE_PROTOC_CMD $<TARGET_FILE:protobuf::protoc>)
    set(LITE_PROTOC_DEP protobuf::protoc)
  endif()
  set(SPM_MODEL_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.h")
  set(SPM_MODEL_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.cc" "${SPM_MODEL_PROTO_HDRS}")
  add_custom_command(
    OUTPUT ${SPM_MODEL_PROTO_SRCS}
    COMMAND ${LITE_PROTOC_CMD} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I${SPM_ROOT_DIR}/src ${SPM_ROOT_DIR}/src/sentencepiece_model.proto
    DEPENDS ${SPM_ROOT_DIR}/src/sentencepiece_model.proto ${LITE_PROTOC_DEP}
    COMMENT "Running cpp protocol buffer compiler on sentencepiece_model.proto for lite"
    VERBATIM
  )
endif()

# ------------------------------------------------------------------------------
# 1. Target: sentencepiece_lite
# Core Lite Runtime C++ static library (libsentencepiece_lite.a)
# ------------------------------------------------------------------------------
add_library(sentencepiece_lite STATIC
  sentencepiece_lite.cc
  sentencepiece_lite.h
  "${LITE_FBS_GEN_HDR}"
)

add_dependencies(sentencepiece_lite generate_sentencepiece_lite_fbs)

target_include_directories(sentencepiece_lite PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(sentencepiece_lite PUBLIC flatbuffers)

if (SPM_LITE_ENABLE_ABSL AND TARGET absl::base)
  target_compile_definitions(sentencepiece_lite PUBLIC SENTENCEPIECE_LITE_USE_ABSL=1)
  target_link_libraries(sentencepiece_lite PUBLIC
    absl::function_ref
    absl::span
    absl::inlined_vector
  )
endif()

if (SPM_LITE_ENABLE_UTF8_RANGE AND TARGET utf8_validity)
  target_compile_definitions(sentencepiece_lite PUBLIC SENTENCEPIECE_LITE_USE_UTF8_RANGE=1)
  target_link_libraries(sentencepiece_lite PUBLIC utf8_validity)
  if (protobuf_SOURCE_DIR)
    target_include_directories(sentencepiece_lite PRIVATE ${protobuf_SOURCE_DIR}/third_party/utf8_range)
  endif()
endif()

# ------------------------------------------------------------------------------
# 2. Target: cached_sentencepiece_lite
# High-performance 2-Way Set-Associative Token Cache for SentencePieceLite
# ------------------------------------------------------------------------------
add_library(cached_sentencepiece_lite STATIC
  cached_sentencepiece_lite.cc
  cached_sentencepiece_lite.h
)

target_include_directories(cached_sentencepiece_lite PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${SPM_ROOT_DIR}>
  $<BUILD_INTERFACE:${SPM_ROOT_DIR}/third_party/rapidhash>
)

target_link_libraries(cached_sentencepiece_lite PUBLIC
  sentencepiece_lite
)

# ------------------------------------------------------------------------------
# 3. Target: sentencepiece_lite_fb_converter
# Standalone Model Converter static library (libsentencepiece_lite_fb_converter.a)
# ------------------------------------------------------------------------------
add_library(sentencepiece_lite_fb_converter STATIC
  sentencepiece_model_converters.cc
  sentencepiece_model_converters.h
  ${SPM_MODEL_PROTO_SRCS}
)

if (NOT MSVC)
  target_compile_definitions(sentencepiece_lite_fb_converter PRIVATE HAVE_PTHREAD=1)
endif()

target_include_directories(sentencepiece_lite_fb_converter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(sentencepiece_lite_fb_converter PUBLIC
  sentencepiece_lite
  flatbuffers
  ${LITE_PROTOBUF_LIB}
  absl::status
  absl::statusor
  absl::strings
)

if (ATOMIC_LIB)
  target_link_libraries(sentencepiece_lite_fb_converter PUBLIC ${ATOMIC_LIB})
endif()

# ------------------------------------------------------------------------------
# Executables / Tools
# ------------------------------------------------------------------------------
add_executable(spm_to_fb spm_to_fb_main.cc)
target_link_libraries(spm_to_fb
  sentencepiece_lite
  sentencepiece_lite_fb_converter
  absl::flags
  absl::flags_parse
)

add_executable(spm_lite spm_lite_main.cc)
target_link_libraries(spm_lite
  sentencepiece_lite
  sentencepiece_lite_fb_converter
  absl::flags
  absl::flags_parse
  absl::strings
)

# ------------------------------------------------------------------------------
# Unit Tests (using GoogleTest)
# ------------------------------------------------------------------------------
if (BUILD_TESTING OR SPM_BUILD_TEST OR SENTENCEPIECE_LITE_STANDALONE_BUILD)
  enable_testing()

  if (NOT TARGET gtest_main)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.18.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(sentencepiece_lite_test
    sentencepiece_lite_test.cc
    ${SPM_MODEL_PROTO_SRCS}
  )

  target_include_directories(sentencepiece_lite_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SPM_ROOT_DIR}/src
  )

  target_link_libraries(sentencepiece_lite_test
    sentencepiece_lite
    sentencepiece_lite_fb_converter
    ${LITE_PROTOBUF_LIB}
    gtest_main
    gmock
    absl::status
    absl::flags
    absl::flags_parse
    absl::strings
  )

  add_test(
    NAME sentencepiece_lite_test
    COMMAND sentencepiece_lite_test
    WORKING_DIRECTORY ${SPM_ROOT_DIR}
  )

  add_executable(cached_sentencepiece_lite_test
    cached_sentencepiece_lite_test.cc
    ${SPM_MODEL_PROTO_SRCS}
  )

  target_include_directories(cached_sentencepiece_lite_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SPM_ROOT_DIR}/src
  )

  target_link_libraries(cached_sentencepiece_lite_test
    cached_sentencepiece_lite
    sentencepiece_lite
    sentencepiece_lite_fb_converter
    ${LITE_PROTOBUF_LIB}
    gtest_main
    gmock
    absl::status
    absl::flags
    absl::flags_parse
    absl::strings
  )

  add_test(
    NAME cached_sentencepiece_lite_test
    COMMAND cached_sentencepiece_lite_test
    WORKING_DIRECTORY ${SPM_ROOT_DIR}
  )

  if (TARGET sentencepiece-static)
    add_executable(sentencepiece_lite_canonical_test
      sentencepiece_lite_canonical_test.cc
      ${SPM_MODEL_PROTO_SRCS}
    )

    target_include_directories(sentencepiece_lite_canonical_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SPM_ROOT_DIR}/src
    )

    target_link_libraries(sentencepiece_lite_canonical_test
      sentencepiece_lite
      sentencepiece_lite_fb_converter
      sentencepiece-static
      ${LITE_PROTOBUF_LIB}
      gtest_main
      gmock
      absl::status
      absl::flags
      absl::flags_parse
      absl::strings
    )

    add_test(
      NAME sentencepiece_lite_canonical_test
      COMMAND sentencepiece_lite_canonical_test
      WORKING_DIRECTORY ${SPM_ROOT_DIR}
    )
  endif()
endif()

# ------------------------------------------------------------------------------
# Benchmark Target (using google/benchmark via FetchContent)
# ------------------------------------------------------------------------------
if (SPM_ENABLE_BENCHMARK)
  if (NOT TARGET benchmark::benchmark)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.5
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(google_benchmark)
  endif()

  add_executable(sentencepiece_lite_benchmark
    sentencepiece_lite_benchmark.cc
    sentencepiece_lite.cc
    sentencepiece_model_converters.cc
    ${SPM_MODEL_PROTO_SRCS}
  )

  if (NOT MSVC)
    target_compile_definitions(sentencepiece_lite_benchmark PRIVATE HAVE_PTHREAD=1)
  endif()

  target_link_libraries(sentencepiece_lite_benchmark
    cached_sentencepiece_lite
    sentencepiece_lite
    flatbuffers
    benchmark::benchmark
    ${LITE_PROTOBUF_LIB}
    absl::status
    absl::statusor
    absl::strings
    absl::hash
    absl::raw_hash_set
  )
endif()

# ------------------------------------------------------------------------------
# 4. Target: Amalgamation Distribution (2-file standalone distribution)
# ------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)

if (Python3_Interpreter_FOUND)
  set(AMALGAMATED_DIST_DIR "${SPM_ROOT_DIR}/dist")
  set(AMALGAMATED_HDR "${AMALGAMATED_DIST_DIR}/sentencepiece_lite.h")
  set(AMALGAMATED_SRC "${AMALGAMATED_DIST_DIR}/sentencepiece_lite.cc")

  add_custom_command(
    OUTPUT "${AMALGAMATED_HDR}" "${AMALGAMATED_SRC}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/amalgamate.py"
            --repo_root "${SPM_ROOT_DIR}"
            --build_dir "${CMAKE_CURRENT_BINARY_DIR}"
            --output_dir "${AMALGAMATED_DIST_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/amalgamate.py"
            generate_sentencepiece_lite_fbs
            "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_lite.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_lite.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/cached_sentencepiece_lite.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/cached_sentencepiece_lite.cc"
    COMMENT "Generating 2-file standalone amalgamation distribution (dist/)"
    VERBATIM
  )

  add_custom_target(amalgamate DEPENDS "${AMALGAMATED_HDR}" "${AMALGAMATED_SRC}")
endif()

