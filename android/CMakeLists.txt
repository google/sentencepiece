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

cmake_minimum_required(VERSION 3.22)

project(sentencepiece_android LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This CMakeLists.txt lives inside the sentencepiece repository under android/.
# The sentencepiece root is one level up.
set(SPM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ---------------------------------------------------------------------------
# Build the sentencepiece C++ library from the parent build
# ---------------------------------------------------------------------------
# Reuses the root CMakeLists.txt (abseil, protobuf, proto generation, etc.).
# SPM_IS_TOP_LEVEL will be FALSE here, so CPack/packaging is skipped.
add_subdirectory("${SPM_ROOT_DIR}" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# JNI shared library
# ---------------------------------------------------------------------------
find_package(JNI REQUIRED)

add_library(sentencepiece_jni SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main/jni/sentencepiece_jni.cc"
)

target_include_directories(sentencepiece_jni PRIVATE
    "${SPM_ROOT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${JNI_INCLUDE_DIRS}"
)

target_link_libraries(sentencepiece_jni PRIVATE
    sentencepiece
    JNI::JNI
)

# Android: strip the "lib" prefix so System.loadLibrary("sentencepiece_jni")
# finds libsentencepiece_jni.so.
set_target_properties(sentencepiece_jni PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# On Android, link the system log library used by sentencepiece's logging.
if (ANDROID)
    target_link_libraries(sentencepiece_jni PRIVATE log)
endif()
