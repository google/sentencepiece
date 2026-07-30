# Copyright 2018 Google Inc.
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
# limitations under the License.!

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
file(STRINGS "VERSION.txt" SPM_VERSION)
message(STATUS "VERSION: ${SPM_VERSION}")

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(sentencepiece VERSION ${SPM_VERSION} LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SPM_ENABLE_NFKC_COMPILE "Enables NFKC compile" OFF)
option(SPM_ENABLE_SHARED "Builds shared libaries in addition to static libraries." ON)
option(SPM_BUILD_TEST "Builds test binaries." OFF)
option(SPM_ENABLE_TCMALLOC "Enable TCMalloc if available." ON)
option(SPM_TCMALLOC_STATIC "Link static library of TCMALLOC." OFF)
option(SPM_ENABLE_MSVC_MT_BUILD, "Use /MT flag in MSVC build" OFF)
option(SPM_CROSS_SYSTEM_PROCESSOR, "Override system processor" "")
option(SPM_DISABLE_EMBEDDED_DATA, "Disable to embed pre-compiled data." OFF)
option(SPM_ENABLE_BENCHMARK "Build benchmark programs" OFF)
option(SPM_NLCODEC_BPE
       "Build the NLCodec fast BPE trainer from contrib/nlcodec (adds --nlcodec_bpe flag)."
       OFF)

set(SPM_PROTOBUF_PROVIDER "module" CACHE STRING "Provider of protobuf library")
set_property(CACHE SPM_PROTOBUF_PROVIDER PROPERTY STRINGS "module" "package")
set(SPM_ABSL_PROVIDER "module" CACHE STRING "Provider of absl library")
set_property(CACHE SPM_ABSL_PROVIDER PROPERTY STRINGS "module" "package")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

if (SPM_ENABLE_SHARED AND NOT WIN32)
 set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Includes processor name to avoid conflicts.
set(CPACK_PACKAGE_FILE_NAME
    "sentencepiece-${SPM_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

if (SPM_CROSS_SYSTEM_PROCESSOR)
 set(CMAKE_SYSTEM_PROCESSOR ${SPM_CROSS_SYSTEM_PROCESSOR})
 set(CMAKE_CROSSCOMPILING TRUE)
endif()

# Disable shared build on windows
if(WIN32)
  set(SPM_ENABLE_SHARED OFF)
endif()

# MSVC C++20 STL has a bug with Protobuf v25.6's UntypedMessage in
# std::variant (error C2139). Use C++17 under MSVC for Abseil and Protobuf.
# See: https://github.com/protocolbuffers/protobuf/issues/21310
if (MSVC)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 10.0) OR
   (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0))
  string(APPEND CMAKE_CXX_FLAGS " -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=''")
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR bin)
endif()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

if (NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
  set(CMAKE_INSTALL_INCLUDEDIR include)
endif()

if (UNIX)
  include(GNUInstallDirs)
  set(prefix ${CMAKE_INSTALL_PREFIX})
  set(exec_prefix "\${prefix}")
  set(libdir "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
  set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
else()
  set(prefix ${CMAKE_INSTALL_PREFIX})
  set(exec_prefix "\${prefix}")
  set(libdir "\${exec_prefix}/lib")
  set(includedir "\${prefix}/include")
endif()
set(GNUCXX_STD_SUPPORT_VERSION "4.3")

if(${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")
add_definitions(-D_FREEBSD)
endif()

if (SPM_PROTOBUF_PROVIDER STREQUAL "module" OR SPM_PROTOBUF_PROVIDER STREQUAL "package")
  set(libprotobuf_lite "protobuf-lite")
else()
  set(libprotobuf_lite "")
endif()

if (MSVC)
  add_definitions("/wd4267 /wd4244 /wd4305 /wd4996 /Zc:strictStrings /utf-8")
  if (SPM_ENABLE_MSVC_MT_BUILD)
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG          ${CMAKE_CXX_FLAGS_DEBUG})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_MINSIZEREL     ${CMAKE_CXX_FLAGS_MINSIZEREL})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE        ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  endif()
endif()

if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()
endif()

# Add -latomic on riscv64
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
  string(APPEND CMAKE_C_STANDARD_LIBRARIES " -latomic")
  string(APPEND CMAKE_CXX_STANDARD_LIBRARIES " -latomic")
endif()

# SPDX-License-Identifier: (MIT OR CC0-1.0)
# Copyright 2020 Jan Tojnar
# https://github.com/jtojnar/cmake-snips
#
# Modelled after Python’s os.path.join
# https://docs.python.org/3.7/library/os.path.html#os.path.join
# Windows not supported
function(join_paths joined_path first_path_segment)
    set(temp_path "${first_path_segment}")
    foreach(current_segment IN LISTS ARGN)
        if(NOT ("${current_segment}" STREQUAL ""))
            if(IS_ABSOLUTE "${current_segment}")
                set(temp_path "${current_segment}")
            else()
                set(temp_path "${temp_path}/${current_segment}")
            endif()
        endif()
    endforeach()
    set(${joined_path} "${temp_path}" PARENT_SCOPE)
endfunction()

join_paths(libdir_for_pc_file "\${exec_prefix}" "${CMAKE_INSTALL_LIBDIR}")
join_paths(includedir_for_pc_file "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")
join_paths(datadir_for_pc_file "\${prefix}" "${CMAKE_INSTALL_DATADIR}" "sentencepiece")

join_paths(datadir_for_config_file "${prefix}" "${CMAKE_INSTALL_DATADIR}" "sentencepiece")
set(INSTALL_DATADIR ${datadir_for_config_file})

configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "config.h")
configure_file("${PROJECT_SOURCE_DIR}/sentencepiece.pc.in" "sentencepiece.pc" @ONLY)

if (NOT MSVC)
  add_compile_options("-Wno-deprecated-declarations")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

if (SPM_BUILD_TEST)
  enable_testing()
  include(FetchContent)
  if (MSVC)
    if (SPM_ENABLE_MSVC_MT_BUILD)
      set(gtest_force_shared_crt OFF CACHE BOOL "Disable shared CRT for gtest" FORCE)
    else()
      set(gtest_force_shared_crt ON CACHE BOOL "Enable shared CRT for gtest" FORCE)
    endif()
  endif()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

if (SPM_DISABLE_EMBEDDED_DATA)
 file(GLOB EMBEDDED_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.bin")
 install(FILES ${EMBEDDED_DATA_FILES} DESTINATION ${INSTALL_DATADIR})
endif()

if (SPM_ABSL_PROVIDER STREQUAL "module")
  include(FetchContent)
  if (MSVC)
    if (SPM_ENABLE_MSVC_MT_BUILD)
      set(ABSL_MSVC_STATIC_RUNTIME ON CACHE BOOL "Use static MSVC runtime for abseil" FORCE)
    else()
      set(ABSL_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Use static MSVC runtime for abseil" FORCE)
    endif()
  endif()
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20260526.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(abseil-cpp)
  FetchContent_GetProperties(abseil-cpp SOURCE_DIR abseil_SOURCE_DIR)
  if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/absl)
    file(CREATE_LINK
         ${abseil_SOURCE_DIR}/absl
         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/absl
         SYMBOLIC)
  endif()
elseif (SPM_ABSL_PROVIDER STREQUAL "package")
  find_package(absl REQUIRED)
  get_target_property(ABSL_INCLUDE_DIRS absl::base INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/absl)
    file(CREATE_LINK
         ${ABSL_INCLUDE_DIRS}/absl
         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/absl
         SYMBOLIC)
  endif()
  include_directories(${ABSL_INCLUDE_DIRS})
endif()

if (SPM_PROTOBUF_PROVIDER STREQUAL "module")
  include(FetchContent)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable protobuf tests" FORCE)
  set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "Disable protobuf examples" FORCE)
  set(protobuf_INSTALL OFF CACHE BOOL "Disable protobuf install" FORCE)
  set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "Disable utf8_range install" FORCE)
  if (MSVC)
    if (SPM_ENABLE_MSVC_MT_BUILD)
      set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "Link static runtime libraries for protobuf" FORCE)
    else()
      set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link static runtime libraries for protobuf" FORCE)
    endif()
  endif()

  if (SPM_PROTOC_EXECUTABLE)
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Disable protoc binary as SPM_PROTOC_EXECUTABLE is set" FORCE)
    set(protobuf_BUILD_LIBPROTOC OFF CACHE BOOL "Disable libprotoc as SPM_PROTOC_EXECUTABLE is set" FORCE)
  else()
    set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "Enable protoc binary" FORCE)
    set(protobuf_BUILD_LIBPROTOC ON CACHE BOOL "Enable libprotoc" FORCE)
  endif()

  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v25.6
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(protobuf)

  if (SPM_PROTOC_EXECUTABLE)
    if (NOT TARGET protobuf::protoc)
      add_executable(protobuf::protoc IMPORTED GLOBAL)
      set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION "${SPM_PROTOC_EXECUTABLE}")
    endif()
  else()
    if (NOT TARGET protobuf::protoc AND TARGET protoc)
      add_executable(protobuf::protoc ALIAS protoc)
    endif()
  endif()

  if (TARGET utf8_validity)
    if (MSVC)
      target_compile_options(utf8_validity PRIVATE /FIstring.h)
    else()
      target_compile_options(utf8_validity PRIVATE -include string.h)
    endif()
  endif()
  FetchContent_GetProperties(protobuf SOURCE_DIR protobuf_SOURCE_DIR)
endif()

if(SPM_ENABLE_BENCHMARK)
  include(FetchContent)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest" FORCE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
  FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(google_benchmark)
  FetchContent_GetProperties(google_benchmark SOURCE_DIR benchmark_SOURCE_DIR)
  include_directories(${benchmark_SOURCE_DIR}/include)
endif()

add_subdirectory(src)
add_subdirectory(third_party)

set(CPACK_SOURCE_GENERATOR "TXZ")
set(CPACK_GENERATOR "7Z")
set(CPACK_PACKAGE_VERSION "${SPM_VERSION}")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_CONTACT "taku@google.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Taku Kudo")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;/dist/;/sdist/;~$;${CPACK_SOURCE_IGNORE_FILES}")
include(CPack)
