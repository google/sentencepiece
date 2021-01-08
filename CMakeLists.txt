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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
file(STRINGS "VERSION.txt" SPM_VERSION)
message(STATUS "VERSION: ${SPM_VERSION}")
project(sentencepiece VERSION ${SPM_VERSION} LANGUAGES C CXX)

option(SPM_ENABLE_NFKC_COMPILE "Enables NFKC compile" OFF)
option(SPM_ENABLE_SHARED "Builds shared libaries in addition to static libraries." ON)
option(SPM_BUILD_TEST "Builds test binaries." OFF)
option(SPM_COVERAGE "Runs gcov to test coverage." OFF)
option(SPM_ENABLE_TENSORFLOW_SHARED "Makes a tensorflow compatible shared file." OFF)
option(SPM_ENABLE_TCMALLOC "Enable TCMalloc if available." ON)
option(SPM_TCMALLOC_STATIC "Link static library of TCMALLOC." OFF)
option(SPM_NO_THREADLOCAL "Disable thread_local operator" OFF)
option(SPM_USE_BUILTIN_PROTOBUF "Use built-in protobuf" ON)
option(SPM_USE_EXTERNAL_ABSL "Use external protobuf" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

if (SPM_USE_BUILTIN_PROTOBUF)
  set(libprotobuf_lite "")
else()
  set(libprotobuf_lite "-lprotobuf-lite")
endif()

if (MSVC)
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG          ${CMAKE_CXX_FLAGS_DEBUG})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_MINSIZEREL     ${CMAKE_CXX_FLAGS_MINSIZEREL})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE        ${CMAKE_CXX_FLAGS_RELEASE})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
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

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR bin)
endif()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

if (NOT DEFINED CMAKE_INSTALL_INCDIR)
  set(CMAKE_INSTALL_INCDIR include)
endif()

configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "config.h")
configure_file("${PROJECT_SOURCE_DIR}/sentencepiece.pc.in" "sentencepiece.pc" @ONLY)

if (NOT MSVC)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

if (SPM_BUILD_TEST)
  enable_testing()
endif()

if (SPM_USE_EXTERNAL_ABSL)
 add_subdirectory(third_party/abseil-cpp)
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
