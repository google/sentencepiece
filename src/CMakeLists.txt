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

list(APPEND SPM_LIBS absl::status)
list(APPEND SPM_LIBS absl::status_builder)
list(APPEND SPM_LIBS absl::strings)
list(APPEND SPM_LIBS absl::flags)
list(APPEND SPM_LIBS absl::flags_parse)
list(APPEND SPM_LIBS absl::log)
list(APPEND SPM_LIBS absl::log_initialize)
list(APPEND SPM_LIBS absl::check)
list(APPEND SPM_LIBS absl::random_random)
list(APPEND SPM_LIBS absl::time)

if (SPM_PROTOBUF_PROVIDER STREQUAL "module")
  FetchContent_GetProperties(protobuf SOURCE_DIR protobuf_SOURCE_DIR)
  set(PROTOBUF_LITE_LIBRARY protobuf::libprotobuf-lite)
  list(APPEND SPM_LIBS ${PROTOBUF_LITE_LIBRARY})
  if (TARGET utf8_validity)
    list(APPEND SPM_LIBS utf8_validity)
  endif()
  set(PROTOBUF_LITE_SRCS "")
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  if (SPM_PROTOC_EXECUTABLE)
    set(PROTOC_CMD "${SPM_PROTOC_EXECUTABLE}")
    set(PROTOC_DEP "")
  elseif (CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    set(PROTOC_CMD ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:protoc>)
    set(PROTOC_DEP protoc)
  else()
    set(PROTOC_CMD $<TARGET_FILE:protoc>)
    set(PROTOC_DEP protoc)
  endif()

  set(SPM_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pb.h")
  set(SPM_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece.pb.cc" "${SPM_PROTO_HDRS}")
  add_custom_command(
    OUTPUT ${SPM_PROTO_SRCS}
    COMMAND ${PROTOC_CMD} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece.proto ${PROTOC_DEP}
    COMMENT "Running cpp protocol buffer compiler on sentencepiece.proto"
    VERBATIM
  )

  set(SPM_MODEL_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.h")
  set(SPM_MODEL_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_model.pb.cc" "${SPM_MODEL_PROTO_HDRS}")
  add_custom_command(
    OUTPUT ${SPM_MODEL_PROTO_SRCS}
    COMMAND ${PROTOC_CMD} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_model.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_model.proto ${PROTOC_DEP}
    COMMENT "Running cpp protocol buffer compiler on sentencepiece_model.proto"
    VERBATIM
  )
elseif (SPM_PROTOBUF_PROVIDER STREQUAL "package")
  find_package(Protobuf REQUIRED)
  include_directories(${Protobuf_INCLUDE_DIRS})
  protobuf_generate_cpp(SPM_PROTO_SRCS SPM_PROTO_HDRS sentencepiece.proto)
  protobuf_generate_cpp(SPM_MODEL_PROTO_SRCS SPM_MODEL_PROTO_HDRS sentencepiece_model.proto)
  set(PROTOBUF_LITE_SRCS "")
  include_directories(${PROTOBUF_INCLUDE_DIR})
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

find_package(OpenMP)
if (OpenMP_FOUND)
  add_compile_definitions(LIBSAIS_OPENMP)
  string(APPEND CMAKE_C_FLAGS " ${OpenMP_C_FLAGS}")
  string(APPEND CMAKE_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
endif()

set(SPM_SRCS
  ${PROTOBUF_LITE_SRCS}
  ${SPM_PROTO_HDRS}
  ${SPM_PROTO_SRCS}
  ${SPM_MODEL_PROTO_HDRS}
  ${SPM_MODEL_PROTO_SRCS}
  bpe_model.h
  common.h
  normalizer.h
  util.h
  freelist.h
  filesystem.h
  init.h
  sentencepiece_processor.h
  word_model.h
  model_factory.h
  char_model.h
  model_interface.h
  testharness.h
  unigram_model.h
  bpe_model.cc
  char_model.cc
  filesystem.cc
  init.cc
  model_factory.cc
  model_interface.cc
  normalizer.cc
  sentencepiece_processor.cc
  unigram_model.cc
  util.cc
  word_model.cc)

set(SPM_TRAIN_SRCS
  ${SPM_PROTO_HDRS}
  ${SPM_MODEL_PROTO_HDRS}
  builder.h
  normalization_rule.h
  unicode_script.h
  unicode_script_map.h
  trainer_factory.h
  trainer_interface.h
  unigram_model_trainer.h
  word_model_trainer.h
  char_model_trainer.h
  bpe_model_trainer.h
  sentencepiece_trainer.h
  builder.cc
  unicode_script.cc
  trainer_factory.cc
  trainer_interface.cc
  unigram_model_trainer.cc
  word_model_trainer.cc
  char_model_trainer.cc
  bpe_model_trainer.cc
  sentencepiece_trainer.cc
  ../third_party/libsais/libsais.c)

set(SPM_TEST_SRCS
  ${SPM_PROTO_HDRS}
  ${SPM_MODEL_PROTO_HDRS}
  testharness.h
  bpe_model_test.cc
  bpe_model_trainer_test.cc
  builder_test.cc
  invalid_model_test.cc
  char_model_test.cc
  char_model_trainer_test.cc
  filesystem_test.cc
  model_factory_test.cc
  model_interface_test.cc
  normalizer_test.cc
  sentencepiece_processor_test.cc
  sentencepiece_processor_parallel_test.cc
  sentencepiece_trainer_test.cc
  test_main.cc
  trainer_factory_test.cc
  trainer_interface_test.cc
  unicode_script_test.cc
  unigram_model_test.cc
  unigram_model_trainer_test.cc
  util_test.cc
  word_model_test.cc
  word_model_trainer_test.cc)

if (SPM_NLCODEC_BPE)
  set(NLCODEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/nlcodec)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_compile_definitions(SPM_NLCODEC_BPE)
  list(APPEND SPM_TRAIN_SRCS ${NLCODEC_DIR}/bpe_model_trainer_nlcodec.cc)
  list(APPEND SPM_TEST_SRCS ${NLCODEC_DIR}/bpe_model_trainer_nlcodec_test.cc)
endif()

find_package(Threads REQUIRED)

list(APPEND SPM_LIBS ${PROTOBUF_LITE_LIBRARY} Threads::Threads)

if (SPM_ENABLE_NFKC_COMPILE)
  find_package(ICU 4.4 COMPONENTS i18n data uc REQUIRED)
  include_directories(${ICU_INCLUDE_DIRS})
  add_compile_definitions(ENABLE_NFKC_COMPILE)
  list(APPEND SPM_LIBS ICU::i18n ICU::data ICU::uc)
endif()

if (SPM_DISABLE_EMBEDDED_DATA)
  add_compile_definitions(DISABLE_EMBEDDED_DATA)
endif()

if (SPM_ENABLE_TCMALLOC)
  if (SPM_TCMALLOC_STATIC)
    find_library(TCMALLOC_LIB NAMES libtcmalloc_minimal.a)
  else()
    find_library(TCMALLOC_LIB NAMES tcmalloc_minimal)
  endif()
  if (TCMALLOC_LIB)
    message(STATUS "Found TCMalloc: ${TCMALLOC_LIB}")
    list(APPEND SPM_LIBS ${TCMALLOC_LIB})
    add_definitions(-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)
  else()
    message(STATUS "Not Found TCMalloc: ${TCMALLOC_LIB}")
  endif()
endif()

if ((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "mips") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "m68k") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "powerpc") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch") OR
    (${CMAKE_SYSTEM_PROCESSOR} MATCHES "sh4"))
  find_library(ATOMIC_LIB NAMES atomic libatomic.so libatomic.so.1)
  if (ATOMIC_LIB)
    message(STATUS "Found atomic: ${ATOMIC_LIB}")
    list(APPEND SPM_LIBS "atomic")
  endif()
endif()


if (SPM_ENABLE_SHARED)
  add_library(sentencepiece SHARED ${SPM_SRCS})
  add_library(sentencepiece_train SHARED ${SPM_TRAIN_SRCS})
  if (ANDROID)
    target_link_libraries(sentencepiece PRIVATE log)
    target_link_libraries(sentencepiece_train PRIVATE log)
  endif()
endif()

add_library(sentencepiece-static STATIC ${SPM_SRCS})
add_library(sentencepiece_train-static STATIC ${SPM_TRAIN_SRCS})

target_link_libraries(sentencepiece-static PUBLIC ${SPM_LIBS})
target_link_libraries(sentencepiece_train-static PUBLIC sentencepiece-static ${SPM_LIBS})
add_dependencies(sentencepiece_train-static sentencepiece-static)

if (SPM_ENABLE_SHARED)
  target_link_libraries(sentencepiece PUBLIC ${SPM_LIBS})
  target_link_libraries(sentencepiece_train PUBLIC ${SPM_LIBS} sentencepiece)
  add_dependencies(sentencepiece_train sentencepiece)
  set(SPM_INSTALLTARGETS sentencepiece sentencepiece_train sentencepiece-static sentencepiece_train-static)
  set_target_properties(sentencepiece sentencepiece_train PROPERTIES SOVERSION 0 VERSION 0.0.0)
  set_target_properties(sentencepiece PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS YES)
  set_target_properties(sentencepiece_train PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS YES)
  if (MSVC)
    set_target_properties(sentencepiece PROPERTIES IMPORT_SUFFIX "_import.lib")
    set_target_properties(sentencepiece_train PROPERTIES IMPORT_SUFFIX "_import.lib")
  elseif (MINGW)
    set_target_properties(sentencepiece PROPERTIES IMPORT_SUFFIX ".dll.a")
    set_target_properties(sentencepiece_train PROPERTIES IMPORT_SUFFIX ".dll.a")
  endif()
else()
  add_library(sentencepiece ALIAS sentencepiece-static)
  add_library(sentencepiece_train ALIAS sentencepiece_train-static)
  set(SPM_INSTALLTARGETS sentencepiece-static sentencepiece_train-static)
endif()

set_target_properties(sentencepiece-static PROPERTIES OUTPUT_NAME "sentencepiece")
set_target_properties(sentencepiece_train-static PROPERTIES OUTPUT_NAME "sentencepiece_train")

if (NOT WIN32)
  set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  set_source_files_properties(
    sentencepiece.pb.cc sentencepiece_model.pb.cc
    PROPERTIES COMPILE_FLAGS "-Wno-misleading-indentation")
  set_source_files_properties(${SPM_TEST_SRCS}
    PROPERTIES COMPILE_FLAGS "-Wno-sign-compare")
  if (SPM_ENABLE_SHARED)
    set_property(TARGET sentencepiece APPEND_STRING PROPERTY COMPILE_FLAGS " -DPIC")
    set_property(TARGET sentencepiece_train APPEND_STRING PROPERTY COMPILE_FLAGS " -DPIC")
  endif()
endif()

add_executable(spm_encode spm_encode_main.cc)
add_executable(spm_decode spm_decode_main.cc)
add_executable(spm_normalize spm_normalize_main.cc)
add_executable(spm_train spm_train_main.cc)
add_executable(spm_export_vocab spm_export_vocab_main.cc)
add_executable(spm_eval spm_eval_main.cc)

target_link_libraries(spm_encode sentencepiece)
target_link_libraries(spm_decode sentencepiece)
target_link_libraries(spm_normalize sentencepiece sentencepiece_train)
target_link_libraries(spm_train sentencepiece sentencepiece_train)
target_link_libraries(spm_export_vocab sentencepiece)
target_link_libraries(spm_eval sentencepiece sentencepiece_train)

if (SPM_ENABLE_NFKC_COMPILE)
  add_executable(compile_charsmap compile_charsmap_main.cc)
  target_link_libraries(compile_charsmap sentencepiece sentencepiece_train)
endif()

list(APPEND SPM_INSTALLTARGETS
  spm_encode spm_decode spm_normalize spm_train spm_export_vocab)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  install(TARGETS ${SPM_INSTALLTARGETS}
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
install(TARGETS ${SPM_INSTALLTARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Generate modified headers for installation
set(GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/installed_headers")
set(MODIFIER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modify_headers.cmake")

add_custom_command(
  OUTPUT "${GENERATED_HEADERS_DIR}/sentencepiece_processor.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_processor.h" "${MODIFIER_SCRIPT}"
  COMMAND ${CMAKE_COMMAND} -DINPUT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_processor.h"
                           -DOUTPUT_FILE="${GENERATED_HEADERS_DIR}/sentencepiece_processor.h"
                           -P "${MODIFIER_SCRIPT}"
  COMMENT "Generating modified sentencepiece_processor.h for installation"
)

add_custom_command(
  OUTPUT "${GENERATED_HEADERS_DIR}/sentencepiece_trainer.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_trainer.h" "${MODIFIER_SCRIPT}"
  COMMAND ${CMAKE_COMMAND} -DINPUT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_trainer.h"
                           -DOUTPUT_FILE="${GENERATED_HEADERS_DIR}/sentencepiece_trainer.h"
                           -P "${MODIFIER_SCRIPT}"
  COMMENT "Generating modified sentencepiece_trainer.h for installation"
)

add_custom_target(generate_installed_headers ALL
  DEPENDS "${GENERATED_HEADERS_DIR}/sentencepiece_processor.h"
          "${GENERATED_HEADERS_DIR}/sentencepiece_trainer.h"
)

install(FILES "${GENERATED_HEADERS_DIR}/sentencepiece_trainer.h" "${GENERATED_HEADERS_DIR}/sentencepiece_processor.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if (NOT SPM_PROTOBUF_PROVIDER STREQUAL "internal")
  install(FILES ${SPM_PROTO_HDRS} ${SPM_MODEL_PROTO_HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/data" data_dir)

if (SPM_BUILD_TEST)
  enable_testing()
  add_executable(spm_test test_main.cc ${SPM_TEST_SRCS})

  target_link_libraries(spm_test sentencepiece sentencepiece_train gtest gmock)

  set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=definite,possible --error-exitcode=1")
  find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)

  add_test(NAME sentencepiece_test
    COMMAND $<TARGET_FILE:spm_test> --test_srcdir=${data_dir})
endif()

if (SPM_ENABLE_BENCHMARK)
  add_executable(sentencepiece_processor_benchmarks_test
        sentencepiece_processor_benchmarks_test.cc)
  target_link_libraries(sentencepiece_processor_benchmarks_test
        sentencepiece sentencepiece_train benchmark::benchmark gtest)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set_xcode_property(spm_encode PRODUCT_BUNDLE_IDENTIFIER "SentencePiece" All)
  set_xcode_property(spm_decode PRODUCT_BUNDLE_IDENTIFIER "SentencePiece" All)
  set_xcode_property(spm_normalize PRODUCT_BUNDLE_IDENTIFIER "SentencePiece" All)
  set_xcode_property(spm_train PRODUCT_BUNDLE_IDENTIFIER "SentencePiece" All)
  set_xcode_property(spm_export_vocab PRODUCT_BUNDLE_IDENTIFIER "SentencePiece" All)
endif()
