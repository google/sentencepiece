#!/bin/sh

VERSION="$1"

mkdir -p build

BUILD_DIR=./build
INSTALL_DIR=./build/root

if [ -f ./sentencepiece/src/CMakeLists.txt ]; then
  SRC_DIR=./sentencepiece
elif [ -f ../src/CMakeLists.txt ]; then
  SRC_DIR=..  
else
  # Try tagged version. Othewise, use head.
  git clone https://github.com/google/sentencepiece.git -b v"${VERSION}" --depth 1 || \
  git clone https://github.com/google/sentencepiece.git --depth 1
  SRC_DIR=./sentencepiece
fi

if [ "${SPM_USE_PROTOC}" = "1" ]; then
  SPM_USE_PROTOC_FLAG="-DSPM_USE_PROTOC=ON"
else
  SPM_USE_PROTOC_FLAG="-DSPM_USE_PROTOC=OFF"
fi

cmake ${SRC_DIR} -B ${BUILD_DIR} \
  -DSPM_ENABLE_SHARED=OFF \
  -DSPM_DISABLE_EMBEDDED_DATA=ON \
  ${SPM_USE_PROTOC_FLAG} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="-fPIC -fvisibility=default -ffunction-sections -fdata-sections"
cmake --build ${BUILD_DIR} --config Release --target install --parallel $(nproc)
