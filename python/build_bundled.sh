#!/bin/sh

mkdir -p build

BUILD_DIR=./build
INSTALL_DIR=./build/root

if [ -f ./sentencepiece/src/CMakeLists.txt ]; then
  SRC_DIR=./sentencepiece
elif [ -f ../src/CMakeLists.txt ]; then
  SRC_DIR=..  
else
  echo "Error: SentencePiece C++ source files not found in ./sentencepiece or ../" >&2
  exit 1
fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake ${SRC_DIR} -B ${BUILD_DIR} \
  -DSPM_ENABLE_SHARED=OFF \
  -DSPM_DISABLE_EMBEDDED_DATA=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="-fPIC -fvisibility=default -ffunction-sections -fdata-sections"
cmake --build ${BUILD_DIR} --config Release --target install --parallel ${NPROC}

