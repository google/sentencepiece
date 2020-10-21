#!/bin/sh

VERSION="$1"

mkdir bundled
cd bundled
# Try taged version. Othewise, use head.
git clone https://github.com/google/sentencepiece.git \
  -b v"${VERSION}" --depth 1 || \
  git clone https://github.com/google/sentencepiece.git --depth 1

cd sentencepiece
mkdir build
cd build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=../..
make -j $(nproc)
make install
cd ../..
