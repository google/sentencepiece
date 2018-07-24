#!/bin/bash

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

set -e  # exit immediately on error
set -x  # display all commands

PROTOBUF_VERSION=3.6.0
CMAKE_VERSION=3.12.0

run_docker() {
  cd `dirname $0`
  docker pull $1
  docker run --rm -ti --name tf_sentencepiece \
    -v `pwd`/../:/sentencepiece -w /sentencepiece/tensorflow \
    -td $1 /bin/bash
  docker exec tf_sentencepiece bash -c "./make_py_wheel.sh native"
  docker stop tf_sentencepiece
}

build() {
  TRG=$1
  rm -fr build
  mkdir -p build
  cd build
  
  export PATH="/opt/python/cp27-cp27mu/bin:${PATH}"

  # Install cmake
  curl -L -O https://cmake.org/files/v3.12/cmake-${CMAKE_VERSION}.tar.gz
  tar zxfv cmake-${CMAKE_VERSION}.tar.gz
  cd cmake-${CMAKE_VERSION}
  ./bootstrap
  make -j4
  make install
  cd ..

  # Install protobuf
  curl -L -O https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
  tar zxfv protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
  cd protobuf-${PROTOBUF_VERSION}
  ./configure --disable-shared --with-pic
  make CXXFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
    CFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" -j4
  make install
  cd ..

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TENSORFLOW_SHARED=ON
  make -j4
  make install
  cd ..

  # Builds _sentencepiece_processor_ops.so
  pip install tensorflow
  TF_CFLAGS="-I/opt/python/cp27-cp27mu/lib/python2.7/site-packages/tensorflow/include"
  TF_LFLAGS="-L/opt/python/cp27-cp27mu/lib/python2.7/site-packages/tensorflow -ltensorflow_framework"

  g++ -std=c++11 -shared \
    -I../../src \
    -fPIC ${TF_CFLAGS[@]} -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -Wl,--whole-archive \
    /usr/local/lib/libprotobuf.a \
    /usr/local/lib/libsentencepiece.a \
    -Wl,--no-whole-archive \
    sentencepiece_processor_ops.cc \
    -o tf_sentencepiece/_sentencepiece_processor_ops.so \
    ${TF_LFLAGS[@]}
  strip tf_sentencepiece/_sentencepiece_processor_ops.so

  # Builds Python manylinux wheel package.
  python setup.py bdist_wheel --universal --plat-name=manylinux1_x86_64
  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info
}

if [ "$1" = "native" ]; then
  build
else
  run_docker quay.io/pypa/manylinux1_x86_64
fi
