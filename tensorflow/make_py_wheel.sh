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

run_docker() {
  cd `dirname $0`
  docker pull $1
  docker run --rm -ti --name tf_sentencepiece \
    -v `pwd`/../:/sentencepiece -w /sentencepiece/tensorflow \
    -td $1 /bin/bash
  docker exec tf_sentencepiece bash -c "./build.sh native"
  docker stop tf_sentencepiece
}

build() {
  rm -fr tmp
  mkdir -p tmp

  export PATH="/opt/python/cp27-cp27mu/bin:${PATH}"

  # Installs necessary libraries under `tmp` sub directory.
  cd tmp

  # Install libtool
  curl -L -O http://ftpmirror.gnu.org/libtool/libtool-2.4.6.tar.gz
  tar zxfv libtool-2.4.6.tar.gz
  cd libtool-2.4.6
  ./configure
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
  make install || true
  cd ../..

  # Install sentencepiece
  cd ..
  make distclean || true
  ./autogen.sh
  grep -v PKG_CHECK_MODULES configure > tmp
  mv tmp -f configure
  chmod +x configure
  LIBS+="-pthread -L/usr/local/lib -lprotobuf" ./configure --disable-shared --with-pic
  make CXXFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
       CFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" -j4
  make install || true

  # Builds _sentencepiece_processor_ops.so
  cd tensorflow
  pip install tensorflow
  TF_CFLAGS="-I/opt/python/cp27-cp27mu/lib/python2.7/site-packages/tensorflow/include"
  TF_LFLAGS="-L/opt/python/cp27-cp27mu/lib/python2.7/site-packages/tensorflow -ltensorflow_framework"

  g++ -std=c++11 -shared \
    -I../src \
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
  python setup.py bdist_wheel --universal --plat-name=linux_x86_64
  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info tmp
  cd .. && make distclean
}

if [ "$1" = "native" ]; then
  build
else
  run_docker quay.io/pypa/manylinux1_x86_64
fi
