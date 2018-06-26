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
  docker run --rm -ti --name py_sentencepiece \
    -v `pwd`/../:/sentencepiece -w /sentencepiece/python \
    -td $1 /bin/bash
  docker exec py_sentencepiece bash -c "./make_py_wheel.sh native"
  docker stop py_sentencepiece
}

build() {
  rm -fr tmp
  mkdir -p tmp

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
  make CXXFLAGS+="-std=c++11 -O3" \
       CFLAGS+="-std=c++11 -O3" -j4
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
  make CXXFLAGS+="-std=c++11 -O3" \
       CFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" -j4
  make install || true

  cd python
  for i in /opt/python/*
  do
    $i/bin/python setup.py bdist
    strip build/*/*.so
    $i/bin/python setup.py bdist_wheel
    $i/bin/python setup.py test
    rm -fr build
    rm -fr *.so
  done

  cd dist
  for i in *${TRG}.whl
  do
    auditwheel repair $i
  done

  mv -f wheelhouse/*${TRG}.whl .
  cd .. && rm -fr tmp
  cd .. && make distclean
}

if [ "$1" = "native" ]; then
  build
elif [ "$#" -eq 1 ]; then
  run_docker quay.io/pypa/manylinux1_${1}  ${1}
else
  run_docker quay.io/pypa/manylinux1_i686
  run_docker quay.io/pypa/manylinux1_x86_64
fi
