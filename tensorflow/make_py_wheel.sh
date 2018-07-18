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
  docker exec tf_sentencepiece bash -c "./make_py_wheel.sh native"
  docker stop tf_sentencepiece
}

build() {
  rm -fr tmp
  mkdir -p tmp

  PYPATH=$(python -c 'import os, sys; print(os.path.dirname(sys.executable))')
  export PATH="${PYPATH}:${PATH}"

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
  PROTOBUF_PKG_PATH=$(pwd)
  ./configure --disable-shared --with-pic
  make CXXFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
       CFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" -j4
  make install || true
  cd ../..
  export PKG_CONFIG_PATH=${PROTOBUF_PKG_PATH}:${PKG_CONFIG_PATH}

  # Install sentencepiece
  cd ..
  make distclean || true
  ./autogen.sh
  grep -v PKG_CHECK_MODULES configure > tmp
  mv -f tmp configure
  chmod +x configure
  LIBS+="-pthread -L/usr/local/lib -lprotobuf" ./configure --disable-shared --with-pic
  make CXXFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
       CFLAGS+="-std=c++11 -O3 -D_GLIBCXX_USE_CXX11_ABI=0" -j4
  make install || true

  # Builds _sentencepiece_processor_ops.so
  cd tensorflow
  pip install tensorflow
  TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  
  CXX_ARGS=(-std=c++11 -shared
    -I../src
    -fPIC ${TF_CFLAGS[@]} -O2
    -D_GLIBCXX_USE_CXX11_ABI=0)

  platform=$(uname)
  if [[ "${platform}" == 'Darwin' ]]; then
    CXX_ARGS+=(-Wl,-all_load
      /usr/local/lib/libprotobuf.a
      /usr/local/lib/libsentencepiece.a
      -Wl,-noall_load
      sentencepiece_processor_ops.cc)
  else
    CXX_ARGS+=(-Wl,--whole-archive
      /usr/local/lib/libprotobuf.a
      /usr/local/lib/libsentencepiece.a
      -Wl,--no-whole-archive
      sentencepiece_processor_ops.cc)
  fi

  CXX_ARGS+=(-o tf_sentencepiece/_sentencepiece_processor_ops.so
  ${TF_LFLAGS[@]})

  g++ ${CXX_ARGS[@]}
  if [[ "${platform}" == 'Darwin' ]]; then
    strip -x tf_sentencepiece/_sentencepiece_processor_ops.so
  else
    strip tf_sentencepiece/_sentencepiece_processor_ops.so
  fi


  # build any wheel package
  python setup.py bdist_wheel --universal

  # Builds Python manylinux wheel package
  python setup.py bdist_wheel --universal --plat-name=manylinux1_x86_64

  # Build platform specific whell package
  plat_name=$(python -c 'import distutils.util; print(distutils.util.get_platform())')
  python setup.py bdist_wheel --universal --plat-name=${plat_name}

  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info tmp
  cd .. && make distclean
}

if [ "$1" = "native" ]; then
  build
else
  run_docker quay.io/pypa/manylinux1_x86_64
fi
