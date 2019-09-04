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

build_tf_wrapper() {
  if [ "$1" != "" ]; then
    pkg_name="==$1"
  fi

  # Builds _sentencepiece_processor_ops.so
  pip install tensorflow${pkg_name} --upgrade

  TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  TF_VERSION=( $(python -c 'import tensorflow as tf; print(tf.__version__)') )
  
  echo TF_CFLAGS=${TF_CFLAGS[@]}
  echo TF_LFLAGS=${TF_LFLAGS[@]}
  echo TF_VERSION=${TF_VERSION}

  g++ -std=c++11 -shared \
    -I../../src \
    -fPIC ${TF_CFLAGS[@]} -O2 \
    -Wl,--whole-archive \
    /usr/local/lib/libsentencepiece.a \
    -Wl,--no-whole-archive \
    sentencepiece_processor_ops.cc \
    -o tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION} \
    ${TF_LFLAGS[@]}

  strip tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION}
}

build() {
  TRG=$1
  rm -fr build
  mkdir -p build
  cd build

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TENSORFLOW_SHARED=ON  
  make -j4
  make install
  cd ..
  
  build_tf_wrapper "2.0.0-rc0"
  build_tf_wrapper "2.0.0-beta1"
  build_tf_wrapper "1.14.0"
  build_tf_wrapper "1.13.1"

  # Builds Python manylinux wheel package.
  python setup.py bdist_wheel --universal --plat-name=manylinux1_x86_64
  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info
}

if [ "$1" = "native" ]; then
  build
else
  run_docker tensorflow/tensorflow:custom-op-ubuntu14
fi
