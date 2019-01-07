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

build_tf_wrapper() {
  if [ "$1" != "" ]; then
    pkg_name="==$1"
  fi

  # Builds _sentencepiece_processor_ops.so
  pip install tensorflow${pkg_name} --upgrade

  TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  TF_VERSION=( $(python -c 'import tensorflow as tf; print(tf.__version__)') )

  g++ -std=c++11 -shared \
    -I../../src \
    -fPIC ${TF_CFLAGS[@]} -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -Wl,-all_load  \
    /usr/local/lib/libsentencepiece.a \
    -Wl,-noall_load \
    sentencepiece_processor_ops.cc \
    -o tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION} \
    ${TF_LFLAGS[@]}

  strip -x tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION}
}

build() {
  cd tensorflow
  rm -fr build
  mkdir -p build
  cd build

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TENSORFLOW_SHARED=ON
  make -j4 VERBOSE=1
  make install
  cd ..

  which python
  which pip
  python --version

  curl -L -O https://bootstrap.pypa.io/get-pip.py
  sudo python get-pip.py --no-setuptools --no-wheel --ignore-installed
  pip install --upgrade setuptools
  pip install wheel

  build_tf_wrapper ""
  build_tf_wrapper "1.11.0"
  build_tf_wrapper "1.10.0"
  build_tf_wrapper "1.9.0"
  build_tf_wrapper "1.8.0"
  build_tf_wrapper "1.7.0"

  # Builds Python manylinux wheel package.
  # Platform name is determined by the tensorflow pip package.
  # TODO(taku): Automatically detect the platname of tensoflow-pip
  # PLAT_NAME=$(python -c 'import distutils.util; print(distutils.util.get_platform())')
  PLAT_NAME=macosx_10_10_x86_64
  python setup.py bdist_wheel --universal --plat-name=${PLAT_NAME}
  python setup.py test
  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info tmp
}

build
