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
  pip install tensorflow${pkg_name} --upgrade --no-cache-dir -I

  pip uninstall numpy -y || true
  pip uninstall numpy -y || true
  pip uninstall numpy -y || true
  pip install numpy --upgrade --no-cache-dir -I

  TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  TF_VERSION=( $(python -c 'import tensorflow as tf; print(tf.__version__)') )

  TF_LFLAGS2=`echo -n ${TF_LFLAGS[@]} | sed -e 's/-l:lib/-l/' -e 's/.[12].dylib//'`

  g++ -std=c++11 -shared -undefined dynamic_lookup \
    -I../../src \
    -D_USE_TF_STRING_VIEW \
    -fPIC ${TF_CFLAGS[@]} -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -Wl,-force_load \
    /usr/local/lib/libsentencepiece.a \
    sentencepiece_processor_ops.cc \
    -o tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION} \
    ${TF_LFLAGS2}

  strip -x tf_sentencepiece/_sentencepiece_processor_ops.so.${TF_VERSION}
}

build() {
  VERSION="3.7"
  URL="https://www.python.org/ftp/python/3.7.0/python-3.7.0-macosx10.6.pkg"
  INSTALL_PATH="/Library/Frameworks/Python.framework/Versions/${VERSION}/bin"
  CURRENT_PATH=${PATH}

  curl -L -o python.pkg ${URL}
  sudo installer -pkg python.pkg -target /

  if [ -f "${INSTALL_PATH}/python3" ]; then
    ln -s ${INSTALL_PATH}/python3        ${INSTALL_PATH}/python
    ln -s ${INSTALL_PATH}/python3-config ${INSTALL_PATH}/python-config
    ln -s ${INSTALL_PATH}/pip3           ${INSTALL_PATH}/pip
  fi

  curl -L -O https://bootstrap.pypa.io/get-pip.py

  export PATH="${INSTALL_PATH}:${CURRENT_PATH}"
  ls -l ${INSTALL_PATH}
  which python
  which pip
  python --version
  sudo python get-pip.py --no-setuptools --no-wheel --ignore-installed
  pip install --upgrade setuptools
  pip install wheel
  pip install delocate

  cd tensorflow
  rm -fr build
  mkdir -p build
  cd build

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TENSORFLOW_SHARED=ON
  make -j4 VERBOSE=1
  make install
  cd ..

  # Remove pre-installed Linux so files.
  rm -f tf_sentencepiece/*.so.*

  build_tf_wrapper "2.2.0"
  build_tf_wrapper "2.1.0"
#  build_tf_wrapper "2.0.1"
  build_tf_wrapper "2.0.0"
#  build_tf_wrapper "1.15.2"
  build_tf_wrapper "1.15.0"
#  build_tf_wrapper "1.14.0"
#  build_tf_wrapper "1.13.2"
#  build_tf_wrapper "1.13.1"

  # Builds Python manylinux wheel package.
  # Platform name is determined by the tensorflow pip package.
  # TODO(taku): Automatically detect the platname of tensoflow-pip
  # PLAT_NAME=$(python -c 'import distutils.util; print(distutils.util.get_platform())')
  PLAT_NAME=macosx_10_10_x86_64
  python setup.py bdist_wheel --universal --plat-name=${PLAT_NAME}
  # python setup.py test
  python setup.py sdist

  rm -fr build tf_sentencepiece.egg-info tmp
}

build
