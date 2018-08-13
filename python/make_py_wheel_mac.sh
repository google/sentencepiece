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

PROTOBUF_VERSION=3.6.1

build_python() {
  VERSION=$1
  URL=$2
  INSTALL_PATH=/Library/Frameworks/Python.framework/Versions/${VERSION}/bin

  curl -L -o python.pkg ${URL}
  installer -pkg python.pkg -target /
  ${INSTALL_PATH}/python get-pip.py --no-setuptools --no-wheel
  ${INSTALL_PATH}/python -m pip install --upgrade setuptools
  ${INSTALL_PATH}/python -m pip install wheel
  ${INSTALL_PATH}/python -m pip install delocate
  ${INSTALL_PATH}/python setup.py bdist_wheel
  ${INSTALL_PATH}/python setup.py test
  ${INSTALL_PATH}/delocate-listdeps dist/*.whl
  ${INSTALL_PATH}/delocate-wheel -w dist/delocated_wheel dist/*.whl
  rm -fr build
  rm -fr *.so
  rm -fr dist/*.whl
  rm -fr python.pkg
}

build() {
  cd python
  rm -fr build
  mkdir -p build
  cd build

  # Install protobuf
  curl -L -O https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
  tar zxfv protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
  cd protobuf-${PROTOBUF_VERSION}
  ./configure --disable-shared --with-pic
  make CXXFLAGS+="-std=c++11 -O3" CFLAGS+="-std=c++11 -O3" -j4
  make install || true
  cd ..

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF
  make -j4
  make install
  cd ..

  mkdir -p dist/delocated_wheel
  curl -L -O https://bootstrap.pypa.io/get-pip.py

  build_python 2.7 https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.6.pkg
  build_python 3.4 https://www.python.org/ftp/python/3.4.4/python-3.4.4-macosx10.6.pkg
  build_python 3.5 https://www.python.org/ftp/python/3.5.4/python-3.5.4-macosx10.6.pkg
  build_python 3.6 https://www.python.org/ftp/python/3.6.6/python-3.6.6-macosx10.6.pkg
  build_python 3.7 https://www.python.org/ftp/python/3.7.0/python-3.7.0-macosx10.6.pkg

  cd ..

  rm -fr build
}

build
