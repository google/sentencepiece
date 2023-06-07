#!/bin/sh

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

setup_ubuntu() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y build-essential cmake git pkg-config python3-pip
  pip3 install --upgrade pip

  export PATH="/usr/local/bin:$PATH"

  . /etc/os-release
  if [ "${VERSION_ID}" = "14.04" ]; then
    apt-get install -y cmake3 python-dev
  fi
}

setup_debian() {
  setup_ubuntu
}

setup_fedora() {
  dnf update -y
  dnf install -y rpm-build gcc-c++ make cmake pkg-config python-pip python-devel
}

build_generic() {
  mkdir -p build
  cd build
  cmake .. -DSPM_BUILD_TEST=ON
  make -j2
  make CTEST_OUTPUT_ON_FAILURE=1 test
  make package_source
  cd ..
}

build_python() {
  cd build
  make install
  cd ..
  export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH
  export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig
  ldconfig -v
  cd python
  python3 setup.py test
  cd ..
}

build_linux_gcc_coverall_ubuntu() {
  setup_debian
  apt-get install -y lcov
  pip3 install cpp-coveralls
  pip3 install 'requests[security]'
  build_generic
  build_python
  mkdir -p build
  cd build
  cmake .. -DSPM_COVERAGE=ON
  make -j2
  make coverage
  coveralls --exclude-pattern '.*(include|usr|test|third_party|pb|_main).*' --gcov-options '\-lp' --gcov gcov
  cd ..
}

build_linux_gcc_ubuntu() {
  setup_ubuntu
  build_generic
  build_python
}

build_linux_gcc_ubuntu_i386() {
  setup_ubuntu
  build_generic
  build_python
}

build_linux_gcc_debian() {
  setup_debian
  build_generic
  build_python
}

build_linux_gcc_fedora() {
  setup_fedora
  build_generic
  build_python
}

build_linux_clang_ubuntu() {
  setup_ubuntu
  apt-get install -y clang
  export CXX="clang++" CC="clang"
  build_generic
  rm -fr build
}

build_osx() {
#  brew update
#  brew install protobuf || brew link --overwrite protobuf
#  brew link --overwrite python@2
  build_generic
#  cd build
#  make install
}

run_docker() {
  docker pull "$1"
  docker run -e COVERALLS_REPO_TOKEN=${COVERALLS_REPO_TOKEN} --rm -ti --name travis-ci -v `pwd`:/sentencepiece -w /sentencepiece -td "$1" /bin/bash
  docker exec travis-ci bash -c "./test.sh native $2"
  docker stop travis-ci
}

## main
if [ "$#" -ne 2 ]; then
  echo "sh test.sh <docker_image> <mode>."
  echo "when <docker_image> is native, runs command natively without docker."
  exit
fi

if [ "$1" = "native" ]; then
  eval "$2"
else
  run_docker $1 $2
fi
