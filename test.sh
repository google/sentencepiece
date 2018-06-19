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
  apt-get update
  apt-get install -y build-essential autoconf automake libtool git \
      pkg-config libprotobuf-c++ protobuf-compiler libprotobuf-dev python-pip python3-pip
}

setup_debian() {
  setup_ubuntu
}

setup_fedora() {
  dnf update -y
  dnf install -y rpm-build gcc-c++ make protobuf-devel autoconf automake libtool pkg-config python-pip python-devel
}

build_generic() {
  ./autogen.sh
  ./configure
  make -j2
  make check -j2
}

build_python() {
  make install
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
  ldconfig -v
  cd python
  python setup.py test
  cd ..
}

build_linux_gcc_coverall_ubuntu() {
  setup_debian
  pip install cpp-coveralls
  pip install 'requests[security]'
  build_generic
  build_python
  make distclean
  ./configure --enable-gcov
  make check -j2
  coveralls --exclude-pattern '.*(include|usr|test|third_party|pb|_main).*' --gcov-options '\-lp' --gcov gcov
}

build_linux_gcc_ubuntu() {
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
#  for v in 3.9 4.0 5.0 6.0; do
  for v in 3.9 6.0; do
    apt-get install -y clang-${v}
    export CXX="clang++-${v}" CC="clang-${v}"
    build_generic
    make distclean
  done
}

build_osx() {
  brew update
  brew install protobuf || brew link --overwrite protobuf
  brew link --force python@2
  brew upgrade python
  build_generic
  make install
  cd python
  python setup.py test
  python setup.py clean
  /usr/local/bin/python setup.py test
  /usr/local/bin/python setup.py clean
  /usr/local/bin/python2 setup.py test
  /usr/local/bin/python2 setup.py clean
  cd ..
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
