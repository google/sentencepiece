#!/bin/sh

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
  dnf install -y rpm-build gcc-c++ make protobuf-devel autoconf automake libtool pkg-config
}

build_generic() {
  ./autogen.sh
  ./configure
  make -j2
  make check -j2
}

build_linux_gcc_coverall_ubuntu() {
  setup_debian
  pip install cpp-coveralls
  pip install 'requests[security]'
  build_generic
  make distclean
  ./configure --enable-gcov
  make check -j2
  coveralls --exclude-pattern '.*(include|usr|test|third_party|pb|_main).*' --gcov-options '\-lp' --gcov gcov
}

build_linux_gcc_ubuntu() {
  setup_ubuntu
  build_generic
}

build_linux_gcc_debian() {
  setup_debian
  build_generic
}

build_linux_gcc_fedora() {
  setup_fedora
  build_generic
}

build_linux_clang_ubuntu() {
  setup_ubuntu
  for v in 3.9 4.0 5.0 6.0; do
    apt-get install -y clang-${v}
    export CXX="clang++-${v}" CC="clang-${v}"
    build_generic
    make distclean
  done
}

build_osx() {
  brew install protobuf autoconf automake libtool
  build_generic
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

