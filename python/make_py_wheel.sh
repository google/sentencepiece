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
  docker run --rm -ti --name py_sentencepiece \
    -v `pwd`/../:/sentencepiece -w /sentencepiece/python \
    -td $1 /bin/bash
  docker exec py_sentencepiece bash -c "./make_py_wheel.sh native $2"
  docker stop py_sentencepiece
}

build() {
  TRG=$1
  rm -fr build
  mkdir -p build
  cd build

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF
  make -j4
  make install
  cd ..

  for i in /opt/python/*
  do
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib
    $i/bin/python setup.py clean
    $i/bin/python setup.py bdist
    strip build/*/*/*.so
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

  cd ..
  rm -fr build
}

if [ "$1" = "native" ]; then
  build $2
elif [ "$#" -eq 1 ]; then
  run_docker quay.io/pypa/manylinux2014_${1}  ${1}
else
  run_docker quay.io/pypa/manylinux2014_i686 i686
  run_docker quay.io/pypa/manylinux2014_x86_64 x86_64
fi
