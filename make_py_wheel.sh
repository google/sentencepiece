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

# Usage:
# > sudo sh make_py_wheel.sh
# wheel packages are built under <pwd>/manylinux_wh dir

set -e  # exit immediately on error
set -x  # display all commands

run_docker() {
  rm -fr manylinux_wh/$2
  mkdir -p manylinux_wh/$2
  docker pull "$1"
  docker run --rm -ti --name manylinux -v `pwd`:/sentencepiece -w /sentencepiece/manylinux_wh/$2 -td "$1" /bin/bash
  docker exec manylinux bash -c "../../make_py_wheel.sh make_wheel $2"
  docker stop manylinux
}

make_wheel() {
  export PATH="/usr/local/bin:$PATH"
  TRG=$1

  wget http://ftpmirror.gnu.org/libtool/libtool-2.4.6.tar.gz
  tar zxfv libtool-2.4.6.tar.gz
  cd libtool-2.4.6
  ./configure
  make -j4
  make install
  cd ..

  git clone https://github.com/google/protobuf.git
  cd protobuf
  ./autogen.sh
  ./configure --disable-shared --with-pic
  make -j4
  make install
  cd ..

  cd ../../
  make distclean || true
  ./autogen.sh
  grep -v PKG_CHECK_MODULES configure > tmp
  mv tmp -f configure
  chmod +x configure
  LIBS+="-pthread -L/usr/local/lib -lprotobuf" ./configure --disable-shared --with-pic
  make -j4
  make install

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

  mv -f wheelhouse/*${TRG}.whl ../../manylinux_wh
}

if [ "$#" -eq 2 ]; then
  eval "$1" $2
elif [ "$#" -eq 1 ]; then
  run_docker quay.io/pypa/manylinux1_${1}  ${1}
else
  run_docker quay.io/pypa/manylinux1_i686   i686
  run_docker quay.io/pypa/manylinux1_x86_64 x86_64
fi
