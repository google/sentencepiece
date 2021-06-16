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

build_python() {
  VERSION=$1
  URL=$2
  INSTALL_PATH="/Library/Frameworks/Python.framework/Versions/${VERSION}/bin"
  CURRENT_PATH=${PATH}

  curl -L -o python.pkg ${URL}
  sudo installer -pkg python.pkg -target /

  if [ -f "${INSTALL_PATH}/python3" ]; then
    ln -s ${INSTALL_PATH}/python3        ${INSTALL_PATH}/python
    ln -s ${INSTALL_PATH}/python3-config ${INSTALL_PATH}/python-config
    ln -s ${INSTALL_PATH}/pip3           ${INSTALL_PATH}/pip
  fi

  export PATH="${INSTALL_PATH}:${CURRENT_PATH}"
  ls -l ${INSTALL_PATH}
  which python
  which pip
  python --version
  sudo python get-pip.py --no-setuptools --no-wheel --ignore-installed
  pip install --upgrade setuptools
  pip install wheel
  pip install delocate
  python setup.py clean
  python setup.py bdist_wheel --plat-name=macosx_10_6_x86_64
  python setup.py test
  delocate-listdeps dist/*.whl
  delocate-wheel -w dist/delocated_wheel dist/*.whl
  export PATH="${CURRENT_PATH}"

  ls -l dist/delocated_wheel
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

  # Install sentencepiece
  cmake ../.. -DSPM_ENABLE_SHARED=OFF -DSPM_NO_THREADLOCAL=ON
  make -j4 VERBOSE=1
  make install
  cd ..

  mkdir -p dist/delocated_wheel

#  build_python 2.7 https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.6.pkg
# latest pip doesn't support Py3.4
  # build_python 3.4 https://www.python.org/ftp/python/3.4.4/python-3.4.4-macosx10.6.pkg
  curl -L -O https://bootstrap.pypa.io/pip/3.5/get-pip.py
  build_python 3.5 https://www.python.org/ftp/python/3.5.4/python-3.5.4-macosx10.6.pkg

  curl -L -O https://bootstrap.pypa.io/get-pip.py
  build_python 3.6 https://www.python.org/ftp/python/3.6.6/python-3.6.6-macosx10.6.pkg
  build_python 3.7 https://www.python.org/ftp/python/3.7.9/python-3.7.9-macosx10.9.pkg
  build_python 3.8 https://www.python.org/ftp/python/3.8.6/python-3.8.6-macosx10.9.pkg
  build_python 3.9 https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg

  cd ..

  rm -fr build
}

build
