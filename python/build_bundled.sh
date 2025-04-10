#!/bin/sh

set -x

REF_NAME="${1}"

mkdir -p build

BUILD_DIR=./build
INSTALL_DIR=./build/root

if [ -f ./sentencepiece/src/CMakeLists.txt ]; then
  SRC_DIR=./sentencepiece
elif [ -f ../src/CMakeLists.txt ]; then
  SRC_DIR=..
else
  # Try taged version. Othewise, use head.
  echo -e "\033[0;35mSource not found, cloning from github...\033[0m"
  git clone https://github.com/google/sentencepiece.git --depth 1
  pushd sentencepiece
  git checkout ${REF_NAME}
  if [[ $? > 0 ]]; then
    echo -e "\033[0;35mRef named ${REF_NAME} not found, trying it as a pull request...\033[0m"
    PR_NUM=${REF_NAME%/merge}
    git pull origin pull/${PR_NUM}/head:pr_${PR_NUM} && git checkout -q pr_${PR_NUM}
  fi
  git submodule init && git submodule update
  echo -e "\033[0;35mWorking from ref:\033[0m"
  git --no-pager log -n 1 --decorate=short --pretty=oneline
  popd
  SRC_DIR=./sentencepiece
fi
echo -e "\033[0;32mSource at ${SRC_DIR}\033[0m"

cmake ${SRC_DIR} -B ${BUILD_DIR} -DBUILD_SHARED_LIBS=OFF -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
cmake --build ${BUILD_DIR} --config Release --target install --parallel $(nproc)
