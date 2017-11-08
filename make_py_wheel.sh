#!/bin/sh

# Shell script to generate SentencePiece Python wrapper
# with manylinux1 docker environment.

# % docker pull quay.io/pypa/manylinux1_x86_64; docker pull quay.io/pypa/manylinux1_i686
# % mkdir docker
# % docker run --rm -ti -v `pwd`/docker:/docker  -w /docker quay.io/pypa/manylinux1_x86_64 bash
# git clone https://github.com/google/sentencepiece.git
# ./sentencepiece/make_py_wheel.sh
# twine sentencepiece/python/dist/wheelhouse/*.whl

export PATH="/usr/local/bin:$PATH"

wget http://ftpmirror.gnu.org/libtool/libtool-2.4.6.tar.gz
tar zxfv libtool-2.4.6.tar.gz
cd libtool-2.4.6
./configure
make
make install
cd ..

git clone https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make
make install
strip /usr/local/lib/*protobuf*
cd ..

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
./autogen.sh
grep -v PKG_CHECK_MODULES configure > tmp
mv tmp -f configure
chmod +x configure
LIBS+="-lprotobuf -pthread" ./configure
make -j4
make install
strip /usr/local/lib/*sentencepiece*

cd python
for i in /opt/python/*
do
 $i/bin/python setup.py bdist_wheel
 rm -fr build
done

cd dist
for i in *.whl
do
auditwheel repair $i
done
