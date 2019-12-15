#!/usr/bin/env python

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

from setuptools import setup, Extension
import codecs
import string
import subprocess
import sys
import os

sys.path.append(os.path.join('.', 'test'))

with codecs.open('README.md', 'r', 'utf-8') as f:
  long_description = f.read()

with codecs.open('VERSION', 'r', 'utf-8') as f:
  version = f.read().rstrip()


def cmd(line):
  try:
    output = subprocess.check_output(line, shell=True)
    if sys.version_info >= (3, 0, 0):
      output = output.decode('utf-8')
  except subprocess.CalledProcessError:
    sys.stderr.write('Failed to find sentencepiece pkgconfig\n')
    sys.exit(1)
  return output.strip().split()


# Fix compile on some versions of Mac OSX
# See: https://github.com/neulab/xnmt/issues/199
def cflags():
  if sys.platform == 'win32':
    return ['/MT', '/I..\\build\\root\\include']
  args = ['-std=c++11']
  if sys.platform == 'darwin':
    args.append('-mmacosx-version-min=10.9')
  args = args + cmd('pkg-config sentencepiece --cflags')
  return args


def libs():
  if sys.platform == 'win32':
    return [
        '..\\build\\root\\lib\\sentencepiece.lib',
        '..\\build\\root\\lib\\sentencepiece_train.lib'
    ]

  return cmd('pkg-config sentencepiece --libs')


setup(
    name='sentencepiece',
    author='Taku Kudo',
    author_email='taku@google.com',
    description='SentencePiece python wrapper',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=version,
    url='https://github.com/google/sentencepiece',
    license='Apache',
    platforms='Unix',
    py_modules=['sentencepiece'],
    ext_modules=[
        Extension(
            '_sentencepiece',
            sources=['sentencepiece_wrap.cxx'],
            extra_compile_args=cflags(),
            extra_link_args=libs())
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable', 'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix', 'Programming Language :: Python',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    test_suite='sentencepiece_test.suite')
