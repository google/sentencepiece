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
import string
import subprocess
import sys

sys.path.append('./test')

with open("README.md") as f:
    long_description = f.read()

def cmd(line):
    try:
        output = subprocess.check_output(line, shell=True)
        if sys.version_info >= (3,0,0):
            output = output.decode('utf-8')
    except subprocess.CalledProcessError:
        sys.stderr.write('Failed to find sentencepiece pkgconfig\n')
        sys.exit(1)
    return output.strip().split()

# Fix compile on some versions of Mac OSX
# See: https://github.com/neulab/xnmt/issues/199
extra_compile_args = ["-std=c++11"]
if sys.platform == "darwin":
  extra_compile_args.append("-mmacosx-version-min=10.9")

setup(name = 'sentencepiece',
      author = 'Taku Kudo',
      author_email='taku@google.com',
      description = 'SentencePiece python wrapper',
      long_description = long_description,
      version='0.1.3',
      url = 'https://github.com/google/sentencepiece',
      license = 'Apache',
      platforms = 'Unix',
      py_modules=['sentencepiece'],
      ext_modules = [Extension('_sentencepiece',
                               sources=['sentencepiece_wrap.cxx'],
                               extra_compile_args=extra_compile_args +
                               cmd('pkg-config sentencepiece --cflags'),
                               extra_link_args=cmd('pkg-config sentencepiece --libs'))
                     ],
      classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ],
      test_suite = 'sentencepiece_test.suite')
