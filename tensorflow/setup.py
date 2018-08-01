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

from setuptools import setup
from setuptools import find_packages
import string
import sys

sys.path.append('./test')

setup(name = 'tf_sentencepiece',
      author = 'Taku Kudo',
      author_email='taku@google.com',
      description = 'SentencePiece Encode/Decode ops for TensorFlow',
      version='0.1.3',
      url = 'https://github.com/google/sentencepiece',
      license = 'Apache',
      platforms = 'Unix',
      packages=find_packages(exclude=['test']),
      package_data={'tf_sentencepiece':  ['_sentencepiece_processor_ops.so']},
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
      keywords='tensorflow machine learning sentencepiece NLP segmentation',
      test_suite = 'tf_sentencepiece_test.suite')
