#!/usr/bin/env python

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

setup(name = 'sentencepiece',
      author = 'Taku Kudo',
      author_email='taku@google.com',
      description = 'SentencePiece python wrapper',
      long_description = long_description,
      version='0.0.7',
      url = 'https://github.com/google/sentencepiece',
      license = 'Apache',
      platforms = 'Unix',
      py_modules=['sentencepiece'],
      ext_modules = [Extension('_sentencepiece',
                               sources=['sentencepiece_wrap.cxx'],
                               extra_compile_args=['-std=c++11'] +
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
