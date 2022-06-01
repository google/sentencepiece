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
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
import codecs
import string
import subprocess
import sys
import os

sys.path.append(os.path.join('.', 'test'))


def long_description():
  with codecs.open('README.md', 'r', 'utf-8') as f:
    long_description = f.read()
  return long_description


exec(open('src/sentencepiece/_version.py').read())


def run_pkg_config(section, pkg_config_path=None):
  try:
    cmd = 'pkg-config sentencepiece --{}'.format(section)
    if pkg_config_path:
      cmd = 'env PKG_CONFIG_PATH={} {}'.format(pkg_config_path, cmd)
    output = subprocess.check_output(cmd, shell=True)
    if sys.version_info >= (3, 0, 0):
      output = output.decode('utf-8')
  except subprocess.CalledProcessError:
    sys.stderr.write('Failed to find sentencepiece pkg-config\n')
    sys.exit(1)
  return output.strip().split()


def is_sentencepiece_installed():
  try:
    subprocess.check_call('pkg-config sentencepiece --libs', shell=True)
    return True
  except subprocess.CalledProcessError:
    return False


class build_ext(_build_ext):
  """Override build_extension to run cmake."""

  def build_extension(self, ext):
    cflags = ['-std=c++11']
    if os.path.exists('./bundled') or not is_sentencepiece_installed():
      # Build sentencepiece from scratch with build_bundled.sh
      # This is useally called as a fallback of pip command.
      if not os.path.exists('./bundled'):
        subprocess.check_call(['./build_bundled.sh', __version__])
      cflags = cflags + ['-I./bundled/include']
      libs = ['-L./bundled/lib' '-lsentencepiece', '-lsentencepiece_train']
    elif os.path.exists('../build/root'):
      cflags = cflags + ['-I../build/root/include']
      libs = ['-L../build/root/lib' '-lsentencepiece', '-lsentencepiece_train']
    else:
      cflags = cflags + run_pkg_config('cflags')
      libs = run_pkg_config('libs')

    # Fix compile on some versions of Mac OSX
    # See: https://github.com/neulab/xnmt/issues/199
    if sys.platform == 'darwin':
      cflags.append('-mmacosx-version-min=10.9')
    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    ext.extra_compile_args = cflags
    ext.extra_link_args = libs
    _build_ext.build_extension(self, ext)


if os.name == 'nt':
  # Must pre-install sentencepice into bundled directory.
  cflags = ['/MT', '/I.\\bundled\\include']
  libs = [
      '.\\bundled\\lib\\sentencepiece.lib',
      '.\\bundled\\lib\\sentencepiece_train.lib'
  ]
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'],
      extra_compile_args=cflags,
      extra_link_args=libs)
  cmdclass = {}
else:
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'])
  cmdclass = {'build_ext': build_ext}

setup(
    name='sentencepiece',
    author='Taku Kudo',
    author_email='taku@google.com',
    description='SentencePiece python wrapper',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    version=__version__,
    package_dir={'': 'src'},
    url='https://github.com/google/sentencepiece',
    license='Apache',
    platforms='Unix',
    py_modules=[
        'sentencepiece/__init__', 'sentencepiece/_version',
        'sentencepiece/sentencepiece_model_pb2',
        'sentencepiece/sentencepiece_pb2'
    ],
    ext_modules=[SENTENCEPIECE_EXT],
    cmdclass=cmdclass,
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
