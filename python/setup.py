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

import codecs
import glob
import os
import pathlib
import platform
import string
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

sys.path.append(os.path.join('.', 'test'))


def long_description():
  with codecs.open('README.md', 'r', 'utf-8') as f:
    long_description = f.read()
  return long_description


exec(open('src/sentencepiece/_version.py').read())

_GITHUB_REF_NAME = os.environ.get('GITHUB_REF_NAME', __version__)


def get_cflags_and_libs(root, pkgcfg_dir):
  # TODO: We'd really like to use pkg-config here, but the GitHub action
  # runners of pkg-config seems old and interacts poorly with absl, causing
  # timeouts: https://github.com/pkgconf/pkgconf/issues/229
  cflags = ['-std=c++17', f'-I{root}/include']
  lib_dir = os.path.join(root, 'lib')
  try:
    libs = [
      os.path.join(lib_dir, f) for f in os.listdir(lib_dir)
      if f.endswith('.a')
    ]
  except:
    libs = []
  return cflags, libs


class build_ext(_build_ext):
  """Override build_extension to run cmake."""

  def build_extension(self, ext):
    root = './build/root'
    pkgcfg_dir = os.path.join(root, 'lib/pkgconfig')
    cflags, libs = get_cflags_and_libs(root, pkgcfg_dir)

    print(f'Found {libs=}')
    if not libs:
      cmd = ['./build_bundled.sh', _GITHUB_REF_NAME]
      print(f'Running {cmd=}')
      subprocess.check_call(cmd)
      cflags, libs = get_cflags_and_libs(root, pkgcfg_dir)
      if sys.platform == 'linux':
        # TODO: Remove these linker flags once pkg-config can be used to return
        # the correct library ordering.
        libs = ['-Wl,--start-group'] + libs + ['-Wl,--end-group']

    # Fix compile on some versions of Mac OSX
    # See: https://github.com/neulab/xnmt/issues/199
    if sys.platform == 'darwin':
      cflags.append('-mmacosx-version-min=10.9')
    else:
      if sys.platform == 'aix':
        cflags.append('-Wl,-s')
        libs.append('-Wl,-s')
      else:
        cflags.append('-Wl,-strip-all')
        libs.append('-Wl,-strip-all')
    if sys.platform == 'linux':
      libs.append('-Wl,-Bsymbolic')
    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    ext.extra_compile_args = cflags
    ext.extra_link_args = libs
    _build_ext.build_extension(self, ext)


def get_win_arch():
  arch = 'win32'
  if sys.maxsize > 2**32:
    arch = 'amd64'
  if 'arm' in platform.machine().lower():
    arch = 'arm64'
  if os.getenv('PYTHON_ARCH', '') == 'ARM64':
    # Special check for arm64 under ciwheelbuild, see https://github.com/pypa/cibuildwheel/issues/1942
    arch = 'arm64'
  return arch


if os.name == 'nt':
  # Must pre-install sentencepice into build/ directory.
  arch = get_win_arch()

  def _win_cflags_and_libs(path):
    if not os.path.exists(path):
      return [], []
    else:
      include_dir = path / 'include'
      cflags = ['/std:c++17', f'/I{include_dir}']
      lib_dir = path / 'lib'
      print(f'{os.listdir(lib_dir)=}')
      libs = [
          str(lib_dir / file)
          for file in os.listdir(lib_dir)
          if file.endswith('.lib')
      ]
      return cflags, libs

  cflags, libs = _win_cflags_and_libs(
      pathlib.PurePath('..', 'build', f'root_{arch}')
  )
  if not cflags:
    cflags, libs = _win_cflags_and_libs(pathlib.PurePath('..', 'build', 'root'))
  if not cflags:
    # build library locally with cmake and vc++.
    cmake_arch = 'Win32'
    if arch == 'amd64':
      cmake_arch = 'x64'
    elif arch == 'arm64':
      cmake_arch = 'ARM64'
    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    subprocess.check_call([
        'cmake',
        'sentencepiece',
        '-A',
        cmake_arch,
        '-B',
        'build',
        '-DSPM_ENABLE_SHARED=OFF',
        '-DCMAKE_INSTALL_PREFIX=build\\root',
    ])
    subprocess.check_call([
        'cmake',
        '--build',
        'build',
        '--config',
        'Release',
        '--target',
        'install',
        '--parallel',
        '8',
    ])
    cflags, libs = _win_cflags_and_libs(pathlib.PurePath('.', 'build', 'root'))

  print('## cflags={}'.format(' '.join(cflags)))
  print('## libs={}'.format(' '.join(libs)))
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'],
      extra_compile_args=cflags,
      extra_link_args=libs,
  )
  cmdclass = {}
else:
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'],
  )
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
        'sentencepiece/__init__',
        'sentencepiece/_version',
        'sentencepiece/sentencepiece_model_pb2',
        'sentencepiece/sentencepiece_pb2',
    ],
    ext_modules=[SENTENCEPIECE_EXT],
    cmdclass=cmdclass,
    install_requires=['protobuf~=6.30.2'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
