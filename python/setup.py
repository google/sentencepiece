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

import glob
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
try:
  from pybind11.setup_helpers import Pybind11Extension
except ImportError:
  Pybind11Extension = None


with open('src/sentencepiece/_version.py') as f:
  line = f.readline().strip()
  __version__ = line.split('=')[1].strip().strip("'")


def is_gil_disabled():
  return sysconfig.get_config_var('Py_GIL_DISABLED')


def find_abseil_lib(search_root):
  print('## searching abseil and protobuf {}'.format(search_root))
  absl_libs = []
  ext = '.lib' if os.name == 'nt' else '.a'
  valid_prefixes = (
      'libabsl',
      'absl',
      'libprotobuf-lite',
      'protobuf-lite',
      'libutf8_validity',
      'utf8_validity',
  )
  for root, dirs, files in os.walk(search_root):
    for file in files:
      if file.startswith(valid_prefixes) and file.endswith(ext):
        full_path = os.path.join(root, file)
        if full_path not in absl_libs:
          absl_libs.append(full_path)

  print('## absl_libs={}'.format(' '.join(absl_libs)))
  return absl_libs


def get_build_includes(build_dir, is_msvc=True):
  prefix = '/I' if is_msvc else '-I'
  candidates = [
      os.path.join(build_dir, 'root', 'include'),
      os.path.join(build_dir, 'src'),
      os.path.join(build_dir, '_deps', 'protobuf-src', 'src'),
      os.path.join(build_dir, '_deps', 'protobuf-src', 'third_party', 'utf8_range'),
      os.path.join(build_dir, '_deps', 'abseil-cpp-src'),
      '../src',
      './sentencepiece/src',
      '../third_party/abseil-cpp',
      './sentencepiece/third_party/abseil-cpp',
      '..',
      './sentencepiece',
  ]
  includes = []
  seen = set()
  for p in candidates:
    norm = os.path.normpath(p)
    if norm not in seen and os.path.exists(norm):
      seen.add(norm)
      includes.append(prefix + norm)
  return includes


def get_cflags_and_libs(root):
  cflags = ['-std=c++20'] + get_build_includes(root, is_msvc=False)
  libs = []
  for lib_name in ['libsentencepiece.a', 'libsentencepiece_train.a']:
    for sub in ['lib', 'lib64', 'src', '']:
      p = os.path.join(root, sub, lib_name)
      if os.path.exists(p) and p not in libs:
        libs.append(p)
        break
  return cflags, libs


class build_ext_unix(_build_ext):
  """Override build_extension to run cmake."""

  def build_extension(self, ext):
    cflags, libs = get_cflags_and_libs('../build')
    abseil_libs = find_abseil_lib('../build')

    if len(libs) == 0:
      cflags, libs = get_cflags_and_libs('./build')
      abseil_libs = find_abseil_lib('./build')

    if len(libs) == 0:
      subprocess.check_call(['./build_bundled.sh'])
      cflags, libs = get_cflags_and_libs('./build')
      abseil_libs = find_abseil_lib('./build')

    # Fix compile on some versions of Mac OSX
    # See: https://github.com/neulab/xnmt/issues/199
    if sys.platform == 'darwin':
      #  non GNU linker
      cflags.append('-mmacosx-version-min=10.9')
      # get correct SDK path by xcrun
      sdk_path = (
          subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
      )
      libs.extend(['-stdlib=libc++', f'-isysroot{sdk_path}'])
      libs.extend(abseil_libs)
      libs.append('-Wl,-dead_strip')
      libs.append('-Wl,-exported_symbols_list,exports_mac.txt')
    else:
      # GNU linker
      libs.append('-Wl,--start-group')
      libs.extend(abseil_libs)
      libs.append('-lgomp')
      libs.append('-Wl,--end-group')
      libs.append('-Wl,--gc-sections')
      libs.append('-Wl,--version-script=exports.txt')
      if sys.platform == 'aix':
        cflags.append('-Wl,-s')
        libs.append('-Wl,-s')
      else:
        cflags.append('-Wl,-strip-all')
        libs.append('-Wl,-strip-all')

    if sys.platform == 'linux':
      libs.append('-Wl,-Bsymbolic')
      libs.append('-fopenmp')
      libs.append('-lgomp')
      cflags.append('-fopenmp')

    if is_gil_disabled():
      cflags.append('-DPy_GIL_DISABLED')

    cflags.append('-Wno-unused-function')

    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    ext.extra_compile_args = cflags
    ext.extra_link_args = libs
    _build_ext.build_extension(self, ext)


class build_ext_win(_build_ext):
  """Override build_extension to run cmake."""

  def build_extension(self, ext):
    # Must pre-install sentencepice into build directory.
    arch = get_win_arch()

    if os.path.exists('..\\build_{}\\root\\lib'.format(arch)):
      build_dir = '..\\build_{}'.format(arch)
    elif os.path.exists('..\\build\\root\\lib'):
      build_dir = '..\\build'
    elif os.path.exists('.\\build\\root\\lib'):
      build_dir = '.\\build'
    else:
      # build library locally with cmake and vc++.
      if arch == 'amd64':
        cmake_arch = 'x64'
      elif arch == 'arm64':
        cmake_arch = 'ARM64'
      else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

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
      build_dir = '.\\build'

    cflags = ['/std:c++17'] + get_build_includes(build_dir, is_msvc=True)
    libs = [
        os.path.join(build_dir, 'root', 'lib', 'sentencepiece.lib'),
        os.path.join(build_dir, 'root', 'lib', 'sentencepiece_train.lib'),
    ]
    libs.extend(find_abseil_lib(build_dir))

    # on Windows, GIL flag is not set automatically.
    # https://docs.python.org/3/howto/free-threading-python.html
    if is_gil_disabled():
      cflags.append('/DPy_GIL_DISABLED')

    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    ext.extra_compile_args = cflags
    ext.extra_link_args = libs
    _build_ext.build_extension(self, ext)


def copy_package_data():
  """Copies shared package data"""

  package_data = os.path.join('src', 'sentencepiece', 'package_data')

  if not os.path.exists(package_data):
    os.makedirs(package_data)

  if glob.glob(os.path.join(package_data, '*.bin')):
    return

  def find_targets(roots):
    for root in roots:
      data = glob.glob(os.path.join(root, '*.bin'))
      if data:
        return data
    return []

  data = find_targets([
      '../build/root/share/sentencepiece',
      './build/root/share/sentencepiece',
      '../data',
      './sentencepiece/data',
  ])

  for filename in data:
    print('## copying {} -> {}'.format(filename, package_data))
    shutil.copy(filename, package_data)


def get_win_arch():
  if sys.maxsize <= 2**32:
    raise RuntimeError("32-bit Windows (Win32) is not supported.")
  arch = 'amd64'
  if 'arm' in platform.machine().lower():
    arch = 'arm64'
  if os.getenv('PYTHON_ARCH', '') == 'ARM64':
    # Special check for arm64 under ciwheelbuild, see https://github.com/pypa/cibuildwheel/issues/1942
    arch = 'arm64'
  return arch


if Pybind11Extension:
  SENTENCEPIECE_EXT = Pybind11Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_pybind.cc'],
  )
else:
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_pybind.cc'],
  )


if os.name == 'nt':
  cmdclass = {'build_ext': build_ext_win}
else:
  cmdclass = {'build_ext': build_ext_unix}

if __name__ == '__main__':
  copy_package_data()
  setup(
      name='sentencepiece',
      package_dir={'': 'src'},
      py_modules=[
          'sentencepiece/__init__',
          'sentencepiece/_version',
          'sentencepiece/sentencepiece_model_pb2',
          'sentencepiece/sentencepiece_pb2',
      ],
      ext_modules=[SENTENCEPIECE_EXT],
      cmdclass=cmdclass,
  )
