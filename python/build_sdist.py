#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

# Copy C++ source files from parent directory to ./sentencepiece/
targets = [
    'CMakeLists.txt',
    'LICENSE',
    'README.md',
    'VERSION.txt',
    'cmake',
    'config.h.in',
    'sentencepiece.pc.in',
    'src',
    'third_party',
    'data',
]

os.makedirs('sentencepiece', exist_ok=True)

for item in targets:
  src = os.path.join('..', item)
  dst = os.path.join('sentencepiece', item)
  if os.path.lexists(src):
    if os.path.islink(src):
      # Skip build-generated symlinks (e.g. third_party/absl)
      continue
    print(f'copying {src} -> {dst}')
    if os.path.isdir(src):
      shutil.copytree(
          src,
          dst,
          dirs_exist_ok=True,
          ignore=shutil.ignore_patterns('absl', '*.pyc', '__pycache__'),
      )
    else:
      shutil.copy2(src, dst)

python_exe = sys.executable
res = subprocess.run([python_exe, '-m', 'build', '--sdist'], check=False)
if res.returncode != 0:
  subprocess.check_call([python_exe, 'setup.py', 'sdist'])

