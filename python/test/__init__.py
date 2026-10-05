import glob
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'data'))
if os.path.isdir(_DATA_DIR):
  for _src in glob.glob(os.path.join(_DATA_DIR, 'botchan*')):
    _dst = os.path.join(_HERE, os.path.basename(_src))
    if not os.path.exists(_dst):
      shutil.copy2(_src, _dst)
