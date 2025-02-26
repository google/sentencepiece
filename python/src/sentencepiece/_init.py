"""
SentencePiece Python Module Initialization
This file handles the proper initialization sequence for the SentencePiece module.
"""
import os
import sys
from pathlib import Path

def initialize_module():
    """Initialize the SentencePiece module by setting up the proper import paths."""
    # Add the directory containing _sentencepiece to Python path if needed
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    # Set LD_LIBRARY_PATH for Linux systems if needed
    if sys.platform.startswith('linux'):
        lib_path = os.environ.get('LD_LIBRARY_PATH', '')
        if str(module_dir) not in lib_path:
            os.environ['LD_LIBRARY_PATH'] = f"{module_dir}:{lib_path}"

# Initialize the module when imported
initialize_module()
