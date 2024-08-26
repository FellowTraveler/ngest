# ngest/ngest/utils/clang_setup.py
#
# Copyright 2024 Chris Odom
# MIT License

import os
from clang.cindex import Config
import logging

def setup_clang(custom_path=None):
    possible_paths = [
        custom_path,
        "/opt/homebrew/opt/llvm/lib",  # macOS Homebrew path
        "/usr/lib/llvm-10/lib",        # Ubuntu/Debian path
        "/usr/lib64/llvm",             # Fedora/RHEL path
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            Config.set_library_path(path)
            logging.info(f"Clang library path set to: {path}")
            return True
    
    logging.warning("Could not find Clang library path. You may need to set it manually.")
    return False


