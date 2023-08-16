"""
A script for installing all subpackages at once

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sys.argv[1:], ".[ALL]"])
    warnings.warn(
        "Installing via `install_all.py` is no longer necessary and has been deprecated. Use `pip install"
        " eo-learn[ALL]` instead."
    )
