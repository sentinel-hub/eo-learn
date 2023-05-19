"""
A script for installing all subpackages at once

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import subprocess
import sys

SUBPACKAGE_LIST = [
    "core",
    "coregistration",
    "features",
    "geometry",
    "io[METEOBLUE]",
    "mask",
    "ml_tools",
    "visualization",
]


def pip_command(name, args):
    args = [arg for arg in args if not arg.startswith(".")]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args, f"./{name}"])


if __name__ == "__main__":
    for subpackage in SUBPACKAGE_LIST:
        pip_command(subpackage, sys.argv[1:])
