"""
A script for installing all subpackages at once
"""

import sys
import subprocess


SUBPACKAGE_LIST = [
    "core",
    "coregistration",
    "features",
    "geometry",
    "io[METEOBLUE]",
    "mask",
    "ml_tools",
    "visualization[HVPLOT]",
]


def pip_command(name, args):
    args = [arg for arg in args if not arg.startswith(".")]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + args + [f"./{name}"])


if __name__ == "__main__":
    for subpackage in SUBPACKAGE_LIST:
        pip_command(subpackage, sys.argv[1:])
