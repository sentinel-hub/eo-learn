"""
A script for installing all subpackages at once
"""

import sys
import subprocess


SUBPACKAGE_LIST = ['core',
                   'coregistration',
                   'features',
                   'geometry',
                   'io',
                   'mask',
                   'ml_tools']


def pip_command(name, args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + args + ['./{}'.format(name)])


if __name__ == '__main__':
    for subpackage in SUBPACKAGE_LIST:
        pip_command(subpackage, sys.argv[1:])
