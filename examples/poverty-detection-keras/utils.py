
"""
utils.py

@author: Development Seed

Utility functions for printing training details
"""
import shutil
import pprint

import numpy as np
import matplotlib as mpl
from keras.models import model_from_yaml
# from pygeotile.tile import Tile

import train_config_pd as tcf


def print_start_details(start_time):
    """Print config at the start of a run."""
    pp = pprint.PrettyPrinter(indent=4)

    print('Start time: ' + start_time.strftime('%d/%m %H:%M:%S'))

    print('\nDatasets used:')
    pp.pprint(tcf.data_dir)
    print('\nTraining details:')
    pp.pprint(tcf.train_params)
    print('\nModel details:')
    pp.pprint(tcf.model_params)
    print('\n\n' + '=' * 40)


def print_end_details(start_time, end_time):
    """Print runtime information."""
    run_time = end_time - start_time
    hours, remainder = divmod(run_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print('\n\n' + '=' * 40)
    print('End time: ' + end_time.strftime('%d/%m %H:%M:%S'))
    print('Total runtime: %i:%02i:%02i' % (hours, minutes, seconds))