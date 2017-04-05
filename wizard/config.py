# TODO: License

"""
.. module:: config
"""

from __future__ import print_function
from os import path, makedirs
import numpy as np
import os

from .dataset import dataset
from .run_pairs import run_pairs
from .dndz import dndz
from .compare_multiplefiles_joint import compare

def wizard(config):
    """Run the wizard code, given a configuration dictionary

    Parameters
    ----------
    config : the dictionary of config files

    """

    # verbosity options *************************************
    import time
    import timeit
    if config['verbose']:
        time0 = timeit.default_timer()
    else:
        time0 = 0


    # mdoules ************************************************
    for entry in config['run']:
        #it creates the necessary folders
        make_directories(config)
        if entry == 'dataset':
            if time0:
                print('dataset module, elapsed time: {0} '.format(time.strftime('%H:%M:%S',time.gmtime(timeit.default_timer() - time0))))

                if config['verbose'] > 1:
                    time0i = timeit.default_timer()
                else:
                    time0i = 0



            dataset(time0=time0i, **config['dataset'])


        elif entry == 'run_pairs':
            if time0:
                print('run_pairs module, elapsed time: {0} '.format(time.strftime('%H:%M:%S',time.gmtime(timeit.default_timer() - time0))))
                if config['verbose'] > 1:
                    time0i = timeit.default_timer()
                else:
                    time0i = 0
            run_pairs(time0=time0i, **config['run_pairs'])


        elif entry == 'dndz':
            if time0:
                print('dndz module, elapsed time: {0} '.format(time.strftime('%H:%M:%S',time.gmtime(timeit.default_timer() - time0))))
                if config['verbose'] > 1:
                    time0i = timeit.default_timer()
                else:
                    time0i = 0
            dndz(time0=time0i, **config['dndz'])


        elif entry == 'compare':
            if time0:
                print('compare module, elapsed time: {0} '.format(time.strftime('%H:%M:%S',time.gmtime(timeit.default_timer() - time0))))
                if config['verbose'] > 1:
                    time0i = timeit.default_timer()
                else:
                    time0i = 0
            compare( **config['compare'])
    if time0:
        print('done', timeit.default_timer() - time0)

def read_config(file_name):
    """Read a configuration dictionary from a file

    :param file_name:   yaml file name which we read
    """
    import yaml
    with open(file_name) as f_in:
        config = yaml.load(f_in.read())
    return config

def save_config(config, file_name):
    """Take a configuration dictionary and save to file

    :param config:  Dictionary of configuration
    :param file_name: file we are saving to
    """
    import yaml
    with open(file_name, 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

def make_directories(config):
    # I have odified this, fixing the folders need to run wizard
    # In this way the program knows where to pick up stuff from each module
    # even if not all the modules are run at the same time

    if not os.path.exists('./pairscount/'):
        os.makedirs('./pairscount/')
    if not os.path.exists('./pairscount/pairs/'):
        os.makedirs('./pairscount/pairs/')
    if not os.path.exists('./pairscount/pairs_dist/'):
        os.makedirs('./pairscount/pairs_dist/')
    if not os.path.exists('./pairscount/data_plot/'):
        os.makedirs('./pairscount/data_plot/')
    if not os.path.exists('./pairscount/pairs_plot/'):
        os.makedirs('./pairscount/pairs_plot/')
    if not os.path.exists('./output_dndz/'):
        os.makedirs('./output_dndz/')

    if not path.exists('./compare/'):
        makedirs('./compare/')
#if not os.path.exists('./output_opt_2/output_NZ/'):
#     os.makedirs('./output_opt_2/output_NZ/')
#if not os.path.exists('./output_opt_2/output_best/'):
#     os.makedirs('./output_opt_2/output_best/')

def check_make(path_check):
    """
    Convenience routine to avoid that annoying 'can't make directory; already
    present!' error.
    """
    if not path.exists(path_check):
        makedirs(path_check)
