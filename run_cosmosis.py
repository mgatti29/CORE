import subprocess
import numpy as np
import os, sys
import sys

sys.path.insert(0, '/global/homes/m/mgatti/clustering-z/WZ_utils/')
from routines_cosmosis import *

import numpy as np

import os
import subprocess


# run theory with redmagic binning  *************
path_cosmosis ='/global/homes/m/mgatti/clustering-z/cosmosis/'
path_output = '/global/homes/m/mgatti/clustering-z/cosmosis_output/'
runs = ['sims_WL_eboss',
        'data_WL_eboss',
        'data_WL_rmg_rmgz_higherlum',
        'data_WL_rmg_rmgz_combined',
        'data_WL_rmg_rmgz_higherlum_data',
        'data_WL_rmg_rmgz_combined_data',
        'data_WL_rmg_truez',
        'sims_WL_rmg_rmgz_higherlum',
        'sims_WL_rmg_rmgz_combined',
        'sims_WL_rmg_truez',
        'sims_WL_rmg_rmgz_higherlum_data',
        'sims_WL_rmg_rmgz_combined_data',
        ]


for run in runs:
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    if not os.path.exists(path_output+run):
        os.mkdir(path_output+run)
    write_values(0,path_output+run)
    write_options(path_cosmosis+run+'.fits',path_output+run)

    #os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
    os.system("cosmosis {0}/demo17.ini".format(path_output+run))


  
# wrong cosmological parameters *********************

run = 'sims_WL_rmg_truez'
if not os.path.exists(path_output):
    os.mkdir(path_output)
if not os.path.exists(path_output+run+'_cosmo_wrong'):
    os.mkdir(path_output+run+'_cosmo_wrong')

write_options(path_cosmosis+run+'.fits',path_output+run+'_cosmo_wrong')
write_values(0,path_output+run+'_cosmo_wrong')
write_params_v(0.35,0.77,0.01,0.72,1.,path_output+run+'_cosmo_wrong')
#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(path_output+run+'_cosmo_wrong'))

    