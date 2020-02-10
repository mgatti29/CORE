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
path_fits = "/global/homes/m/mgatti/clustering-z/cosmosis_nonlimber/run_cosmosis_20_rmg.fits"
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis_nonlimber/out_20_rmg/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
write_values(0,folder_run)
write_options(path_fits,folder_run)

#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(folder_run))

"""

# run theory with true binning eboss *************
path_fits = "/global/homes/m/mgatti/clustering-z/cosmosis/run_cosmosis_20_eboss.fits"
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/out_20_eboss/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
write_values(0,folder_run)
write_options(path_fits,folder_run)

#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(folder_run))





# run hyperrank ***********************************************

number_of_rel = 300
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/hyperrank/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
    
for i in range(number_of_rel+1):
    try:
        folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/hyperrank/out_{0}".format(i)

        
        if not os.path.exists(folder_run):
            os.mkdir(folder_run)
        write_values(0,folder_run)
        write_options(path_fits,folder_run,nz_source='nz_sources_realisation_{0}'.format(i))

        #os.system("source config/setup-cosmosis-nersc")
        os.system("cosmosis {0}/demo17.ini".format(folder_run))

    except:
        print( "failed run {0}".format(i))

        
        
# run theory with redmagic binning  *************
path_fits = "/global/homes/m/mgatti/clustering-z/cosmosis/run_cosmosis_20_rmg.fits"
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/out_20_rmg/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
write_values(0,folder_run)
write_options(path_fits,folder_run)

#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(folder_run))



# run theory with true binning  *************
path_fits = "/global/homes/m/mgatti/clustering-z/cosmosis/run_cosmosis_20.fits"
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/out_20/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
write_values(0,folder_run)
write_options(path_fits,folder_run)

#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(folder_run))



# mag mag only run *************
path_fits = "/global/homes/m/mgatti/clustering-z/cosmosis/run_cosmosis_20_mm.fits"
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/out_20_mm/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
write_values(0,folder_run)
write_options(path_fits,folder_run)

#os.system("source ./cosmosis/cosmosis/config/setup-cosmosis-nersc")
os.system("cosmosis {0}/demo17.ini".format(folder_run))












# run theory with varying cosmological parameters *************

number_of_rel = 100
folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/multi/"

if not os.path.exists(folder_run):
    os.mkdir(folder_run)
    
for i in range(number_of_rel+1):
    try:
        folder_run = "/global/homes/m/mgatti/clustering-z/cosmosis/multi/out_{0}".format(i)

        
        if not os.path.exists(folder_run):
            os.mkdir(folder_run)
        write_values(0,folder_run)
        write_options(path_fits,folder_run)

        #vary input parameters according to Y1 error bars.
        om_0 = 0.286
        s8_0 = 0.82

        cov = np.array([[0.025,0],[0,0.025]])**2
        om,s8 = np.random.multivariate_normal(np.array([om_0,s8_0]),cov)
        ns = np.array(0.87 + 1*np.random.randint(0,100,1)/100.*0.2)
        h0 = np.array(0.55 + 1*np.random.randint(0,100,1)/100.*0.36)
        ob = np.array(0.03 + 1*np.random.randint(0,100,1)/100.*0.09)


        write_params_v(om,h0[0],ob[0],s8,ns[0],folder_run)


        #os.system("source config/setup-cosmosis-nersc")
        os.system("cosmosis {0}/demo17.ini".format(folder_run))

    except:
        print( "failed run {0}".format(i))



"""