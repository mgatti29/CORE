import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import beta
import scipy.integrate as pyint
import matplotlib.transforms as mtransforms
import matplotlib as mpl
import os
import scipy.optimize as optimization
from scipy.optimize import fmin as simplex
import time
from numpy import matrix
from numpy import linalg
import matplotlib.mlab as mlab
import math
import astropy
from astropy.cosmology import WMAP9 as WMAP9
from astropy.cosmology import FlatLambdaCDM
from functions_nz import *
from scale_optimization import *
from regularization import *
import sys
import pyfits as pf
import pandas as pd
import timeit
import shutil
import copy
from astropy.table import Table, vstack, hstack

def dndz(methods=['Menard_physical_scales'],
        photo_z_column=None,
        bias_correction_reference_Menard='None',bias_correction_Newman=0,bias_correction_reference_Schmidt='None',
        bias_correction_unknown_Menard='None',bias_correction_unknown_Schmidt='None',
        use_physical_scale_Newman=False,
        weight_variance=False,pairs_weighting=False,
        fit_free=True,
        gamma=1.,
        optimization=False,
        resampling= 'bootstrap',
        number_of_bootstrap= 2000,
        resampling_pairs= 'False',
        plot_compare_all=False,
        Nbins=[8], interval_width=3,step_width=6,
        show_fit=False,
        z_min=None,z_max=None,
        mcmc_negative=False,
        only_diagonal=True,
        verbose=False,
        sum_schmidt=False,
        just_pickup_best_scales=False,
        regularization= False,
        prior_gaussian_process='None',
        fit='None',
        set_negative_to_zero='None',
        w_estimator='LS',
        bounds_CC_fit= ([-1,-0.01],[1.,0.01]),
        initial_guess_CC_fit=[0.01,0.01],
        bounds_AC_U_fit= ([0.01,1,-0.01],[1.,3,0.01]),
        initial_guess_AC_U_fit=([0.01,2,0.01]),
        bounds_AC_R_fit=([0.01,-1,-0.01],[200.,3,0.01]),
        initial_guess_AC_R_fit=([2,1.5,0.01]),
        tomo_bins='ALL',
        time0=0.):



    if time0>0:
        verbose=True
    else:
        verbose=False


    # make folders for the tomobins *********************************************
    unknown_bins_interval=load_obj('./pairscount/unknown_bins_interval')
    for i in range(unknown_bins_interval['z'].shape[0]):
        if not os.path.exists('./output_dndz/TOMO_{0}/'.format(i+1)):
            os.makedirs(('./output_dndz/TOMO_{0}/'.format(i+1)))
            if not os.path.exists('./output_dndz/TOMO_{0}/Nz/'.format(i+1)):
                os.makedirs('./output_dndz/TOMO_{0}/Nz/'.format(i+1))
            if not os.path.exists('./output_dndz/TOMO_{0}/best_Nz/'.format(i+1)):
                os.makedirs('./output_dndz/TOMO_{0}/best_Nz/'.format(i+1))
            if not os.path.exists('./output_dndz/TOMO_{0}/fit/'.format(i+1)):
                os.makedirs('./output_dndz/TOMO_{0}/fit/'.format(i+1))
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    # load redshift arrays and impose external redshift cuts. *******************
    reference_bins_interval=cut_redshift_range(z_min,z_max,tomo_bins)



    #loading the true distribution *************************
    N=load_true_distribution(reference_bins_interval,tomo_bins,photo_z_column)

    #  estimate number of iterations for each method: ***************
    totiter=number_of_iterations(Nbins,optimization,methods,interval_width,step_width)

    iter=0

    #*******************************************************************
    # create a dictionary to keep the relevant parameters of each method.
    best_params=create_dictionary_bestscales(methods,weight_variance,pairs_weighting, bias_correction_reference_Menard,bias_correction_Newman,
                        bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,fit_free,sum_schmidt)



    ###################################################################################
    #                                                                                 #
    #                                 MAIN CYCLE                                      #
    #                                                                                 #
    ###################################################################################


    if just_pickup_best_scales:
        pass
    else:


        # cycle over the possible binnings **************************************
        for nnn in range(len(Nbins)):

                if not optimization:
                    # if optimization is set to false , for each angular bins it just run considering
                    # the full angular/physical distance range.
                    interval_bins=Nbins[nnn]
                    step=1
                else:
                    interval_bins=int(max([math.ceil(Nbins[nnn]/2.),interval_width]))
                    #interval_bins=int(max([interval_width,interval_width]))
                    step=int(math.ceil(Nbins[nnn]/step_width))

                update_progress(0.)
                start=time.time()
                for thetmax in range(interval_bins,Nbins[nnn]+1,step):
                    for thetmin in range(0,thetmax-interval_bins+1,step):


                        # READING FILES AND Nz COMPUTATION ****************************

                        # Reading files and create resamples ******************************************
                        correlation,methods,jk_r=load_w_treecorr(methods,N,reference_bins_interval,
                                        Nbins[nnn],thetmin,thetmax,bias_correction_reference_Menard,bias_correction_Newman,
                                        bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,sum_schmidt,
                                        use_physical_scale_Newman,w_estimator,resampling,resampling_pairs,number_of_bootstrap,verbose)


                        update_progress((0.2+float(iter))/totiter,timeit.default_timer(),start)

                        # Optimization of scales. **************************************
                        # For each method, for each redshift bin, it integrates over the
                        # correlation signal. If required, it also perform the fits
                        correlation_optimized=optimize(correlation,methods,N,reference_bins_interval,bias_correction_reference_Menard,bias_correction_Newman,
                                            bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,weight_variance,
                                            pairs_weighting,fit_free,use_physical_scale_Newman,bounds_CC_fit,initial_guess_CC_fit,bounds_AC_U_fit,
                                            initial_guess_AC_U_fit,bounds_AC_R_fit,initial_guess_AC_R_fit,verbose,gamma)


                        update_progress((0.6+float(iter))/totiter,timeit.default_timer(),start)


                        # Save fit  ****************************************************
                        # It saves the results of the fitting (Newman and Schmidt methods)
                        if show_fit and ('Newman' in methods):
                            label_save='{0}_{1}_{2}'.format(thetmin,thetmax,Nbins[nnn])
                            save_fit(N,reference_bins_interval,correlation_optimized,correlation,label_save,verbose)

                        update_progress((0.8+float(iter))/totiter,timeit.default_timer(),start)


                        # Compute N(z) *************************************************
                        Nz_tomo,BNz_tomo=compute_Nz(correlation_optimized,jk_r,methods,bias_correction_reference_Menard,bias_correction_Newman,
                                            bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,
                                                N,reference_bins_interval,use_physical_scale_Newman,verbose)


                        update_progress((0.9+float(iter))/totiter,timeit.default_timer(),start)

                        # Stacking *****************************************************
                        # it normalizes
                        #print (Nz_tomo)
                        Nz,BNz=stacking(Nz_tomo,BNz_tomo,jk_r,N)

                        #Plotting and compute statistics.  ***********************************************
                        for method in Nz.keys():

                            label,label_bias= define_label(method,fit_free,sum_schmidt,weight_variance,pairs_weighting,bias_correction_reference_Menard,bias_correction_Newman,
                                                bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt)

                            if resampling_pairs:
                                resp='pairs'
                            else:
                                resp=''

                            label_save='Nz_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(label,thetmin,thetmax,Nbins[nnn],resampling,resp,jk_r)
                            label_save1='BNz_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(label_bias,thetmin,thetmax,Nbins[nnn],resampling,resp,jk_r)


                            #compute statistics

                            statistics=plot(resampling,reference_bins_interval,N,Nz[method],label_save,jk_r,False,False,False,only_diagonal,verbose)

                            BBstatistics,BGstatistics=plot(resampling,reference_bins_interval,N,BNz[method],label_save1,jk_r,False, False,False,only_diagonal,verbose)


                        iter+=1
                        update_progress(float(iter)/totiter,timeit.default_timer(),start)





    # OPTIMIZATION OF SCALES
    scale_optimization(best_params,Nbins,interval_width,step_width,only_diagonal,optimization,N,resampling,resampling_pairs,jk_r)

    # REGULARIZATION (GAUSSIAN PROCESS)
    if regularization :
        #IT HAS TO BE UPDATED!
        regularization_routine(best_params,reference_bins_interval,tomo_bins,N,jk_r,only_diagonal,set_negative_to_zero,fit,prior_gaussian_process,resampling,resp)


def load_true_distribution(reference_bins_interval,tomo_bins,photo_z_column):


    hdf = pd.HDFStore('./pairscount/dataset.h5')
    unknown_bins = hdf['unk']['bins']
    unknown_z = hdf['unk'][photo_z_column]

    N=dict()

    if tomo_bins=='ALL':
        list_of_tomo=range(len(unknown_bins_interval['z']))+1
    else:
        list_of_tomo=[]
        for i in range(len(tomo_bins)):
            list_of_tomo.append(int(tomo_bins[i]))
        list_of_tomo=np.array(list_of_tomo)

    for mute,i in enumerate(list_of_tomo):
        Nu=dict()
        i=i-1

        zpt=unknown_z[unknown_bins==i+1]
        zpt=zpt[(zpt>reference_bins_interval[str(i)]['z_edges'][0]) & (zpt<reference_bins_interval[str(i)]['z_edges'][-1])]
        zpt=np.array(zpt)

        NN,ztruth=np.histogram(zpt,bins=reference_bins_interval[str(i)]['z_edges'])
        Nu.update({'zp_t':zpt})
        Nu.update({'N':NN})
        Nu.update({'ztruth':ztruth})
        N.update ({str(i):Nu})
    return N

def number_of_iterations(Nbins,optimization,methods,interval_width,step_width):
    totiter=0
    for nnn in range(len(Nbins)):
        if not optimization:
            # if optimization is set to false , for each angular bins it just runs considering
            # the full angular/physical distance range.
            interval_bins=Nbins[nnn]
            step=1
        else:
            interval_bins=int(max([math.ceil(Nbins[nnn]/2.),interval_width]))
            #interval_bins=int(max([interval_width,interval_width]))
            step=int(math.ceil(Nbins[nnn]/step_width))
        for thetmax in range(interval_bins,Nbins[nnn]+1,step):
            for thetmin in range(0,thetmax-interval_bins+1,step):
                totiter+=1

    return totiter

def cut_redshift_range(z_min,z_max,tomo_bins):

    unknown_bins_interval_dict=load_obj('./pairscount/unknown_bins_interval')
    reference_bins_interval_dict=load_obj('./pairscount/reference_bins_interval')

    reference_bins_interval=dict()

    if tomo_bins=='ALL':
        list_of_tomo=range(len(unknown_bins_interval['z']))+1
    else:
        list_of_tomo=[]
        for i in range(len(tomo_bins)):
            list_of_tomo.append(int(tomo_bins[i]))
        list_of_tomo=np.array(list_of_tomo)
    for mute,i in enumerate(list_of_tomo):
        redshift_slice=dict()
        z_unk_value=unknown_bins_interval_dict['z'][mute]
        i=i-1

        z=reference_bins_interval_dict['z']
        z_point=reference_bins_interval_dict['z']
        z_bin=reference_bins_interval_dict['z_edges']

        label_up=-1
        label_dwn=0
        if z_min[i] != 'None':
            z_bin=z_bin[(z_bin>=z_min[i])]
            z=0.5*(z_bin[:-1]+z_bin[1:])
            label_dwn=np.where(z_point==z[0])[0][0]
        if z_max[i] != 'None':
            z_bin=z_bin[(z_bin<=z_max[i])]
            z=0.5*(z_bin[:-1]+z_bin[1:])
            label_up=np.where(z_point==z[-1])[0][0]

        redshift_slice.update({'z':z})
        redshift_slice.update({'z_edges':z_bin})
        redshift_slice.update({'label_dwn':label_dwn})

        reference_bins_interval.update({str(i):redshift_slice})
    return reference_bins_interval

def create_dictionary_bestscales(methods,weight_variance,pairs_weighting,bias_correction_reference_Menard,bias_correction_Newman,
                    bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,fit_free,sum_schmidt):
        best_params=dict()

        for method in np.unique(methods):
            label,label_bias=define_label(method,fit_free,sum_schmidt,weight_variance,pairs_weighting,bias_correction_reference_Menard,bias_correction_Newman,
                                bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt)

            best_params_method=dict()
            best_params.update({'Nz_'+label:best_params_method})
            if label_bias!=label:
                best_params.update({'BNz_'+label_bias:best_params_method})
        return best_params

def define_label(method,fit_free,sum_schmidt,weight_variance,pairs_weighting,bias_correction_reference_Menard,bias_correction_Newman,
                    bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt):

    label_bias=''
    label_weight=''
    labeladd=''
    if method == 'Newman' and fit_free:
        labeladd='_exp'
    if method =='Schmidt' and sum_schmidt:
        labeladd='_sum'

    if method != 'Newman' and method!='Schmidt':
        if weight_variance: label_weight+='_wvar_'
        if  pairs_weighting: label_weight+='_wpairs_'
        if bias_correction_reference_Menard!='None':
            label_bias+='_biasr_'+bias_correction_reference_Menard+'_'
        if bias_correction_unknown_Menard!='None':
            label_bias+='_biasu_'+bias_correction_unknown_Menard+'_'
    elif method == 'Newman':
        if bias_correction_Newman==1:
            label_bias+='_biasiter_'
    elif method == 'Schmidt':
        if bias_correction_reference_Schmidt!='None':
            label_bias+='_biasr_'+bias_correction_reference_Schmidt+'_'
        if bias_correction_unknown_Schmidt!='None':
            label_bias+='_biasu_'+bias_correction_unknown_Schmidt+'_'

    return '{0}_{1}_{2}'.format(method,label_weight,labeladd),'{0}_{1}_{2}{3}'.format(method,label_weight,labeladd,label_bias)
