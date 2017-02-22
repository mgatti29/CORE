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
        bias_correction_Menard=0,bias_correction_Newman=0,bias_correction_Schmidt=0,
        use_physical_scale_Newman=False,
        weight_variance=False,pairs_weighting=False,
        fit_free=True,
        optimization=False,
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
        time0=0.):



    if time0>0:
        verbose=True
    else:
        verbose=False


    # load redshift arrays  *******************************
    unknown_bins_interval=load_obj('./pairscount/unknown_bins_interval')
    reference_bins_interval=load_obj('./pairscount/reference_bins_interval')

    z=reference_bins_interval['z']

    # impose external redshift cuts.
    label_dwn=0

    if z_min != 'None' or z_max != 'None':
        label_dwn,reference_bins_interval=cut_redshift_range(reference_bins_interval,z_min,z_max)




    #loading the true distribution *************************
    N,ztruth,zp_t,zp_t_TOT,jk_r=load_true_distribution(unknown_bins_interval,reference_bins_interval)


    #  estimate number of iterations for each method: ***************
    totiter=number_of_iterations(Nbins,optimization,methods,interval_width,step_width)

    iter=0



    #*******************************************************************
    # create a dictionary to keep the relevant parameters of each method.
    best_params,times,weight_variance_tot,pairs_weighting_tot,bias_Menard_tot,bias_Schmidt_tot,bias_Newman_tot,methods_tot,key,fit_free_tot,sum_schmidt_tot=create_dictionary_bestscales(methods,weight_variance,pairs_weighting,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt,fit_free,sum_schmidt)



    ###################################################################################
    #                                                                                 #
    #                                 MAIN CYCLE                                      #
    #                                                                                 #
    ###################################################################################


    if just_pickup_best_scales:
        pass
    else:

        for iii in range(times):
            if (key=='ALL') or (key=='RELIABLE') or (key=='REL_short') or (key=='test'):

                methods=methods_tot[iii]
                weight_variance=weight_variance_tot[iii]
                pairs_weighting=pairs_weighting_tot[iii]
                bias_correction_Menard=bias_Menard_tot[iii]
                bias_correction_Newman=bias_Newman_tot[iii]
                bias_correction_Schmidt=bias_Schmidt_tot[iii]
                fit_free=fit_free_tot[iii]
                sum_schmidt=sum_schmidt_tot[iii]

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



                for thetmax in range(interval_bins,Nbins[nnn]+1,step):
                    for thetmin in range(0,thetmax-interval_bins+1,step):
                        iter+=1
                        if verbose:

                            print '{0},{1},{2},{3}'.format(method,thetmin,thetmax,Nbins[nnn])
                        else:
                            if iter==1:
                                start=time.time()
                                update_progress(0.)




                        # READING FILES AND Nz COMPUTATION ****************************

                        # Reading files ***********************************************
                        correlation,methods=load_w_treecorr(methods,unknown_bins_interval['z'].shape[0],
                                        reference_bins_interval['z'].shape[0],
                                        Nbins[nnn],thetmin,thetmax,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt,sum_schmidt,
                                        use_physical_scale_Newman,w_estimator,verbose,label_dwn)



                        # Optimization of scales. **************************************
                        # For each method, for each redshift bin, it integrates over the
                        # correlation signal. If required, it also perform the fits
                        correlation_optimized=optimize(correlation,methods,z,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt,weight_variance,
                                            pairs_weighting,fit_free,use_physical_scale_Newman,bounds_CC_fit,initial_guess_CC_fit,bounds_AC_U_fit,
                                            initial_guess_AC_U_fit,bounds_AC_R_fit,initial_guess_AC_R_fit,verbose)




                        # Save fit  ****************************************************
                        # It saves the results of the fitting (Newman and Schmidt methods)
                        if show_fit and ('Newman' in methods):
                            label_save='{0}_{1}_{2}'.format(thetmin,thetmax,Nbins[nnn])
                            save_fit(correlation_optimized,correlation,'./output_dndz/fit/',label_save,verbose,label_dwn)



                        # Compute N(z) *************************************************
                        Nz_tomo,BNz_tomo=compute_Nz(correlation_optimized,methods,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt
                                                ,jk_r,reference_bins_interval['z'],use_physical_scale_Newman,verbose,label_dwn)


                        #TODO: iterative bias correction (only for Newman & Schmidt):
                        #while:
                        #  correlation_optimized,Nz=iterative(correlation_optimized,Nz)



                        # Stacking *****************************************************
                        # It stacks different tomo bins for each method,after having normalized.
                        Nz,BNz=stacking(Nz_tomo,BNz_tomo,reference_bins_interval['z'],jk_r,zp_t)



                        #Plotting and compute statistics.  ***********************************************
                        for method in Nz.keys():
                            label_weight=''
                            label_bias=''
                            labeladd=''
                            if method == 'Newman' and fit_free:
                                labeladd='_exp'
                            if method =='Schmidt' and sum_schmidt:
                                labeladd='_sum'
                            if method != 'Newman' and method!='Schmidt':
                                if weight_variance: label_weight+='_wvar_'
                                if  pairs_weighting: label_weight+='_wpairs_'
                                if bias_correction_Menard==2 :
                                    label_bias+='_newbias_'
                            elif method == 'Newman':
                                if bias_correction_Newman==1:
                                    label_bias+='_biasiter_'
                                elif bias_correction_Newman==2:
                                    label_bias+='_biasrr_'
                            elif method == 'Schmidt':
                                if bias_correction_Schmidt==1:
                                    label_bias+='_biasiter_'
                                elif bias_correction_Schmidt==2:
                                    label_bias+='_biasrr_'
                                elif bias_correction_Schmidt==3:
                                    label_bias+='_bias1bin_'
                                elif bias_correction_Schmidt==4:
                                    label_bias+='_bias1bin_AC'
                            #Define labels for the output



                            label_save='Nz_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(method,label_weight,label_bias,labeladd,thetmin,thetmax,Nbins[nnn],labeladd)
                            label_save1='BNz_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(method,label_weight,label_bias,labeladd,thetmin,thetmax,Nbins[nnn])


                            #compute statistics
                            statistics=plot(reference_bins_interval['z'],reference_bins_interval['z_edges'],zp_t_TOT,Nz[method],N,label_save,'./output_dndz/Nz/',
                                            jk_r,False,False,False,only_diagonal,verbose)

                            if np.sum(BNz[method])>0.:
                                BBstatistics,BGstatistics=plot(reference_bins_interval['z'],reference_bins_interval['z_edges'],zp_t_TOT,BNz[method],
                                                N,label_save1,'./output_dndz/Nz/',jk_r,False, False,False,only_diagonal,verbose)

                            else:  BBstatistics,BGstatistics=None,None



                            #add statistics to the dictionary (it reads it later)
                            '''
                            statistics={'stats':copy.deepcopy(statistics),
                            'bb_stats':copy.deepcopy(BBstatistics),
                            'bg_stats':copy.deepcopy(BGstatistics),
                            'thetmin':thetmin,
                            'thetmax':thetmax,
                            'Nbins':Nbins[nnn]}

                            best_params['{0}_{1}'.format(method,label_weight)].update({'{0}'.format(counter):statistics})
                            counter+=1
                            '''

                        #TODO: make this update progress thinner
                        update_progress(float(iter)/totiter,timeit.default_timer(),start)





    # OPTIMIZATION OF SCALES
    scale_optimization(best_params,Nbins,interval_width,step_width,only_diagonal,optimization)

    # REGULARIZATION (GAUSSIAN PROCESS)



    if regularization :
        regularization_routine(reference_bins_interval['z'],reference_bins_interval['z_edges'],
                                zp_t_TOT,N,jk_r,only_diagonal,set_negative_to_zero,fit,prior_gaussian_process)

    if plot_compare_all: make_plot_compare_all(only_diagonal)

def load_true_distribution(unknown_bins_interval,reference_bins_interval):
    hdf = pd.HDFStore('./pairscount/dataset.h5')
    unknown_bins = hdf['unk']['bins']
    unknown_z = hdf['unk']['Z_T']

    jk_r=len(np.unique(hdf['unk']['HPIX']))

    zp_t=dict()
    zp_t_TOT=[]
    for i in range(unknown_bins_interval['z'].shape[0]):
        zpt=unknown_z[unknown_bins==i+1]
        zpt=zpt[(zpt>reference_bins_interval['z_edges'][0]) & (zpt<reference_bins_interval['z_edges'][-1])]
        zp_t.update({'{0}'.format(i+1):zpt}) #point estimate
        zp_t_TOT.append(zpt)

    zp_t_TOT=np.array(zp_t_TOT)
    zp_t_TOT=zp_t_TOT[0,:]
    N,ztruth=np.histogram(zp_t_TOT,bins=reference_bins_interval['z_edges'])
    #stacked, true distribution
    N=np.array(N)

    return N,ztruth,zp_t,zp_t_TOT,jk_r

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

    if 'ALL'in methods:
        totiter=totiter*17

    elif 'RELIABLE'in methods:
        totiter=totiter*6

    elif 'REL_short' in methods:
        totiter=totiter*3.
    elif 'testu' in methods:
        totiter=totiter*5.
    #else:
    #    totiter=len(np.unique(methods))*14
    return totiter

def cut_redshift_range(reference_bins_interval,z_min,z_max):
    z_point=reference_bins_interval['z']
    z_bin=reference_bins_interval['z_edges']

    label_up=-1
    label_dwn=0
    if z_min != 'None':
        z_bin=z_bin[(z_bin>=z_min)]
        z=0.5*(z_bin[:-1]+z_bin[1:])
        label_dwn=np.where(z_point==z[0])[0][0]
    if z_max != 'None':
        z_bin=z_bin[(z_bin<=z_max)]
        z=0.5*(z_bin[:-1]+z_bin[1:])
        label_up=np.where(z_point==z[-1])[0][0]

    reference_bins_interval['z']=z
    reference_bins_interval['z_edges']=z_bin
    return label_dwn,reference_bins_interval

def create_dictionary_bestscales(methods,weight_variance,pairs_weighting,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt,fit_free,sum_schmidt):
    best_params=dict()
    if 'ALL'in methods:
        times=17
        methods_tot=['Menard','Menard_physical_scales','Menard_physical_weighting','Menard','Menard_physical_scales',
                'Menard_physical_weighting','Menard','Menard_physical_scales','Menard_physical_weighting','Menard',
                'Menard_physical_scales','Menard_physical_weighting','Schmidt','Schmidt','Schmidt','Newman','Newman']
        weight_variance_tot=[True,True,True,False,False,False,True,True,True,False,False,False,False,False,False,False,False]
        pairs_weighting_tot=[True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False]
        fit_free_tot=[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
        sum_schmidt_tot=[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
        bias_Menard_tot=[1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,1,2]
        bias_Newman_tot=[1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,1,2]
        bias_Schmidt_tot=[1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,1,2]
        key='ALL'

    #create dictionary
        for iii in range(times):
            method=methods_tot[iii]
            weight_variance=weight_variance_tot[iii]
            pairs_weighting=pairs_weighting_tot[iii]
            bias_correction_Menard=bias_Menard_tot[iii]
            bias_correction_Newman=bias_Newman_tot[iii]
            bias_correction_Schmidt=bias_Schmidt_tot[iii]
            fitfree=fit_free_tot[iii]
            sum_schmidt=sum_schmidt_tot[iii]

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
                if bias_correction_Menard==2 :
                    label_bias+='_newbias_'
            elif method == 'Newman':
                if bias_correction_Newman==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Newman==2:
                    label_bias+='_biasrr_'
            elif method == 'Schmidt':
                if bias_correction_Schmidt==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Schmidt==2:
                    label_bias+='_biasrr_'
                elif bias_correction_Schmidt==3:
                    label_bias+='_bias1bin_'
                elif bias_correction_Schmidt==4:
                    label_bias+='_bias1bin_AC'
            best_params_method=dict()
            best_params.update({'{0}_{1}_{2}_{3}'.format(method,label_weight,label_bias,labeladd):best_params_method})

    elif 'RELIABLE' in methods:
        times=6
        methods_tot=['Menard_physical_scales','Schmidt','Schmidt','Schmidt','Newman','Newman']
        weight_variance_tot=[False,False,False,False,False,False]
        pairs_weighting_tot=[False,False,False,False,False,False]
        fit_free_tot=[False,False,False,False,False,False]
        sum_schmidt_tot=[False,False,False,False,False,False]
        bias_Menard_tot=[1,1,2,3,1,2]
        bias_Newman_tot=[1,1,2,3,1,2]
        bias_Schmidt_tot=[1,1,2,3,1,2]

        key='RELIABLE'
        for iii in range(times):
            method=methods_tot[iii]
            weight_variance=weight_variance_tot[iii]
            pairs_weighting=pairs_weighting_tot[iii]
            bias_correction_Menard=bias_Menard_tot[iii]
            bias_correction_Newman=bias_Newman_tot[iii]
            bias_correction_Schmidt=bias_Schmidt_tot[iii]
            fitfree=fit_free_tot[iii]
            sum_schmidt=sum_schmidt_tot[iii]
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
                if bias_correction_Menard==2 :
                    label_bias+='_newbias_'
            elif method == 'Newman':
                if bias_correction_Newman==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Newman==2:
                    label_bias+='_biasrr_'
            elif method == 'Schmidt':
                if bias_correction_Schmidt==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Schmidt==2:
                    label_bias+='_biasrr_'
                elif bias_correction_Schmidt==3:
                    label_bias+='_bias1bin_'
                elif bias_correction_Schmidt==4:
                    label_bias+='_bias1bin_AC'

            best_params_method=dict()
            best_params.update({'{0}_{1}_{2}{3}'.format(method,label_weight,label_bias,labeladd):best_params_method})

    elif 'REL_short' in methods:
        times=3
        methods_tot=['Menard_physical_scales','Schmidt','Newman']
        weight_variance_tot=[False,False,False]
        pairs_weighting_tot=[False,False,False]
        fit_free_tot=[False,False,False]
        sum_schmidt_tot=[False,False,False]
        bias_Menard_tot=[1,3,1]
        bias_Newman_tot=[1,3,1]
        bias_Schmidt_tot=[1,3,1]
        key='REL_short'


    #create dictionary
        for iii in range(times):
            method=methods_tot[iii]
            weight_variance=weight_variance_tot[iii]
            pairs_weighting=pairs_weighting_tot[iii]
            bias_correction_Menard=bias_Menard_tot[iii]
            bias_correction_Newman=bias_Newman_tot[iii]
            bias_correction_Schmidt=bias_Schmidt_tot[iii]
            fitfree=fit_free_tot[iii]
            sum_schmidt=sum_schmidt_tot[iii]

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
                if bias_correction_Menard==2 :
                    label_bias+='_newbias_'
            elif method == 'Newman':
                if bias_correction_Newman==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Newman==2:
                    label_bias+='_biasrr_'
            elif method == 'Schmidt':
                if bias_correction_Schmidt==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Schmidt==2:
                    label_bias+='_biasrr_'
                elif bias_correction_Schmidt==3:
                    label_bias+='_bias1bin_'
                elif bias_correction_Schmidt==4:
                    label_bias+='_bias1bin_AC'
            best_params_method=dict()
            best_params.update({'{0}_{1}_{2}_{3}'.format(method,label_weight,label_bias,labeladd):best_params_method})

    elif 'testu' in methods:
        times=5
        methods_tot=['Menard_physical_scales','Schmidt','Newman','Schmidt','Newman']
        weight_variance_tot=[False,False,False,False,False]
        pairs_weighting_tot=[False,False,False,False,False]
        fit_free_tot=[False,False,False,False,True]
        sum_schmidt_tot=[False,False,False,True,False]

        bias_Menard_tot=[1,3,1,3,1]
        bias_Newman_tot=[1,3,1,3,1]
        bias_Schmidt_tot=[1,3,1,3,1]
        key='test'


    #create dictionary
        for iii in range(times):
            method=methods_tot[iii]
            weight_variance=weight_variance_tot[iii]
            pairs_weighting=pairs_weighting_tot[iii]
            bias_correction_Menard=bias_Menard_tot[iii]
            bias_correction_Newman=bias_Newman_tot[iii]
            bias_correction_Schmidt=bias_Schmidt_tot[iii]
            fitfree=fit_free_tot[iii]
            sum_schmidt=sum_schmidt_tot[iii]

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
                if bias_correction_Menard==2 :
                    label_bias+='_newbias_'
            elif method == 'Newman':
                if bias_correction_Newman==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Newman==2:
                    label_bias+='_biasrr_'
            elif method == 'Schmidt':
                if bias_correction_Schmidt==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Schmidt==2:
                    label_bias+='_biasrr_'
                elif bias_correction_Schmidt==3:
                    label_bias+='_bias1bin_'
                elif bias_correction_Schmidt==4:
                    label_bias+='_bias1bin_AC'
            best_params_method=dict()
            best_params.update({'{0}_{1}_{2}_{3}'.format(method,label_weight,label_bias,labeladd):best_params_method})




    else:
        key=False
        times=1
        methods_tot=None
        weight_variance_tot=None
        pairs_weighting_tot=None
        bias_Menard_tot=None
        bias_Newman_tot=None
        bias_Schmidt_tot=None
        fit_free_tot=None
        sum_schmidt_tot=None
        for method in np.unique(methods):
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
                if bias_correction_Menard==2 :
                    label_bias+='_newbias_'
            elif method == 'Newman':
                if bias_correction_Newman==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Newman==2:
                    label_bias+='_biasrr_'
            elif method == 'Schmidt':
                if bias_correction_Schmidt==1:
                    label_bias+='_biasiter_'
                elif bias_correction_Schmidt==2:
                    label_bias+='_biasrr_'
                elif bias_correction_Schmidt==3:
                    label_bias+='_bias1bin_'
                elif bias_correction_Schmidt==4:
                    label_bias+='_bias1bin_AC'
            best_params_method=dict()
            best_params.update({'{0}_{1}_{2}_{3}'.format(method,label_weight,label_bias,labeladd):best_params_method})

    return best_params,times,weight_variance_tot,pairs_weighting_tot,bias_Menard_tot,bias_Schmidt_tot,bias_Newman_tot,methods_tot,key,fit_free_tot,sum_schmidt_tot
