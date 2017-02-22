import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import beta
from scipy import special
import scipy.integrate as pyint
import matplotlib.transforms as mtransforms
import matplotlib as mpl
import os
import scipy.optimize as optimization
from scipy.optimize import fmin as simplex
import time
from numpy import linalg
import matplotlib.mlab as mlab
import math
import sys
import astropy
from astropy.cosmology import Planck15 as Planck15
from scipy.optimize import minimize as mnn
from scipy.integrate import trapz
import copy
import timeit
import pickle
import emcee
from gapp import dgp
from .dataset import save_obj, load_obj, update_progress
import pandas as pd
from scipy.optimize import curve_fit
cosmol=Planck15


#*************************************************************************
#                 reading  trecorr files
#*************************************************************************

def load_w_treecorr(methods,narrow_bin_num,numspecbins,Nbins,thetmin,thetmax,bias_correction_Menard,bias_correction_Newman,
                    bias_correction_Schmidt,sum_schmidt,use_physical_scale_Newman,w_estimator,verbose,label_dwn):
    '''
    #It loads the correlations function computed from treecorr.

    INPUT:
    methods: list of methods you want to use to compute the N(z).
    narrow_bin_num: number of tomographic bins of the unknown sample
    numspecbins: number of bins of the reference sample

    Nbins: number of angular/physical bins your correlation functions are divided into
    thetmin,thetmax: indexes of the min/max angular/physical bin you are considering.
    bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt: bias correction options.
    label_dwn: index of the 1st and last redshift bins you want to consider out of the full interval

    OUTPUT:
    correlation: dictionary containing all the correlations divided per method and redshift.

                 structure of the dictionary:

                 -tomo_unknown (tomographic bin of the unknown)
                    -redshidt_ref (redshift bin of the reference)
                        -correlation_type (type of correlation)
                            -label (type of correlation)
                            -basis (theta or physical distance array of the correlation)
                            -w (correlation function)
                            -err (error on the correlation function)
                            -jk (correlation function for each jackknife region)
                            -cov (covariance between different angular/physical distance bins)

    new_methods: unique list of methods

    '''

    # Here is the description of which type of correlation is needed by each method.
    # the code will try to load the ones needed depending on the methods used.

    if bias_correction_Menard == 1 or bias_correction_Menard == 2:
        Menard={'methods':['CC_A_','AC_R_A_']}
        Menard_physical_scales={'methods':['CC_P_','AC_R_P_']}
        Menard_physical_weighting={'methods':['CC_A_','AC_R_A_']}
    else:
        Menard={'methods':['CC_A_']}
        Menard_physical_scales={'methods':['CC_P_']}
        Menard_physical_weighting={'methods':['CC_A_']}


    if use_physical_scale_Newman:
        Newman={'methods':['CC_P_','AC_U_','AC_R_R_']}
    else:
        Newman={'methods':['CC_A_','AC_U_','AC_R_R_']}


    if bias_correction_Schmidt==3 :
        Schmidt={'methods':['CC_D_','AC_R_P_']}
    elif bias_correction_Schmidt==0 :
        Schmidt={'methods':['CC_D_']}
    elif bias_correction_Schmidt==4:
        Schmidt={'methods':['CC_D_','AC_R_D_']}
    else :
        Schmidt={'methods':['CC_D_','AC_U_','AC_R_R_']}


    list_of_methods={'Menard':Menard,
                'Menard_physical_scales':Menard_physical_scales,
                'Menard_physical_weighting':Menard_physical_weighting,
                'Newman':Newman,
                'Schmidt':Schmidt}


    if verbose:
        print('\n**** LOADING FILE MODULE ****')
        print('Checking the methods')
    toberead=[]
    new_methods=[]
    for samp in np.unique(methods):
        try:
            if samp in list_of_methods.keys():
                new_methods.append(samp)
            for labels in list_of_methods[samp].keys():
                for label_method in np.unique(list_of_methods[samp][labels]):
                    toberead.append(label_method)
        except:
            #exception: method not in list of methods.
            print('-> Exception: No method {0} implemented'.format(samp))
    new_methods=np.unique(new_methods)


    #load max_rpar
    max_rpar=load_obj('./pairscount/max_rpar')
    if verbose:
        print('Loading files')
    correlation=dict()

    for i in range(narrow_bin_num):
        redshift_dict=dict()
        for j in range(numspecbins):
            slice_method_dict=dict()
            for correlation_type in toberead:

                #try:

                    if correlation_type=='AC_U_':
                        pairs=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format(correlation_type,Nbins,i+1,1))

                    else:
                        pairs=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format(correlation_type,Nbins,i+1,j+1+label_dwn))


                    if correlation_type=='CC_D_' or correlation_type=='AC_R_D_':
                        #In case of the density method for Schmidt, we have to sum up bins  to produce one bin estimate
                        w_summed=np.zeros(pairs['w'].shape[1])
                        DD_summed=np.zeros(pairs['DD'].shape[1])
                        DR_summed=np.zeros(pairs['DR'].shape[1])
                        RD_summed=np.zeros(pairs['RD'].shape[1])
                        RR_summed=np.zeros(pairs['RR'].shape[1])
                        for jck in range(w_summed.shape[0]):

                            DD_summed[jck]=np.sum(pairs['DD'][thetmin:thetmax,jck])
                            DR_summed[jck]=np.sum(pairs['DR'][thetmin:thetmax,jck])
                            RD_summed[jck]=np.sum(pairs['RD'][thetmin:thetmax,jck])
                            RR_summed[jck]=np.sum(pairs['RR'][thetmin:thetmax,jck])

                            if sum_schmidt:
                                w_summed[jck]=np.sum(pairs['w'][thetmin:thetmax,jck])
                            else:
                                w_summed[jck]=estimator(w_estimator,DD_summed[jck],DR_summed[jck],RD_summed[jck],RR_summed[jck])

                        w={'label':correlation_type,
                            'basis':None,
                            'w':w_summed,
                            'err':None,
                            'DD':w_summed,
                            'DR':w_summed,
                            'RD':w_summed,
                            'RR':w_summed,
                            'estimator':w_estimator}


                    else:
                        # compute covariance and errors.
                        dict_cov=covariance_jck(pairs['w'][thetmin:thetmax,1:],pairs['w'].shape[1]-1)
                        DD_dict_cov=covariance_jck(pairs['DD'][thetmin:thetmax,1:],pairs['DD'].shape[1]-1)
                        DR_dict_cov=covariance_jck(pairs['DR'][thetmin:thetmax,1:],pairs['DR'].shape[1]-1)
                        RD_dict_cov=covariance_jck(pairs['RD'][thetmin:thetmax,1:],pairs['RD'].shape[1]-1)
                        RR_dict_cov=covariance_jck(pairs['RR'][thetmin:thetmax,1:],pairs['RR'].shape[1]-1)



                        #'w':copy.deepcopy(pairs['w'][thetmin:thetmax,:]),

                        ww=np.zeros((pairs['DD'][thetmin:thetmax,:].shape[0],pairs['DD'][thetmin:thetmax,:].shape[1]))
                        for hh in range(pairs['DD'][thetmin:thetmax,:].shape[0]):
                         for kk in range(pairs['DD'][thetmin:thetmax,:].shape[1]):
                            ddw=copy.deepcopy(pairs['DD'][thetmin+hh,kk])
                            drw=copy.deepcopy(pairs['DR'][thetmin+hh,kk])
                            rdw=copy.deepcopy(pairs['RD'][thetmin+hh,kk])
                            rrw=copy.deepcopy(pairs['RR'][thetmin+hh,kk])
                            ww[hh,kk]=estimator(w_estimator,ddw,drw,rdw,rrw)
                            if correlation_type=='AC_R_R_':
                                ww[hh,kk]=ww[hh,kk]*max_rpar*2.

                        w={'label':correlation_type,
                            'basis':copy.deepcopy(pairs['theta'][thetmin:thetmax]),
                            #'w':copy.deepcopy(pairs['w'][thetmin:thetmax]),
                            'w':ww,
                            'err':copy.deepcopy(dict_cov['err']),
                            'cov':copy.deepcopy(dict_cov['cov']),
                            'DD':copy.deepcopy(pairs['DD'][thetmin:thetmax,:]),
                            'DD_err':copy.deepcopy(DD_dict_cov['err']),
                            'DR':copy.deepcopy(pairs['DR'][thetmin:thetmax,:]),
                            'DR_err':copy.deepcopy(DR_dict_cov['err']),
                            'RD':copy.deepcopy(pairs['RD'][thetmin:thetmax,:]),
                            'RD_err':copy.deepcopy(RD_dict_cov['err']),
                            'RR':copy.deepcopy(pairs['RR'][thetmin:thetmax,:]),
                            'RR_err':copy.deepcopy(RR_dict_cov['err']),
                            'estimator':w_estimator}

                    #Add the dictionary to the collection of dictionaries
                    slice_method_dict.update({'{0}'.format(correlation_type):w})

                #except:
                #    print('->Exception: files missing for /SLICE_'+str(i+1)+'/optimize_new/{2}_{3}_{4}_{5}_{6}'.format(names_final,i+1,correlation_type,Nbins,thetmin,thetmax,label_dwn+j+1))

            redshift_dict.update({'{0}'.format(label_dwn+j+1):slice_method_dict})
    correlation.update({'{0}'.format(i+1):redshift_dict})


    return correlation,new_methods


#*************************************************************************
#                optimization
#*************************************************************************

def optimize(correlation,methods,redshift,bias_correction_Menard,bias_correction_Newman,
                    bias_correction_Schmidt,weight_variance,pairs_weighting,fit_free,use_physical_scale_Newman,
                    bounds_CC_fit,initial_guess_CC_fit,bounds_AC_U_fit,initial_guess_AC_U_fit,bounds_AC_R_fit,
                    initial_guess_AC_R_fit,verbose,gamma=1,label_dwn=0):
    '''
    From a dictionary containing correlation functions, it integrates the correlation functions over the angular interval.
    In case, it applies a minimum variance weighting or fit the autocorrelation function (see below).

    INPUT:
    correlation: dictionary containing all the correlations divided per method and redshift.

                 structure of the dictionary:

                 -tomo_unknown (tomographic bin of the unknown)
                    -redshidt_ref (redshift bin of the reference)
                        -correlation_types (type of correlation)
                            -label (type of correlation)
                            -basis (theta or physical distance array of the correlation)
                            -w (correlation function)
                            -err (error on the correlation function)
                            -jk (correlation function for each jackknife region)
                            -cov (covariance between different angular/physical distance bins)

    redshift: redshift array (needed for physical weighting method)
    methods: list of methods that will be used to compute the N(z)
    bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt: bias correction options.
    weight_variance: keyword. If set, it applies the minimum variance weighting scheme.
    pairs_weighting: keyword. If set, it weights the pairs (DD,RD,DR,RR) first and then combine them into w.
    gamma: exponent of theta^(-gamma) in the weighting function.

    OUTPUT:
    correlation_optimized: dictionary containing all the correlations divided per method and redshift.

                 structure of the dictionary:

                 -tomo_unknown (tomographic bin of the unknown)
                    -redshidt_ref (redshift bin of the reference)
                        -correlation_type (type of correlation)
                            if the method does not involve fitting:
                            -<w> (correlation function integrated over angular range. from the second element corresponds to the jackknife valaue)

                             if the  method involves fitting:
                            -<params> (params of the fitting. from the second element corresponds to the jackknife valaue)

    n.b.: the fitting and the optimization procedure are done taking into account only the diagonal of the covariance. This can be modified
    '''
    if verbose:
        print('\n**** OPTIMIZE MODULE ****')

    if use_physical_scale_Newman:
        cc_newman='CC_P_'
    else:
        cc_newman='CC_A_'
    correlation_optimized=dict()

    for i,tomo_bin in enumerate(correlation.keys()):
        redshift_dict=dict()
        for j,reference_bin in enumerate(correlation[tomo_bin].keys()):

            slice_method_dict=dict()
            types=correlation[tomo_bin][reference_bin].keys()

           # put CC_P_ at the end of the list (we  need it for Newman method)
           # put AC_P at the beginning of the list ( we need it for bias_correction_4)
            mute=types[len(types)-1]
            mute1=types[0]
            for ii,mode in enumerate(types):
                if mode==cc_newman:
                    types[ii]=mute
                    types[len(types)-1]=cc_newman
                if mode=='AC_R_P_':
                    types[ii]='AC_R_P_'
                    types[ii]=mute1

            if verbose:
                update_progress((np.float(j)+1)/ len(correlation[tomo_bin].keys()))

            for correlation_type in types:
                # angular or physical distance optimization:


                if (correlation_type=='CC_P_') and ('Menard_physical_scales' in methods ) and ( not pairs_weighting) and bias_correction_Menard==2:
                    w_weighted_bias4=[]
                    w_weighted_bias44=[]
                    x=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['basis'])
                    w=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'])
                    err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['err'])
                    w_AC=copy.deepcopy(correlation[tomo_bin][reference_bin]['AC_R_P_']['w'])
                    for ik in range(w.shape[1]):
                        mute_w_ratio=np.zeros(len(w[:,ik]))
                        mask=(np.isfinite(w_AC[:,ik]))
                        for ih in range(len(w[:,ik])):
                            mute_w_ratio[ih]=  w[ih,ik]/w_AC[ih,ik]
                        #print (mask,mute_w_ratio,w_AC[:,ik],w[:,ik])
                        if (not np.isfinite(weight_w(mute_w_ratio[mask],x[mask],err[mask],False,0)['integr'])):
                            w_weighted_bias4.append(0.)
                        else:
                            w_weighted_bias4.append(weight_w(mute_w_ratio[mask],x[mask],err[mask],False,0)['integr'])
                    w_weighted_bias44={'<w>':w_weighted_bias4}
                    slice_method_dict.update({'{0}bias4'.format(correlation_type):w_weighted_bias44})

                if correlation_type =='CC_A_' or correlation_type =='AC_R_A_' or correlation_type=='CC_P_' or correlation_type=='AC_R_P_':

                    w_weighted=[]
                    w_weighted_phys=[]
                    x=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['basis'])
                    w=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'])
                    err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['err'])

                    w_estimator=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['estimator'])

                    if pairs_weighting:
                        DD=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['DD'])
                        DR=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['DR'])
                        RD=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['RD'])
                        RR=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['RR'])
                        DD_err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['DD_err'])
                        DR_err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['DR_err'])
                        RD_err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['RD_err'])
                        RR_err=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['RR_err'])

                        for ik in range(w.shape[1]):
                            if correlation_type =='CC_A_' or correlation_type =='AC_R_A_' and ('Menard_physical_weighting' in methods):
                                DD_weighted=weight_phys(DD[:,ik],x,DD_err,redshift[int(reference_bin)-1],weight_variance,gamma)['integr']
                                DR_weighted=weight_phys(DR[:,ik],x,DR_err,redshift[int(reference_bin)-1],weight_variance,gamma)['integr']
                                RD_weighted=weight_phys(RD[:,ik],x,RD_err,redshift[int(reference_bin)-1],weight_variance,gamma)['integr']
                                RR_weighted=weight_phys(RR[:,ik],x,RR_err,redshift[int(reference_bin)-1],weight_variance,gamma)['integr']
                            else:
                                DD_weighted=weight_w(DD[:,ik],x,DD_err,weight_variance,gamma)['integr']
                                DR_weighted=weight_w(DR[:,ik],x,DR_err,weight_variance,gamma)['integr']
                                RD_weighted=weight_w(RD[:,ik],x,RD_err,weight_variance,gamma)['integr']
                                RR_weighted=weight_w(RR[:,ik],x,RR_err,weight_variance,gamma)['integr']
                            mute_w=estimator(w_estimator,DD_weighted,DR_weighted,RD_weighted,RR_weighted)
                            #mute_w=(DD_weighted-DR_weighted-RD_weighted+RR_weighted)/RR_weighted
                            w_weighted.append(mute_w)

                    else:
                        for ik in range(w.shape[1]):

                            if correlation_type =='CC_A_' or correlation_type =='AC_R_A_' and ('Menard_physical_weighting' in methods):
                                 w_weighted_phys.append(weight_phys(w[:,ik],x,err,redshift[int(reference_bin)-1],weight_variance,gamma)['integr'])

                            w_weighted.append(weight_w(w[:,ik],x,err,weight_variance,gamma)['integr'])


                    w_weight={'<w>':w_weighted}
                    if correlation_type =='CC_A_' or correlation_type =='AC_R_A_' and ('Menard_physical_weighting' in methods):
                        w_weight_phys={'<w>':w_weighted}
                        slice_method_dict.update({'{0}weighted'.format(correlation_type):w_weight_phys})

                    #UPDATE THE DICTIONARY WITH THE WEIGHTED W(THETA)
                    slice_method_dict.update({'{0}'.format(correlation_type):w_weight})


                if (correlation_type =='AC_U_' or correlation_type =='AC_R_R_') and  (('Newman' in methods) or ('Schmidt' in methods)): #(not fit_free) and

                    paramsjk_0=[]
                    paramsjk_1=[]
                    paramsjk_2=[]


                    x=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['basis'])
                    w=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'])
                    cov=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['cov'])



                    save_txt=open('./output_dndz/fit/fitparams_{0}_{1}_{2}.txt'.format(correlation_type,i,j),'w')
                    for ik in range(w.shape[1]):
                        if correlation_type =='AC_R_R_':
                            fits=fit_rp(w[:,ik],x,cov,bounds_AC_R_fit,initial_guess_AC_R_fit)
                        else:
                            fits=fit(w[:,ik],x,cov,bounds_AC_U_fit,initial_guess_AC_U_fit)

                        paramsjk_0.append(fits['params'][0])
                        paramsjk_1.append(fits['params'][1])
                        paramsjk_2.append(fits['params'][2])
                        save_txt.write('{0} \t {1} \t {2} \n'.format(fits['params'][0],fits['params'][1],fits['params'][2]))
                    save_txt.close()
                    w_params={'params_0':paramsjk_0,
                        'params_1':paramsjk_1,
                        'params_2':paramsjk_2}
                    slice_method_dict.update({'{0}fit'.format(correlation_type):w_params})


                if correlation_type =='CC_D_' or correlation_type =='AC_R_D_':

                    w_weighted=[]
                    for ik in range(correlation[tomo_bin][reference_bin][correlation_type]['w'].shape[0]):
                        w_weighted.append(copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'][ik]))
                    w_params={'<w>':w_weighted}

                    slice_method_dict.update({'{0}'.format(correlation_type):w_params})




                if correlation_type == cc_newman and 'Newman' in methods:


                    paramsjk_0=[]
                    paramsjk_1=[]
                    paramsjk_2=[]

                    # fit CC_P,fit AC_R_P,fit AC_R_R_,fit AC_U
                    x=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['basis'])
                    w=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'])
                    cov=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['cov'])


                    #the fit to AC_R_R_ and AC_U should have already be computed. if it is the case,
                    #use their exponent to fit the cross correlation.

                    save_txt=open('./output_dndz/fit/fitparams_{0}_{1}_{2}.txt'.format(correlation_type,i,j),'w')
                    for ik in range(w.shape[1]):

                        if fit_free:
                            fits=fit(w[:,ik],x,cov,bounds_CC_fit,initial_guess_CC_fit)
                        elif 'AC_U_fit'  in slice_method_dict.keys():
                            exp_a=copy.deepcopy(slice_method_dict['AC_U_fit']['params_1'][ik])
                            exp_b=copy.deepcopy(slice_method_dict['AC_R_R_fit']['params_1'][ik])
                            exp=(exp_a+exp_b)/2.
                            fits=fit_fixed(w[:,ik],x,cov,exp,bounds_CC_fit,initial_guess_CC_fit)
                        else:
                            print ('warning the code could not fix the index')
                            fits=fit(w[:,ik],x,cov,bounds_CC_fit,initial_guess_CC_fit)
                      #  print fits['params'],fits['fitted']
                        paramsjk_0.append(fits['params'][0])
                        paramsjk_1.append(fits['params'][1])
                        paramsjk_2.append(fits['params'][2])

                    # save the params of the fit.

                        save_txt.write('{0} \t {1} \t {2} \n'.format(fits['params'][0],fits['params'][1],fits['params'][2]))
                    save_txt.close()
                    w_params={'params_0':paramsjk_0,
                        'params_1':paramsjk_1,
                        'params_2':paramsjk_2}
                    #print paramsjk_0,paramsjk_1,paramsjk_2,'\n\n\n'
                    slice_method_dict.update({'{0}fit'.format(correlation_type):w_params})

            redshift_dict.update({'{0}'.format(reference_bin):slice_method_dict})
    correlation_optimized.update({'{0}'.format(tomo_bin):redshift_dict})

    return correlation_optimized

def weight_w(uw,ux,uerr,weight_variance,gamma=1):
    '''
    Integrates w(theta) using different weighting functions

    INPUT:
    uw : corralation function
    ux : theta array
    weight variance: keyword. if set, the weight function is of the form 1/(cov_ii*theta^gamma). Otherwise is just 1/(theta^gamma)
    gamma: exponent for the wighting function.

    OUTPUT:
    dictionary containing the integrated w and the normalization.
    '''


    normd=np.zeros(ux.shape[0])
    for nn in range(ux.shape[0]):
        if weight_variance:
            normd[nn]=1./((ux[nn]**gamma)*uerr[nn]*uerr[nn])
            uw[nn]/=((ux[nn]**gamma)*uerr[nn]*uerr[nn])
        else:
            normd[nn]=1./((ux[nn])**gamma)
            uw[nn]/=((ux[nn])**gamma)

    normdd=trapz(normd,ux)
    wfun=trapz(uw,ux)/normdd


    return {'integr':wfun,
            'err':normdd}

def weight_phys(uw,ux,uerr,z,weight_variance,gamma=1):
    '''
    Integrates w(theta) using different weighting functions

    INPUT:
    uw : corralation function
    ux : theta array
    weight variance: keyword. if set, the weight function is of the form 1/((angular_dist(z)**2)*cov_ii*theta^gamma). Otherwise is just 1/((angular_dist(z)**2)*theta^gamma)
    gamma: exponent for the wighting function.

    OUTPUT:
    dictionary containing the integrated w and the normalization.
    '''



    normd=np.zeros(ux.shape[0])
    for nn in range(ux.shape[0]):

        dist_2=((1+z)*(1+z)*cosmol.angular_diameter_distance(z).value*ux[nn]*(2.*math.pi)/360.)**2.
        if weight_variance:
            normd[nn]=dist_2/((ux[nn]**gamma)*uerr[nn]*uerr[nn])
            uw[nn]/=dist_2*((ux[nn]**gamma)*uerr[nn]*uerr[nn])
        else:
            normd[nn]=dist_2/((ux[nn])**gamma)
            uw[nn]/=dist_2*((ux[nn])**gamma)

    normdd=trapz(normd,ux)
    wfun=trapz(uw,ux)/normdd

    return {'integr':wfun,
            'err':normdd}

def estimator(w_estimator,DD,DR,RD,RR):

    if w_estimator == 'LS':
        #print (w_estimator)
        results = (DD-DR-RD+RR)/(RR)
    elif w_estimator == 'Natural':
        results = (DD - RR) / (RR)
    elif w_estimator == 'Hamilton':
        results = (DD * RR - RD * DR) / (RD * DR)
    elif w_estimator == 'Natural_noRu':
        results = (DD - DR) / DR
    elif w_estimator == 'Natural_noRr':
        results = (DD - RD) / RD
    return results

#*************************************************************************
#                fitting modules
#*************************************************************************

def fit(w,x,cov,bounds,initial):
    '''
    Fit w with params0*x^(1-params1)-params2
    it just uses the diagonal of the covariance matrix for the fit.

    INPUT:
    w: correlation function
    x: theta array
    cov: covariance matrix

    OUTPUT:
    dictionary. fitted: fitted function computed in x; params: parameters of the fit.
    '''


    err=np.zeros(cov.shape[0])
    for i in range(cov.shape[0]):
        err[i]=np.sqrt(cov[i,i])
    mask=(np.isfinite(w))
    w=w[mask]
    err=err[mask]
    x=x[mask]
    mask=(np.isfinite(err))
    w=w[mask]
    err=err[mask]
    x=x[mask]


    params_guess= np.array(initial)
    bnds = np.array(bounds)

    popt, pcov = curve_fit(fitting_powerlaw, x, w, sigma=err,bounds=bnds,p0=params_guess)

    params=popt


    '''
    params_guess=[0.01,2,0.01]
    bnds = ((-0.01, 1.), (1.,3), (-0.5,0.5))

    #try to integrate with SLSQP. if it fails, it tries with Nelder-Mead
    obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,1),method= 'SLSQP', bounds=bnds, options={'maxiter': 300})
    if obj_minim.success==False:
        #add some noise to diagonal (0.25)
        cov1=copy.deepcopy(cov)
        cov1=cov1+cov1/4.
        obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,1),method= 'SLSQP', bounds=bnds, options={'maxiter': 300})
    if obj_minim.success==False:
        #print ('fail')
        obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,1),method= 'Nelder-Mead', options={'maxiter': 300})

    params =obj_minim.x
    '''
    fitted=(params[0]*(x)**(1-params[1]))-params[2]
    #print 'fitting',w,x, fitted,params
    return {'fitted':fitted,
            'params':params}

def fit_fixed(w,x,cov,index,bounds,initial):
    '''
    Fit w with params0*x^(1-index)-params1
    it just uses the diagonal of the covariance matrix for the fit.

    INPUT:
    w: correlation function
    x: theta array
    cov: covariance matrix

    OUTPUT:
    dictionary. fitted: fitted function computed in x; params: parameters of the fit.
    '''


    err=np.zeros(cov.shape[0])
    for i in range(cov.shape[0]):
        err[i]=np.sqrt(cov[i,i])
    mask=(np.isfinite(w))
    w=w[mask]
    err=err[mask]
    x=x[mask]
    mask=(np.isfinite(err))
    w=w[mask]
    err=err[mask]
    x=x[mask]

    params_guess=np.array([initial[0],initial[2],index])

    bnds = ([bounds[0][0],bounds[0][2],index-0.005],[bounds[1][0],bounds[1][2],index+0.005])

    #params_guess=[0.01,0.01]
    #bnds = ((-1, 1.), (-0.1,0.1))

    popt, pcov = curve_fit(fitting_powerlaw_fixed, x, w, sigma=err,bounds=bnds,p0=params_guess)

    params=popt


    #print (params)

    '''
    params_guess=[0.01,0.01]
    bnds = ((-1, 1.), (-0.1,0.1))

    #try to integrate with SLSQP. if it fails, it tries with Nelder-Mead
    obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,3,index),method= 'SLSQP', bounds=bnds, options={'maxiter': 200})
    if obj_minim.success==False:
        #print ('fail')
        obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,3,index),method= 'Nelder-Mead', options={'maxiter': 300})

    params =obj_minim.x
    '''
    fitted=(params[0]*(x)**(1-index))-params[1]
    params_new=np.zeros(3)
    params_new[0]=params[0]
    params_new[1]=index
    params_new[2]=params[1]

   # print 'fitting',w,x, fitted,params_new
    return {'fitted':fitted,
            'params':params_new}

def fit_rp(w,x,cov,bounds,initial):

    '''
    Fit w with (params0/x)^(params1)-params2
    it just uses the diagonal of the covariance matrix for the fit.

    INPUT:
    w: correlation function
    x: r_p array
    cov: covariance matrix

    OUTPUT:
    dictionary. fitted: fitted function computed in x; params: parameters of the fit.
    '''
    err=np.zeros(cov.shape[0])
    for i in range(cov.shape[0]):
        err[i]=np.sqrt(cov[i,i])

    mask=(np.isfinite(w))
    w=w[mask]
    err=err[mask]
    x=x[mask]
    mask=(np.isfinite(err))
    w=w[mask]
    err=err[mask]
    x=x[mask]

    for i in range(w.shape[0]):
        w[i]=w[i]/x[i]


    params_guess=initial

    bnds = bounds

    popt, pcov = curve_fit(fitting_powerlaw2, x, w, sigma=err,bounds=bnds,p0=params_guess)

    params=popt
    '''
    params_guess=[2,1.5,0.01]
    bnds = ([0.01, 200.], [-1,3],[-1,1])
    obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,2),method= 'SLSQP', bounds=bnds, options={'maxiter': 200})
    if obj_minim.success==False:
    #    print ('fail')
        obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,2),method= 'SLSQP', bounds=bnds, options={'maxiter': 500})
    #    obj_minim=mnn(minim_fun,params_guess, args=(x,w,cov,2,),method= 'Nelder-Mead', options={'maxiter': 300})

    params =obj_minim.x
    print ('try2',params)
    '''
    fitted=(params[0]/x)**(params[1])

    for i in range(len(w)):
        fitted[i]*=x[i]
        fitted[i]=fitted[i]-params[2]
        w[i]=w[i]*x[i]

    #print 'fitting',w,x, fitted,params
    return {'fitted':fitted,
            'params':params}

def fitting_powerlaw_fixed(x, para0,para1,index):
    return (para0*(x)**(1-index))-para1

def fitting_powerlaw(x, para0,para1,para2):
    return (para0*(x)**(1-para1))-para2
def fitting_powerlaw2(x, para0,para1,para2):
    return (para0/x)**(para1)-para2

# function to be minimized by Simplex
def minim_fun(params,xx,yy,cov,shape,index=1):
        chi2=0.

        if shape==1:
            funz=np.array(fitting_powerlaw(xx,params[0],params[1],params[2]))-yy

        if shape==2:
            funz=np.array(fitting_powerlaw2(xx,params[0],params[1],params[2]))-yy

        if shape==3:
            funz=np.array(fitting_powerlaw_fixed(xx,params[0],params[1],index))-yy

        funz=np.array(funz)

        for i in range(len(funz)):
            chi2+=funz[i]*funz[i]/cov[i,i]

        return chi2

def save_fit(correlation_optimized,correlation,output,label_save,verbose,label_dwn):
    '''
    It saves the fit to the CC and AC.
    '''

    if verbose:
        print('\n**** SAVE FIT MODULE ****')



    #print 'saving fit module'
    save_modes=['AC_R_R_fit','AC_U_fit','AC_R_P_fit','CC_A_fit','CC_P_fit']
    for i,tomo_bin in enumerate(correlation.keys()):
        for mode_k,save_mode in enumerate(save_modes):

            #create plot:
            n_rows=int(math.ceil(len(correlation[tomo_bin].keys())/4.))
            n_cols=4
            if n_rows==1:
                n_cols=2
                n_rows=2
            fig, ax = plt.subplots(n_rows,n_cols,sharex=True, sharey=True, figsize=(11,10))
            fig.subplots_adjust(wspace=0.,hspace=0.)
            #fig = plt.figure()
            kk=0
            xx=0

            for j,_ in enumerate(correlation[tomo_bin].keys()):
                reference_bin=str(xx*n_cols+kk+1+label_dwn)
                if verbose:
                    update_progress((mode_k*len(correlation[tomo_bin].keys())+np.float(j)+1)/ (4*len(correlation[tomo_bin].keys())))

                for correlation_type in correlation_optimized[tomo_bin][reference_bin].keys():
                    if correlation_type==save_mode:


                        #print correlation_type, tomo_bin,reference_bin
                        label=correlation_type.replace("fit", "")
                        x=copy.deepcopy(correlation[tomo_bin][reference_bin][label]['basis'])
                        w=copy.deepcopy(correlation[tomo_bin][reference_bin][label]['w'])
                        err=copy.deepcopy(correlation[tomo_bin][reference_bin][label]['err'])
                        ax[xx,kk].errorbar(x,w[:,0],err,color='black')

                    #    a=open((output+'params0_{0}_{1}_{2}_'+label_save).format(correlation_type,tomo_bin,reference_bin),'w')
                    #    b=open((output+'params1_{0}_{1}_{2}_'+label_save).format(correlation_type,tomo_bin,reference_bin),'w')
                    #    c=open((output+'params2_{0}_{1}_{2}_'+label_save).format(correlation_type,tomo_bin,reference_bin),'w')

                        for k in range(1,w.shape[1]):

                            p0=copy.deepcopy(correlation_optimized[tomo_bin][reference_bin][correlation_type]['params_0'][k])
                           # a.write('{0} \n '.format(p0))
                            p1=copy.deepcopy(correlation_optimized[tomo_bin][reference_bin][correlation_type]['params_1'][k])
                            #b.write('{0} \n '.format(p1))
                            p2=copy.deepcopy(correlation_optimized[tomo_bin][reference_bin][correlation_type]['params_2'][k])
                            #c.write('{0} \n '.format(p2))

                      #  print p0,p1,p2
                            if correlation_type=='AC_R_R_fit':
                                fit=x*((p0/x)**(p1))-p2

                            else:

                                fit=(p0*(x)**(1-p1))-p2
                            ax[xx,kk].plot(x,fit,color='red')


                        #ax[xx,kk].set_ylim([-0.4,1])




                        ax[xx,kk].xaxis.set_tick_params(labelsize=8)
                        ax[xx,kk].yaxis.set_tick_params(labelsize=8)
                        ax[xx,kk].set_xscale("log")

                        ax[xx,kk].text(0.1, 0.78, 'z_bin {0}'.format(reference_bin), verticalalignment='bottom', horizontalalignment='left',  transform=ax[xx,kk].transAxes,fontsize=7)


                        #cancel 1st and last number
                        xticks = ax[xx ,kk ].xaxis.get_major_ticks()
                        xticks[0].label1.set_visible(False)
                        xticks[-1].label1.set_visible(False)
                        xticks = ax[xx ,kk ].yaxis.get_major_ticks()
                        xticks[0].label1.set_visible(False)
                        xticks[-1].label1.set_visible(False)
                        kk+=1
                        if kk==n_cols:
                            kk=0
                            xx+=1



            plt.xlabel('x')
            plt.ylabel('w')
            plt.xscale('log')
            plt.savefig((output+'{0}_{1}_'+label_save+'.pdf').format(save_mode,tomo_bin), format='pdf',dpi=1000)
            plt.close()



#*************************************************************************
#                 NZ computation
#*************************************************************************


def compute_Nz(correlation_optimized,methods,bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt,jk_r,redshift,use_physical_scale_Newman,verbose,label_dwn):
    ''' compute the Nz for a number of methods.

        INPUT:
        correlation_optimized: dictionary conatining the vlaues of w(theta) integrated over angular scales/ or fitted.
        methods: methods for which is necessary to compute the Nz
        bias_correction_Menard,bias_correction_Newman,bias_correction_Schmidt: bias correction options.
        jkr: number of jackknife
        redshift: redshift array.

        OUTPUT:
        NZ: redshift distribution.
        BNZ: redshift distribution corrected for the bias bias evolution.

    '''
    if verbose:
        print('\n**** NZ MODULE ****')


    if bias_correction_Menard == 1 or bias_correction_Menard== 2:
        Menard={'methods':['CC_A_','AC_R_A_']}
        Menard_physical_scales={'methods':['CC_P_','AC_R_P_','CC_P_bias4']}
        Menard_physical_weighting={'methods':['CC_A_weighted','AC_R_A_weighted']}


    else:
        Menard={'methods':['CC_A_']}
        Menard_physical_scales={'methods':['CC_P_']}
        Menard_physical_weighting={'methods':['CC_A_']}


    if use_physical_scale_Newman:
        Newman={'methods':['CC_P_fit','AC_U_fit','AC_R_R_fit']}
    else:
        Newman={'methods':['CC_A_fit','AC_U_fit','AC_R_R_fit']}




    if bias_correction_Schmidt==3:
        Schmidt={'methods':['CC_D_','AC_R_P_']}
        #Schmidt={'methods':['CC_D_','AC_U_fit','AC_R_R_fit','CC_A_fit']}
    elif bias_correction_Schmidt==4:
        Schmidt={'methods':['CC_D_','AC_R_D_']}
    elif bias_correction_Schmidt==0:
        Schmidt={'methods':['CC_D_']}
    else :
        Schmidt={'methods':['CC_D_','AC_U_fit','AC_R_R_fit','CC_A_fit']}


    list_of_methods={'Menard':Menard,
                'Menard_physical_scales':Menard_physical_scales,
                'Menard_physical_weighting':Menard_physical_weighting,
                'Newman':Newman,
                'Schmidt':Schmidt}

    Nz_dict=dict()
    BNz_dict=dict()

    for labels in np.unique(methods):
        #try:

           # print np.unique(methods)
           # print list_of_methods[samp].keys()

           # for labels in list_of_methods[samp].keys():
                Nz_method_dict=dict()
                BNz_method_dict=dict()
                bias_method_dict=dict()
                for i,tomo_bin in enumerate(correlation_optimized.keys()):
                    Nz=np.zeros((len(redshift),jk_r+1))
                    BNz=np.zeros((len(redshift),jk_r+1))
                    bias=np.zeros((len(redshift),jk_r+1))

                   # print("\nmethods: "+labels)
                    for z,reference_bin in enumerate(redshift):
                       # update_progress((np.float(z)+1)/ len(redshift))
                        if labels=='Menard' or labels=='Menard_physical_scales':

                            for j in range(jk_r+1):
                                Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]

                                if bias_correction_Menard==1:
                                    BNz[z,j]=Nz[z,j]/correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]
                                    bias[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]
                                elif bias_correction_Menard==2 and labels=='Menard_physical_scales':
                                    BNz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['<w>'][j]

                        if labels=='Menard_phys_w':
                             for j in range(jk_r+1):
                                Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]
                                if bias_correction_Menard==1:
                                    BNz[z,j]=Nz[z,j]/correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]
                                    bias[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]


                        if labels=='Schmidt':
                            for j in range(jk_r+1):
                                if bias_correction_Schmidt==2 or bias_correction_Schmidt==1:
                                #old version;
                                #mute=correlation_optimized[tomo_bin][str(z+1)][list_of_methods[labels]['methods'][0]]['<w>'][j]
                                #Nz[z,j]=mute

                                #new version

                                    if bias_correction_Schmidt==2:
                                        exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j]+correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j])/2.


                                    else:
                                        exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['params_1'][j]+correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j])/2.
                                        #exp=(correlation_optimized[tomo_bin][str(z+1)][list_of_methods[labels]['methods'][3]]['params_1'][j])

                                    Dc=(cosmol.hubble_distance*cosmol.inv_efunc(redshift[z])).value
                                    h0=special.gamma(0.5)*special.gamma((exp-1)/0.5)/special.gamma(exp*0.5)
                                    dist=((1.+redshift[z])*cosmol.angular_diameter_distance(redshift[z]).value)**(1-exp)
                                    mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]

                                    Nz[z,j]=Dc*mute/(dist*h0)

                                    if bias_correction_Schmidt==1:
                                        BNz[z,j]=Nz[z,j]/((correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_0'][j])**exp)
                                    elif bias_correction_Schmidt==2:
                                        BNz[z,j]=Nz[z,j]/((correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_0'][j])**exp)

                                elif bias_correction_Schmidt==0:
                                    Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]

                                elif bias_correction_Schmidt==3:
                                    Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]
                                    BNz[z,j]=Nz[z,j]/correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]

                                elif bias_correction_Schmidt==4:
                                    Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]
                                    #print (z,j,Nz[z,j],correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j])
                                    BNz[z,j]=Nz[z,j]/correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['<w>'][j]



                        if labels=='Newman'  :
                            for j in range(jk_r+1):


                                if bias_correction_Newman==2:
                                    exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j]+correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j])/2.
                                else:
                                    exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['params_1'][j])

                                    #we shall get exp from the autocorrelation. Alternately, we
                                Dc=(cosmol.hubble_distance*cosmol.inv_efunc(redshift[z])).value
                                h0=special.gamma(0.5)*special.gamma((exp-1)/0.5)/special.gamma(exp*0.5)
                                dist=((1.+redshift[z])*cosmol.angular_diameter_distance(redshift[z]).value)**(1-exp)
                                mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['params_0'][j]
                                #print (z,mute,correlation_optimized[tomo_bin][str(z+1)][list_of_methods[labels]['methods'][0]]['params_1'][j])
                                Nz[z,j]=mute*Dc/(dist*h0)

                                if bias_correction_Newman!=0:
                                    r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_0'][j]
                                    exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j]
                                    exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['params_1'][j]
                                    r0_u=r0_r
                                    r0_ur=np.sqrt((r0_r**exp_r)*(r0_u**exp_u))
                                    BNz[z,j]=Nz[z,j]/r0_ur

                                #for the iterative procedure we need to pass: params[j],

                    #p#rint (Nz[:,0],BNz[:,0],BNz[:,0]/Nz[:,0])


                    #ITERATIVE PROCEDURE:
                    if (labels=='Newman' and bias_correction_Newman==1) or (labels=='Schmidt' and bias_correction_Schmidt==1):
                        r0_new=np.zeros((Nz.shape[0],Nz.shape[1]))
                        r0_new_j=np.zeros(Nz.shape[1])


                        for j in range(jk_r+1):

                                for z,reference_bin in enumerate(redshift):

                                    r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_0'][j]
                                    exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j]
                                    exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['params_1'][j]


                                    if labels=='Schmidt':
                                        exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['params_1'][j]+correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j])/2.
                                    else:
                                        exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['params_1'][j])
                                    h0=special.gamma(0.5)*special.gamma((exp-1)/0.5)/special.gamma(exp*0.5)
                                    Dc=(cosmol.hubble_distance*cosmol.inv_efunc(redshift[z])).value
                                    dist=((1.+redshift[z])*cosmol.angular_diameter_distance(redshift[z]).value)**(1-exp)
                                    if labels=='Schmidt':
                                        mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['<w>'][j]
                                    else:
                                        mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][0]]['params_0'][j]

                                    r0_new[z,j]=(((mute*Dc/(h0*dist*BNz[z,j]))**2)/(r0_r**exp_r))**(1./exp_u)
                                r0_new_j[j]=np.mean(r0_new[:,j])

                                for z,reference_bin in enumerate(redshift):
                                    r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_0'][j]
                                    exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][2]]['params_1'][j]
                                    exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)][list_of_methods[labels]['methods'][1]]['params_1'][j]
                                    r0_u=r0_new_j[j]
                                    r0_ur=np.sqrt((r0_r**exp_r)*(r0_u**exp_u))
                                    BNz[z,j]=Nz[z,j]/r0_ur

                    Nz_method_dict.update({'{0}'.format(i+1):Nz})
                    BNz_method_dict.update({'{0}'.format(i+1):BNz})

                Nz_dict.update({'{0}'.format(labels):Nz_method_dict})
                BNz_dict.update({'{0}'.format(labels):BNz_method_dict})
      #  except:
            #exception: method not in list of methods.
      #      print('-> Exception: No method {0} implemented - check for spelling'.format(labels))


    return   Nz_dict,BNz_dict

def stacking(Nz_tomo,BNz_tomo,z,jk_r,zp_t):

    '''
    It stacks different tomo bins for each method,after having normalized.
    INPUT:
    Nz_tomo,BNZ_tomo: redshift distribution dictionaries, output of compute_NZ module
    bias_correction: keyword. If sets, it stacks the Nz bias corrected
    z: redshift array
    jkr: number of jackknifes

    OUTPUT:
    Nz,BNz: dictionaries of redshift distributions.
    '''

    Nz=dict()
    BNz=dict()
    for method in Nz_tomo.keys():
        uNz=np.zeros((len(z),jk_r+1))
        uBNz=np.zeros((len(z),jk_r+1))
        for i,tomo in enumerate(Nz_tomo[method].keys()):
            norm=len(zp_t[tomo])
            for k in range(jk_r+1):
                Nz_tomo[method][tomo][:,k]=Nz_tomo[method][tomo][:,k]*norm/np.sum(Nz_tomo[method][tomo][:,k])
                uNz[:,k]+=Nz_tomo[method][tomo][:,k]





                if np.sum(BNz_tomo[method][tomo][:,k])>0.:
                    BNz_tomo[method][tomo][:,k]=BNz_tomo[method][tomo][:,k]*norm/np.sum(BNz_tomo[method][tomo][:,k])
                    uBNz[:,k]+=BNz_tomo[method][tomo][:,k]

        Nz.update({'{0}'.format(method):uNz})
        BNz.update({'{0}'.format(method):uBNz})

    return Nz,BNz


#*************************************************************************
#                 plotting & saving
#*************************************************************************

def plot(z,z_bin,zp_t_TOT,Nz,N,label_save,output,jk_r,gaussian_process,set_to_zero,mcmc_negative,only_diagonal,verbose,save_fig=1):
    '''
    plot the Nz and save the outputs. It also computes all the relevant statistics.

    INPUT:
    zp_t_TOT: redshift array of the true distribution
    z: redshift
    Nz: WZ redshift distribution
    N: true redshift distribution (for chi square)
    label:  label for the output file
    output: folder for the output file
    jk_r: number of jackknife regions
    gaussian_process: keyword. If set, it uses gaussian process to fit data points.
    set_to_zero: keyword. Set to zero negative points when performing gaussian process.
    mcmc_negative: keyword. It imposes a prior of N(z)>0
    save_fig: if 0, doe not save as pdf the covariance and the Nz.
              if 1, it saves only the Nz. If 2, it saves both. Default is 1

    OUTPUT:
    statistics: dictionary with all the statistics.


    '''





    #compute covariance
    dict_2=covariance_jck(Nz[:,1:],jk_r)

    if save_fig==2:
        plt.pcolor(dict_2['corr'])
        plt.colorbar()
        plt.savefig((output+'/cor_tot_{0}.pdf').format(label_save), format='pdf', dpi=1000)
        plt.close()



    if gaussian_process:
        try:
            with Silence(stdout='gaussian_log.txt', mode='w'):
                dict_stat_gp,rec,theta,rec1,theta1,cov_gp=gaussian_process_module(z,Nz[:,0],dict_2['err'],dict_2['cov'],N,set_to_zero)



        except:
            print ("gaussian process failed")
            dict_stat_gp=None
            gaussian_process=False
    else:
        dict_stat_gp=None

    #compute statistics.
    dict_stat=compute_statistics(z_bin,z,N,Nz[:,0],dict_2['cov'],Nz[:,1:])

    if mcmc_negative:
        Nz_corrected,sigma_dwn,sigma_up,mean_z,sigma_mean_dwn,sigma_mean_up,std_z,std_dwn,std_up=negative_emcee(z,dict_2['cov'],Nz[:,0])

    if save_fig>=1:
        fig= plt.figure()
        ax = fig.add_subplot(111)
        plt.hist(zp_t_TOT,bins=z_bin,color='blue',alpha=0.4,label='True distribution',histtype='stepfilled',edgecolor='None')

        #colors=['red','blue','green','black','yellow']
        #for key in N_z_dict.keys():
        #    plt.hist(N_z_dict[key],bins=z_bin,color=colors[int(key)-1],alpha=0.4,label='Z_{0}'.format(key),histtype='step',edgecolor=colors[int(key)-1])

        plt.errorbar(z,Nz[:,0],dict_2['err'],fmt='o',color='black',label='clustz')

        if gaussian_process:

            plt.plot(rec[:,0], rec[:,1], 'k', color='#CC4F1B',label='gaussian process')
            plt.fill_between(rec[:,0], rec[:,1]-rec[:,2], rec[:,1]+rec[:,2],
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


        if mcmc_negative:
            ytop = sigma_up-Nz_corrected
            ybot = Nz_corrected-sigma_dwn
            plt.errorbar(z,Nz_corrected,yerr=(ybot, ytop),fmt='o',color='red',label='mcmc corrected')

        plt.xlim(min(z-0.1),max(z+0.4))
        plt.xlabel('$z$')
        plt.ylabel('$N(z)$')


        #put text where I want
        mute_phi=max(Nz[:,0])
        mute_z=max(z)


        label_diag=''
        if only_diagonal:
            label_diag='_diag'
        ax.text(0.8, 0.9,'<z>_pdf_bin='+str(("%.3f" % dict_stat['mean_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.85,'<z>_clustz='+str(("%.3f" % dict_stat['mean_rec']))+'+-'+str(("%.3f" % dict_stat['mean_rec_err'+label_diag])),fontsize=11, ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.8,'median_pdf_bin='+str(("%.3f" % dict_stat['median_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.75,'median_clustz='+str(("%.3f" % dict_stat['median_rec']))+'+-'+str(("%.3f" % dict_stat['median_rec_err'])),fontsize=11, ha='center', transform=ax.transAxes)

        ax.text(0.8, 0.7,'std_pdf='+str(("%.3f" % dict_stat['std_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.65,'std_clustz='+str(("%.3f" % dict_stat['std_rec']))+'+-'+str(("%.3f" % dict_stat['std_rec_err'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.6,'$\chi^2/dof=$'+str(("%.3f" % dict_stat['chi_reduced'])),fontsize=11 , ha='center', transform=ax.transAxes)

        if gaussian_process:
            ax.text(0.8, 0.55,'<z>_clustz_GP='+str(("%.3f" % dict_stat_gp['mean_rec']))+'+-'+str(("%.3f" % dict_stat_gp['mean_rec_err'+label_diag])),fontsize=11, ha='center', transform=ax.transAxes)
            ax.text(0.8, 0.5,'std_clustz_GP='+str(("%.3f" % dict_stat_gp['std_rec']))+'+-'+str(("%.3f" % dict_stat_gp['std_rec_err'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)
            ax.text(0.8, 0.45,'median_clustz_GP='+str(("%.3f" % dict_stat_gp['median_rec'])),fontsize=11, ha='center', transform=ax.transAxes)

        if mcmc_negative:
      #  mean_z,sigma_mean_dwn,sigma_mean_up
            ax.text(0.8, 0.45,'<z>_clustz_mcmc='+str(("%.3f" % mean_z))+'+'+str(("%.3f" % sigma_mean_up))+'-'+str(("%.3f" % sigma_mean_dwn)),fontsize=11, ha='center', transform=ax.transAxes)
            ax.text(0.8, 0.4,'std_clustz_mcmc='+str(("%.3f" % std_z))+'+'+str(("%.3f" % (std_up)))+'-'+str(("%.3f" % (std_dwn))),fontsize=11 , ha='center', transform=ax.transAxes)


        plt.legend(loc=2,prop={'size':10},fancybox=True)


        plt.savefig((output+'/{0}.pdf').format(label_save), format='pdf', dpi=100)
        plt.close()




    save_wz(Nz,dict_2['cov'],z,z_bin,(output+'/{0}.h5').format(label_save))



    save_obj((output+'/statistics_{0}').format(label_save),dict_stat)
    if gaussian_process:
        save_obj((output+'/statistics_gauss_{0}').format(label_save),dict_stat_gp)

        save_wz(rec1,cov_gp,z,z_bin,(output+'/gaussian_{0}.h5').format(label_save))


        pd.DataFrame(rec[:,0]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'results')
        pd.DataFrame(rec[:,1:]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'err')






    return dict_stat,dict_stat_gp

def gaussian_process_module(z,Nz,err,cov,N,set_to_zero):


    Nz1=copy.deepcopy(Nz)
    err1=copy.deepcopy(err)

    def f_z(Nz,z,a,b):
        kernel_gp=np.zeros((len(z),len(z)))
        Ap=np.zeros((len(z),len(z)))
        cov_gp=np.zeros((len(z),len(z)))
        for i in range(len(z)):
            for j in range(len(z)):

                kernel_gp[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))
                if i==j:
                    Ap[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))+cov[i,j]
                else:
                    Ap[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))
                L = linalg.cholesky(Ap)

                b = linalg.solve(L, Nz)
                alpha=linalg.solve(np.transpose(L), b)
                mean = np.dot(np.transpose(kernel_gp), alpha) #equal to Nz_gp

        return mean

    #prior on theta[0]
    def prior_theta(theta,max_Nz):
        #top-hat priors on theta
        s1=theta[0]
        s2=theta[1]
        p=1
        if s1<max_Nz/1000. or  s1>max_Nz*2000.:
            p=0.
        if s2<0.01 or  s2>10.:
            p=0.



        return p


    # new x axis
    xmin=z[0]
    xmax=z[-1]
    nstar=len(z)*60

    #initial guess[sigma_l,L]
    initheta=[max(Nz),1.]


    #prior on theta[1]

    if set_to_zero:
        Nz1[Nz1<0.]=0.
        err1[Nz1<0.]=0.



    g=dgp.DGaussianProcess(z,Nz1,err1,cXstar=(xmin,xmax,nstar),prior=prior_theta,priorargs=(max(Nz)),grad='False')#,verbose=False) #)Xstar=z)
    g1=dgp.DGaussianProcess(z,Nz1,err1,Xstar=z,prior=prior_theta,priorargs=(max(Nz)),grad='False')#,verbose=False) #this is for the statistics.
    (rec,theta)=g.gp(theta=initheta)
    (rec1,theta1)=g1.gp(theta=initheta)



    Nz_gp=np.zeros(len(z))
    err_gp=np.zeros(len(z))



    for i in range(len(z)):
        Nz_gp[i]=rec1[i,1]
        err_gp[i]=rec1[i,2]
    kernel_gp=np.zeros((len(z),len(z)))
    Ap=np.zeros((len(z),len(z)))
    cov_gp=np.zeros((len(z),len(z)))
    for i in range(len(z)):
        for j in range(len(z)):

            kernel_gp[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))
            if i==j:
                Ap[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))+cov[i,j]
            else:
                Ap[i,j]=theta1[0]*theta1[0]*np.exp(-((z[i]-z[j])**2)/(2*theta1[1]*theta1[1]))




    #def compute_fz(Nz,z1,z2):

    L = linalg.cholesky(Ap)

    b = linalg.solve(L, Nz1)
    alpha=linalg.solve(np.transpose(L), b)
    mean = np.dot(np.transpose(kernel_gp), alpha) #equal to Nz_gp


    # calculate predictive standard deviation
    v = linalg.solve(L, kernel_gp)
    cov_gp = kernel_gp - np.dot(np.transpose(v), v)


    dict_stat_gp=compute_statistics(z_bin,z,N,Nz_gp,cov_gp,np.zeros((10,10)))


    return  dict_stat_gp,rec,theta,rec1,theta1,cov_gp

def negative_emcee(z,cov,Nz):

    '''

    It corrects for negative points with a positive prior
    TODO: traces as an output (they can be used to propagates uncertainties to FoM

    '''


    def stat(trace):
        L, xbins = np.histogram(trace, 300)
        print L
        bin_width=  0.5 * (xbins[1] - xbins[0])
        for j in range(len(xbins)):
            xbins[j] +=bin_width
        xbins=xbins[:-1]
        xmax=xbins[np.argmax(L)]
        L_tot_up=np.sum(L[np.argmax(L)+1:-1])+max(L)/2.
        L_tot_dwn=np.sum(L[0:np.argmax(L)])+max(L)/2.
        xmin=xbins[np.argmax(L)]
        xmax=xbins[np.argmax(L)]

        z_cum_min=max(L)/2.
        z_cum_max=max(L)/2.
        centr=np.argmax(L)
        up=True
        dwn=True
        for j in range(1,len(L)):

            if up and centr+j<len(L):
                z_cum_max+=L[centr+j]
                print z_cum_max/L_tot_up,xbins[centr+j],xbins[centr],j
                if z_cum_max/L_tot_up>0.68:

                    up=False
                    xmax=xbins[centr+j]


            if dwn and centr-j>=0:
                z_cum_min+=L[centr-j]
                if z_cum_max/L_tot_up>0.68:
                    dwn=False
                    xmin=xbins[centr-j]



        '''
        L[L == 0] = 1E-16

        shape = L.shape
        L = L.ravel()
        L=L*1.
        i_sort = np.argsort(L)[::-1]
        i_unsort = np.argsort(i_sort)


        L_cumsum = L[i_sort].cumsum()
        L_cumsum /= L_cumsum[-1]

        bin_width=  0.5 * (xbins[1] - xbins[0])

        for j in range(len(xbins)):
            xbins[j] +=bin_width
        xbins=xbins[:-1]

        xmin=xbins[np.argmax(L)]
        xmax=xbins[np.argmax(L)]
        new_L=L_cumsum[i_unsort].reshape(shape)


        for j in range(len(L_cumsum)):

            if new_L[i_sort[j]]<0.68:#0.955 :
                if xbins[i_sort[j]]<xmin:
                    xmin=xbins[i_sort[j]]
                if xbins[i_sort[j]]>xmax:
                    xmax=xbins[i_sort[j]]
        '''
        return xbins[np.argmax(L)],xmin-bin_width,xmax+bin_width

    def trace_nz_to_trace_mean(trace,z):

        def compute_mean_Std(y,x):
            mute_mean=0.
            mute_norm=0.
            mute_std=0.
            for jk in range(len(z)):
                mute_mean+=y[jk]*x[jk]
                mute_norm+=y[jk]
            mute_mean=mute_mean/mute_norm


            for jk in range(len(x)):
                mute_std+=y[jk]*(x[jk]-mute_mean)*(x[jk]-mute_mean)
            mute_std=np.sqrt(mute_std/mute_norm)
            return mute_mean,mute_std



        mean_trace=np.zeros(trace.shape[0])
        std_trace=np.zeros(trace.shape[0])

        for k in range(trace.shape[0]):

            mean_trace[k],std_trace[k]=compute_mean_Std(trace[k,:],z)
            #print mean_trace[k],std_trace[k],k


        return mean_trace,std_trace



    def lnprior(x):
        for i in range(len(x)):
            if x[i]<0.:
                return -np.inf
        return 0

    def lnlike(theta,Nz,cov):
        inv_sigma2 =0.
        for i in range(len(Nz)):
            inv_sigma2+=((theta[i]-Nz[i])**2.)/cov[i,i]
        return -0.5*inv_sigma2


    def lnprob(theta,Nz,cov_diag):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + lnlike(theta,Nz,cov)

    Nz_corrected=np.zeros(len(Nz))
    sigma_dwn=np.zeros(len(Nz))
    sigma_up=np.zeros(len(Nz))



    Nz1=np.zeros(len(z))
    for i in range(len(z)):
        if Nz[i]>=0.:
            Nz1[i]=Nz[i]
        else:
            Nz1[i]=0.
    cov_diag=np.zeros((len(z),len(z)))
    for i in range(len(z)):
        cov_diag[i,i]=cov[i,i]


    #defin dimensionality and number of walkers
    ndim, nwalkers = len(z), 500

    #initial position
    pos = [Nz1 + 1e-2*max(Nz)*(np.random.randn(ndim))**2. for i in range(nwalkers)]


    #set the mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Nz,cov_diag))

    #run the walkers
    sampler.run_mcmc(pos, 5000)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim)) #output: walkers,steps,dimensions) sample[

    #read traces
    mean_trace,std_trace=trace_nz_to_trace_mean(samples,z)
    mean_z,sigma_mean_dwn,sigma_mean_up=stat(mean_trace)

    std_z,std_dwn,std_up=stat(std_trace)

    for i in range(len(z)):

        Nz_corrected[i],sigma_dwn[i],sigma_up[i]=stat(samples[:,i])



    #compute also mean and std!
    return Nz_corrected,sigma_dwn,sigma_up,mean_z,mean_z-sigma_mean_dwn,sigma_mean_up-mean_z,std_z,std_z-std_dwn,std_up-std_z



#*************************************************************************
#                covariance & statistics
#*************************************************************************

def covariance_jck(TOTAL_PHI,jk_r):

  #  Covariance estimation

  average=np.zeros(TOTAL_PHI.shape[0])
  cov_jck=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
  err_jck=np.zeros(TOTAL_PHI.shape[0])


  for kk in range(jk_r):
    average+=TOTAL_PHI[:,kk]
  average=average/(jk_r)

 # print average
  for ii in range(TOTAL_PHI.shape[0]):
     for jj in range(ii+1):
          for kk in range(jk_r):
            cov_jck[jj,ii]+=TOTAL_PHI[ii,kk]*TOTAL_PHI[jj,kk]

          cov_jck[jj,ii]=(-average[ii]*average[jj]*jk_r+cov_jck[jj,ii])*(jk_r-1)/(jk_r)
          cov_jck[ii,jj]=cov_jck[jj,ii]

  for ii in range(TOTAL_PHI.shape[0]):
   err_jck[ii]=np.sqrt(cov_jck[ii,ii])
 # print err_jck

  #compute correlation
  corr=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
  for i in range(TOTAL_PHI.shape[0]):
      for j in range(TOTAL_PHI.shape[0]):
        corr[i,j]=cov_jck[i,j]/(np.sqrt(cov_jck[i,i]*cov_jck[j,j]))

  average=average*(jk_r)/(jk_r-1)
  return {'cov' : cov_jck,
          'err' : err_jck,
          'corr':corr,
          'mean':average}

def covariance_scalar_jck(TOTAL_PHI,jk_r):

  #  Covariance estimation

  average=0.
  cov_jck=0.
  err_jck=0.


  for kk in range(jk_r):
    average+=TOTAL_PHI[kk]
  average=average/(jk_r)

  for kk in range(jk_r):
    #cov_jck+=TOTAL_PHI[kk]#*TOTAL_PHI[kk]

    cov_jck+=(-average+TOTAL_PHI[kk])*(-average+TOTAL_PHI[kk])


  err_jck=np.sqrt(cov_jck*(jk_r-1)/(jk_r))


  #average=average*(jk_r)/(jk_r-1)
  return {'cov' : cov_jck,
          'err' : err_jck,
          'mean': average}



def compute_median(z,N,z_edges):
    median_value=0.
    norm=0.

    for i in range(len(N)):
        norm+=(z_edges[i+1]-z_edges[i])*N[i]

    median_t=False
    median=0.
    for i in range(len(N)):
        for j in range(100):
            median_value+=((z_edges[i+1]-z_edges[i])/100.)*N[i]
            if median_value>norm/2 and not median_t:
                median=j*((z_edges[i+1]-z_edges[i])/100.)+z_edges[i]
                median_t=True
    return median

def compute_statistics(z_edges,ztruth,N,phi_sum,cov,Njack=np.zeros((10,10))):


    #** compute true mean and true std ************************
    mean_true=0.
    norm_mean_true=0.
    std_true=0.

    for jk in range(len(ztruth)):
        mean_true+=N[jk]*ztruth[jk]
        norm_mean_true+=N[jk]

    mean_true=mean_true/norm_mean_true

    for jk in range(len(ztruth)):
        std_true+=N[jk]*(ztruth[jk]-mean_true)*(ztruth[jk]-mean_true)
    std_true=np.sqrt(std_true/norm_mean_true)



    # compute median statistics *************************
    median_true=compute_median(ztruth,N,z_edges)

    median_rec=compute_median(ztruth,phi_sum,z_edges)
    median_err=0.


    if np.sum(Njack)>0.01:

        median_jck=np.zeros(Njack.shape[1])
        for i in range(Njack.shape[1]):
            median_jck[i]=compute_median(ztruth,Njack[:,i],z_edges)
        dict_med=covariance_scalar_jck(median_jck,Njack.shape[1])
        median_err=dict_med['err']

    #** compute rec mean and std ************************
    mean_bin=0.
    norm_mean_bin=0.
    mean_cov=0.
    mean_cov_diag=0.
    #compute rec mean
    for k in range(len(phi_sum)):
        mean_bin += phi_sum[k]*ztruth[k]
        norm_mean_bin += phi_sum[k]
    mean_bin=mean_bin/norm_mean_bin


    for k in range(len(phi_sum)):
        for i in range(len(phi_sum)):
            if i==k:
                mean_cov_diag+=((cov[k,i])*(norm_mean_bin*ztruth[k]-mean_bin*norm_mean_bin)*(norm_mean_bin*ztruth[i]-mean_bin*norm_mean_bin))
            mean_cov+=((cov[k,i])*(norm_mean_bin*ztruth[k]-mean_bin*norm_mean_bin)*(norm_mean_bin*ztruth[i]-mean_bin*norm_mean_bin))
    mean_cov=np.sqrt(mean_cov)/(norm_mean_bin*norm_mean_bin)
    mean_cov_diag=np.sqrt(mean_cov_diag)/(norm_mean_bin*norm_mean_bin)

    square_root=0.

    for k in range(len(phi_sum)):
        square_root+=(phi_sum[k])*(ztruth[k]-mean_bin)**2
    square_root=np.sqrt(square_root/norm_mean_bin)

    error_variance_cov_diag=0.
    error_variance_cov=0.
    for k in range(len(phi_sum)):
        for i in range(len(phi_sum)):
            if i==k:
                error_variance_cov_diag+=(((ztruth[k]-mean_bin)*(ztruth[i]-mean_bin)-square_root**2)*((cov[k,i])))
            error_variance_cov+=(((ztruth[k]-mean_bin)*(ztruth[i]-mean_bin)-square_root**2)*((cov[k,i])))

    error_variance_cov=np.sqrt(error_variance_cov)/(2*square_root*norm_mean_bin)
    error_variance_cov_diag=np.sqrt(error_variance_cov_diag)/(2*square_root*norm_mean_bin)

    # mean error jackknife ****************************************************

    mean_jck_err=0.
    mean_jck=np.zeros(Njack.shape[1])
    for i in range(Njack.shape[1]):
        mean_true1=0
        norm1=0.
        for jk in range(len(ztruth)):
            mean_true1+=Njack[jk,i]*ztruth[jk]
            norm1+=Njack[jk,i]
        mean_jck[i]=mean_true1/norm1
        dict_sc=covariance_scalar_jck(mean_jck,Njack.shape[1])
        mean_jck_err=dict_sc['err']


    # theory with covariance **************************************************
    chi2_val=0

    for i in range(len(N)):

       chi2_val+=(N[i] - phi_sum[i])*(N[i] - phi_sum[i])/cov[i,i]
    chi2_val /=(len(N)-1)

    return {'chi_reduced' : chi2_val,
          'mean_true' : mean_true,
          'std_true':std_true,
          'mean_rec' : mean_bin,
          'mean_rec_err': mean_cov,
          'mean_rec_err_jck': mean_jck_err,
          'mean_rec_err_diag': mean_cov_diag,
          'std_rec':square_root,
          'std_rec_err':error_variance_cov,
          'std_rec_err_diag':error_variance_cov_diag,
          'median_true':median_true,
          'median_rec':median_rec,
          'median_rec_err':median_err
          }

def save_wz(Nz,cov,z,z_bin,label_save):
    pd.DataFrame(Nz[:,0]).to_hdf(label_save, 'results')
    pd.DataFrame(Nz[:,1:]).to_hdf(label_save, 'jackknife')
    pd.DataFrame(cov).to_hdf(label_save, 'cov')
    pd.DataFrame(z).to_hdf(label_save, 'z')
    pd.DataFrame(z_bin).to_hdf(label_save, 'z_edges')

#  silent module **********************************************************
# WARNING! IT MIGHT NOT WORK WITH PYTHON 3.0

# http://code.activestate.com/recipes/577564-context-manager-for-low-level-redirection-of-stdou/
class Silence:
    """Context manager which uses low-level file descriptors to suppress
    output to stdout/stderr, optionally redirecting to the named file(s).

    >>> import sys, numpy.f2py
    >>> # build a test fortran extension module with F2PY
    ...
    >>> with open('hellofortran.f', 'w') as f:
    ...     f.write('''\
    ...       integer function foo (n)
    ...           integer n
    ...           print *, "Hello from Fortran!"
    ...           print *, "n = ", n
    ...           foo = n
    ...       end
    ...       ''')
    ...
    >>> sys.argv = ['f2py', '-c', '-m', 'hellofortran', 'hellofortran.f']
    >>> with Silence():
    ...     # assuming this succeeds, since output is suppressed
    ...     numpy.f2py.main()
    ...
    >>> import hellofortran
    >>> foo = hellofortran.foo(1)
     Hello from Fortran!
     n =  1
    >>> print "Before silence"
    Before silence
    >>> with Silence(stdout='output.txt', mode='w'):
    ...     print "Hello from Python!"
    ...     bar = hellofortran.foo(2)
    ...     with Silence():
    ...         print "This will fall on deaf ears"
    ...         baz = hellofortran.foo(3)
    ...     print "Goodbye from Python!"
    ...
    ...
    >>> print "After silence"
    After silence
    >>> # ... do some other stuff ...
    ...
    >>> with Silence(stderr='output.txt', mode='a'):
    ...     # appending to existing file
    ...     print >> sys.stderr, "Hello from stderr"
    ...     print "Stdout redirected to os.devnull"
    ...
    ...
    >>> # check the redirected output
    ...
    >>> with open('output.txt', 'r') as f:
    ...     print "=== contents of 'output.txt' ==="
    ...     print f.read()
    ...     print "================================"
    ...
    === contents of 'output.txt' ===
    Hello from Python!
     Hello from Fortran!
     n =  2
    Goodbye from Python!
    Hello from stderr

    ================================
    >>> foo, bar, baz
    (1, 2, 3)
    >>>

    """
    def __init__(self, stdout=os.devnull, stderr=os.devnull, mode='w'):
        self.outfiles = stdout, stderr
        self.combine = (stdout == stderr)
        self.mode = mode

    def __enter__(self):
        import sys
        self.sys = sys
        # save previous stdout/stderr
        self.saved_streams = saved_streams = sys.__stdout__, sys.__stderr__
        self.fds = fds = [s.fileno() for s in saved_streams]
        self.saved_fds = map(os.dup, fds)
        # flush any pending output
        for s in saved_streams: s.flush()

        # open surrogate files
        if self.combine:
            null_streams = [open(self.outfiles[0], self.mode, 0)] * 2
            if self.outfiles[0] != os.devnull:
                # disable buffering so output is merged immediately
                sys.stdout, sys.stderr = map(os.fdopen, fds, ['w']*2, [0]*2)
        else: null_streams = [open(f, self.mode, 0) for f in self.outfiles]
        self.null_fds = null_fds = [s.fileno() for s in null_streams]
        self.null_streams = null_streams

        # overwrite file objects and low-level file descriptors
        map(os.dup2, null_fds, fds)

    def __exit__(self, *args):
        sys = self.sys
        # flush any pending output
        for s in self.saved_streams: s.flush()
        # restore original streams and file descriptors
        map(os.dup2, self.saved_fds, self.fds)
        sys.stdout, sys.stderr = self.saved_streams
        # clean up
        for s in self.null_streams: s.close()
        for fd in self.saved_fds: os.close(fd)
        return False
