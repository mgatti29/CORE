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
from scipy import linalg
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
def make_wz_errors(path,resampling,number_of_bootstrap,pairs_resampling):
    '''
    take as an input pairs between subregions and creates resmapling and errors.
    '''
    pairs=load_obj(path)
    jck_N=pairs['jck_N']

    # errors ********************************
    if resampling=='jackknife':
        njk=pairs['DD_a'].shape[0]
    elif resampling=='bootstrap':
        njk=number_of_bootstrap

    DD_j=np.zeros((njk+1,pairs['DD_a'].shape[1]))
    DR_j=np.zeros((njk+1,pairs['DD_a'].shape[1]))
    RD_j=np.zeros((njk+1,pairs['DD_a'].shape[1]))
    RR_j=np.zeros((njk+1,pairs['DD_a'].shape[1]))
    #print (pairs['DD'].shape,DD_j.shape)
    ndd=(np.sum(jck_N[:,0]))*(np.sum(jck_N[:,1]))
    ndr=(np.sum(jck_N[:,0]))*(np.sum(jck_N[:,3]))
    nrd=(np.sum(jck_N[:,2]))*(np.sum(jck_N[:,1]))
    nrr=(np.sum(jck_N[:,2]))*(np.sum(jck_N[:,3]))
    norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]

    DD_j[0,:],DR_j[0,:],RD_j[0,:],RR_j[0,:]=pairs['DD'],pairs['DR']*norm[1],pairs['RD']*norm[2],pairs['RR']*norm[3]
    #****************************************
    if resampling=='bootstrap':
        if  os.path.exists('./output_dndz/bootstrap_indexes_{0}_{1}.pkl'.format(njk,pairs['DD_a'].shape[0])):
            bootstrap_dict=load_obj('./output_dndz/bootstrap_indexes_{0}_{1}'.format(njk,pairs['DD_a'].shape[0]))
        else:
            bootstrap_dict=dict()
            for jk in range(njk):

                bootstrap_indexes=np.random.random_integers(0, pairs['DD_a'].shape[0]-1,pairs['DD_a'].shape[0])
                bootstrap_dict.update({str(jk):bootstrap_indexes})
            save_obj(('./output_dndz/bootstrap_indexes_{0}_{1}').format(njk,pairs['DD_a'].shape[0]),bootstrap_dict)


    for jk in range(njk):
        if resampling=='jackknife':
            if pairs_resampling:
                fact=1.
            else:
                fact=2.

            ndd=(np.sum(jck_N[:,0])-jck_N[jk,0])*(np.sum(jck_N[:,1])-jck_N[jk,1])
            ndr=(np.sum(jck_N[:,0])-jck_N[jk,0])*(np.sum(jck_N[:,3])-jck_N[jk,3])
            nrd=(np.sum(jck_N[:,2])-jck_N[jk,2])*(np.sum(jck_N[:,1])-jck_N[jk,1])
            nrr=(np.sum(jck_N[:,2])-jck_N[jk,2])*(np.sum(jck_N[:,3])-jck_N[jk,3])
            norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]


            DD_j[jk+1,:]=(pairs['DD'][:]-pairs['DD_a'][jk,:]-fact*pairs['DD_c'][jk,:])*norm[0]
            #print((pairs['DD'][:]-pairs['DD_a'][jk,:]-pairs['DD_c'][jk,:])*norm[0],(pairs['DD'][:]-pairs['DD_a'][jk,:]-2.*pairs['DD_c'][jk,:])*norm[0])
            DR_j[jk+1,:]=(pairs['DR'][:]-pairs['DR_a'][jk,:]-fact*pairs['DR_c'][jk,:])*norm[1]
            RD_j[jk+1,:]=(pairs['RD'][:]-pairs['RD_a'][jk,:]-fact*pairs['RD_c'][jk,:])*norm[2]
            RR_j[jk+1,:]=(pairs['RR'][:]-pairs['RR_a'][jk,:]-fact*pairs['RR_c'][jk,:])*norm[3]

        elif resampling=='bootstrap':
            bootstrap_indexes=bootstrap_dict[str(jk)]
            N1,N2,N3,N4=0.,0.,0.,0.

            if pairs_resampling:
                fact=1.
            else:
                fact=0.

            for boot in range(pairs['DD_a'].shape[0]):

                DD_j[jk+1,:]+=pairs['DD_a'][bootstrap_indexes[boot],:]+fact*pairs['DD_c'][bootstrap_indexes[boot],:]
                DR_j[jk+1,:]+=pairs['DR_a'][bootstrap_indexes[boot],:]+fact*pairs['DR_c'][bootstrap_indexes[boot],:]
                RD_j[jk+1,:]+=pairs['RD_a'][bootstrap_indexes[boot],:]+fact*pairs['RD_c'][bootstrap_indexes[boot],:]
                RR_j[jk+1,:]+=pairs['RR_a'][bootstrap_indexes[boot],:]+fact*pairs['RR_c'][bootstrap_indexes[boot],:]

                N1+=jck_N[bootstrap_indexes[boot],0]
                N2+=jck_N[bootstrap_indexes[boot],1]
                N3+=jck_N[bootstrap_indexes[boot],2]
                N4+=jck_N[bootstrap_indexes[boot],3]

            ndd,ndr,nrd,nrr=N1*N2,N1*N4,N3*N2,N3*N4
            norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]
            DD_j[jk+1,:],DR_j[jk+1,:],RD_j[jk+1,:],RR_j[jk+1,:] = DD_j[jk+1,:]*norm[0],DR_j[jk+1,:]*norm[1],RD_j[jk+1,:]*norm[2],RR_j[jk+1,:]*norm[3]
            #print (DD_j[jk+1,:],DR_j[jk+1,:])

    return pairs['theta'],DD_j.T,DR_j.T,RD_j.T,RR_j.T,njk

def  resampling_pairs(pairs,resampling,number_of_bootstrap):
    pairs_scheme=pairs['pairs_scheme']

    jck_N=pairs['jck_N']
    nbins=pairs['w'].shape[0]
    if resampling=='jackknife':
        njk=pairs['w'].shape[1]-1
    elif  resampling=='bootstrap':
        njk=number_of_bootstrap
    DD_j,DR_j,RD_j,RR_j=np.zeros((nbins,njk)),np.zeros((nbins,njk)),np.zeros((nbins,njk)),np.zeros((nbins,njk))

    if resampling=='jackknife':
        for jk in range(njk):
            C = np.delete(pairs_scheme, jk, 0)
            C = np.delete(C, jk, 1)
            shp = C.shape
            C = C.reshape(shp[0]*shp[1], shp[2], shp[3])
            stack = np.sum(C, 0)
            ndd=(np.sum(jck_N[:,0])-jck_N[jk,0])*(np.sum(jck_N[:,1])-jck_N[jk,1])
            ndr=(np.sum(jck_N[:,0])-jck_N[jk,0])*(np.sum(jck_N[:,3])-jck_N[jk,3])
            nrd=(np.sum(jck_N[:,2])-jck_N[jk,2])*(np.sum(jck_N[:,1])-jck_N[jk,1])
            nrr=(np.sum(jck_N[:,2])-jck_N[jk,2])*(np.sum(jck_N[:,3])-jck_N[jk,3])
            norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]

            DD_j[:,jk],DR_j[:,jk],RD_j[:,jk],RR_j[:,jk] = stack[0,:]*norm[0],stack[1,:]*norm[1],stack[2,:]*norm[2],stack[3,:]*norm[3]

    if resampling=='bootstrap':
        #diagonalize the matrix (i.e.: split cross-corr pairs)

        for i4 in range(pairs_scheme.shape[3]): #nbins
            for i3 in range(pairs_scheme.shape[2]): #DD,DR,RD,RR
                for i2 in range(pairs_scheme.shape[1]): #jck
                    for i1 in range(pairs_scheme.shape[0]): #jck
                        if i1!=i2:
                            pairs_scheme[i1,i1,i3,i4]+=pairs_scheme[i1,i2,i3,i4]/2.
                            pairs_scheme[i2,i2,i3,i4]+=pairs_scheme[i1,i2,i3,i4]/2.
                            pairs_scheme[i1,i2,i3,i4]=0


        #resampling indexes

        if  os.path.exists('./output_dndz/bootstrap_indexes_{0}_{1}.pkl'.format(njk,pairs['w'].shape[1]-1)):
            bootstrap_dict=load_obj('./output_dndz/bootstrap_indexes_{0}_{1}'.format(njk,pairs['w'].shape[1]-1))
        else:
            bootstrap_dict=dict()
            for jk in range(njk):

                bootstrap_indexes=np.random.random_integers(0, pairs['w'].shape[1]-2,pairs['w'].shape[1]-1)
                bootstrap_dict.update({str(jk):bootstrap_indexes})
            save_obj(('./output_dndz/bootstrap_indexes_{0}_{1}').format(njk,pairs['w'].shape[1]-1),bootstrap_dict)

        for jk in range(njk):
            bootstrap_indexes=bootstrap_dict[str(jk)]
            #print (len(np.unique(bootstrap_indexes)))
            #print(len(bootstrap_indexes))
            #print(bootstrap_indexes.shape)
            N1,N2,N3,N4=0.,0.,0.,0.
            for boot in range(pairs['w'].shape[1]-1):
                DD_j[:,jk]+=pairs_scheme[bootstrap_indexes[boot],bootstrap_indexes[boot],0,:]
                DR_j[:,jk]+=pairs_scheme[bootstrap_indexes[boot],bootstrap_indexes[boot],1,:]
                RD_j[:,jk]+=pairs_scheme[bootstrap_indexes[boot],bootstrap_indexes[boot],2,:]
                RR_j[:,jk]+=pairs_scheme[bootstrap_indexes[boot],bootstrap_indexes[boot],3,:]
                N1+=jck_N[bootstrap_indexes[boot],0]
                N2+=jck_N[bootstrap_indexes[boot],1]
                N3+=jck_N[bootstrap_indexes[boot],2]
                N4+=jck_N[bootstrap_indexes[boot],3]
            ndd,ndr,nrd,nrr=N1*N2,N1*N4,N3*N2,N3*N4
            norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]
            DD_j[:,jk],DR_j[:,jk],RD_j[:,jk],RR_j[:,jk] = DD_j[:,jk]*norm[0],DR_j[:,jk]*norm[1],RD_j[:,jk]*norm[2],RR_j[:,jk]*norm[3]


    return DD_j,DR_j,RD_j,RR_j,njk



def load_w_treecorr(methods,N,reference_bins_interval,Nbins,thetmin,thetmax,bias_correction_reference_Menard,bias_correction_Newman,
                    bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,sum_schmidt,
                    use_physical_scale_Newman,w_estimator,resampling,resampling_pairs,number_of_bootstrap,verbose):
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


    if verbose:
        print('\n**** LOADING FILE MODULE ****')
        print('Checking the methods')

    toberead=[]


    if 'Newman' in methods:
        if use_physical_scale_Newman:
            toberead.append('CC_P_')
            toberead.append('AC_R_R_')
            toberead.append('AC_U_')
        else:
            toberead.append('CC_A_')
            toberead.append('AC_R_R_')
            toberead.append('AC_U_')

    if 'Schmidt' in methods:
        toberead.append('CC_D_')
        if bias_correction_reference_Schmidt!='None':
            toberead.append(bias_correction_reference_Schmidt)
        if bias_correction_unknown_Schmidt!='None':
            toberead.append(bias_correction_unknown_Schmidt)

    if 'Menard' in methods:
        toberead.append('CC_A_')
    if 'Menard_physical_scales' in methods:
        toberead.append('CC_P_')
    if ('Menard_physical_scales' in methods) or  ('Menard' in methods):
        if bias_correction_reference_Menard!='None':
            toberead.append(bias_correction_reference_Menard)
        if bias_correction_unknown_Menard!='None':
            toberead.append(bias_correction_unknown_Menard)

    toberead=np.unique(toberead)
    new_methods=np.unique(methods)


    #load max_rpar
    max_rpar=load_obj('./pairscount/max_rpar')
    if verbose:
        print('Loading files')
    correlation=dict()


    for i in N.keys():

        redshift_dict=dict()
        for j,mute in enumerate(reference_bins_interval[i]['z']):

            slice_method_dict=dict()
            for correlation_type in toberead:

                #try:

                    if correlation_type=='AC_U_':
                        pairs_path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format(correlation_type,Nbins,int(i)+1,1)

                    else:
                        pairs_path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format(correlation_type,Nbins,int(i)+1,j+1+reference_bins_interval[i]['label_dwn'])


                    if correlation_type=='CC_D_' or correlation_type=='AC_R_D_'  or correlation_type=='AC_U_D_':
                        # Resampling

                        #DD_j,DR_j,RD_j,RR_j,njk=resampling_pairs(pairs,resampling,number_of_bootstrap
                        theta,DD_j,DR_j,RD_j,RR_j,njk=make_wz_errors(pairs_path,resampling,number_of_bootstrap,resampling_pairs)

                        w=estimator(w_estimator,DD_j[:,0],DR_j[:,0],RD_j[:,0],RR_j[:,0])
                        #*****************************************
                        w_summed=np.zeros(njk+1)
                        DD_summed=np.zeros(njk+1)
                        DR_summed=np.zeros(njk+1)
                        RD_summed=np.zeros(njk+1)
                        RR_summed=np.zeros(njk+1)

                        for jck in range(njk+1):


                            DD_summed[jck]=np.sum(DD_j[thetmin:thetmax,jck])
                            DR_summed[jck]=np.sum(DR_j[thetmin:thetmax,jck])
                            RD_summed[jck]=np.sum(RD_j[thetmin:thetmax,jck])
                            RR_summed[jck]=np.sum(RR_j[thetmin:thetmax,jck])
                            if sum_schmidt:
                                w_summed[jck]=np.sum(pairs['w'][thetmin:thetmax,jck])
                            else:
                                w_summed[jck]=estimator(w_estimator,DD_summed[jck],DR_summed[jck],RD_summed[jck],RR_summed[jck])
                        w={'label':correlation_type,
                            'basis':None,
                            'w':w_summed,
                            'err':None,
                            'DD':DD_summed,
                            'DR':DR_summed,
                            'RD':RD_summed,
                            'RR':RR_summed,
                            'estimator':w_estimator}


                    else:


                        # resampling *****************************
                        theta,DD_j,DR_j,RD_j,RR_j,njk=make_wz_errors(pairs_path,resampling,number_of_bootstrap,resampling_pairs)


                        # compute covariance and errors
                        DD_dict_cov=covariance_jck(DD_j[thetmin:thetmax,:],njk,resampling)
                        DR_dict_cov=covariance_jck(DR_j[thetmin:thetmax,:],njk,resampling)
                        RD_dict_cov=covariance_jck(RD_j[thetmin:thetmax,:],njk,resampling)
                        RR_dict_cov=covariance_jck(RR_j[thetmin:thetmax,:],njk,resampling)


                        #'w':copy.deepcopy(pairs['w'][thetmin:thetmax,:]),

                        ww=np.zeros((DD_j[thetmin:thetmax,:].shape[0],njk+1))
                        for hh in range(DD_j[thetmin:thetmax,:].shape[0]):
                         for kk in range(njk+1):
                            ddw=copy.deepcopy(DD_j[thetmin+hh,kk])
                            drw=copy.deepcopy(DR_j[thetmin+hh,kk])
                            rdw=copy.deepcopy(RD_j[thetmin+hh,kk])
                            rrw=copy.deepcopy(RR_j[thetmin+hh,kk])
                            ww[hh,kk]=estimator(w_estimator,ddw,drw,rdw,rrw)
                            if correlation_type=='AC_R_R_':
                                ww[hh,kk]=ww[hh,kk]*max_rpar*2.


                        dict_cov=covariance_jck(ww[:,1:],ww.shape[1]-1,resampling)
                        w={'label':correlation_type,
                            'basis':copy.deepcopy(theta[thetmin:thetmax]),
                            #'w':copy.deepcopy(pairs['w'][thetmin:thetmax]),
                            'w':ww,
                            'err':copy.deepcopy(dict_cov['err']),
                            'cov':copy.deepcopy(dict_cov['cov']),
                            'DD':copy.deepcopy(DD_j[thetmin:thetmax,:]),
                            'DD_err':copy.deepcopy(DD_dict_cov['err']),
                            'DR':copy.deepcopy(DR_j[thetmin:thetmax,:]),
                            'DR_err':copy.deepcopy(DR_dict_cov['err']),
                            'RD':copy.deepcopy(RD_j[thetmin:thetmax,:]),
                            'RD_err':copy.deepcopy(RD_dict_cov['err']),
                            'RR':copy.deepcopy(RR_j[thetmin:thetmax,:]),
                            'RR_err':copy.deepcopy(RR_dict_cov['err']),
                            'estimator':w_estimator}

                    #Add the dictionary to the collection of dictionaries
                    slice_method_dict.update({'{0}'.format(correlation_type):w})

                #except:
                #    print('->Exception: files missing for /SLICE_'+str(i+1)+'/optimize_new/{2}_{3}_{4}_{5}_{6}'.format(names_final,i+1,correlation_type,Nbins,thetmin,thetmax,label_dwn+j+1))

            redshift_dict.update({'{0}'.format(reference_bins_interval[i]['label_dwn']+j+1):slice_method_dict})
        correlation.update({'{0}'.format(i):redshift_dict})


    return correlation,new_methods,njk


#*************************************************************************
#                optimization
#*************************************************************************

def optimize(correlation,methods,N,reference_bins_interval,bias_correction_reference_Menard,bias_correction_Newman,
                    bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,
                    weight_variance,pairs_weighting,fit_free,use_physical_scale_Newman,
                    bounds_CC_fit,initial_guess_CC_fit,bounds_AC_U_fit,initial_guess_AC_U_fit,bounds_AC_R_fit,
                    initial_guess_AC_R_fit,verbose,gamma=1):
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

                if correlation_type =='CC_A_' or correlation_type =='AC_R_A_' or correlation_type=='CC_P_' or correlation_type=='AC_R_P_' or correlation_type=='AC_U_P_':

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
                            DD_weighted=weight_w(DD[:,ik],x,DD_err,weight_variance,gamma)['integr']
                            DR_weighted=weight_w(DR[:,ik],x,DR_err,weight_variance,gamma)['integr']
                            RD_weighted=weight_w(RD[:,ik],x,RD_err,weight_variance,gamma)['integr']
                            RR_weighted=weight_w(RR[:,ik],x,RR_err,weight_variance,gamma)['integr']
                            mute_w=estimator(w_estimator,DD_weighted,DR_weighted,RD_weighted,RR_weighted)

                            w_weighted.append(mute_w)

                    else:
                        for ik in range(w.shape[1]):

                            w_weighted.append(weight_w(w[:,ik],x,err,weight_variance,gamma)['integr'])

                    w_weight={'<w>':w_weighted}

                    #UPDATE THE DICTIONARY WITH THE WEIGHTED W(THETA)
                    slice_method_dict.update({'{0}'.format(correlation_type):w_weight})


                if (correlation_type =='AC_U_' or correlation_type =='AC_R_R_'): #(not fit_free) and

                    paramsjk_0=[]
                    paramsjk_1=[]
                    paramsjk_2=[]


                    x=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['basis'])
                    w=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['w'])
                    cov=copy.deepcopy(correlation[tomo_bin][reference_bin][correlation_type]['cov'])

                    save_txt=open('./output_dndz/TOMO_'+str(int(tomo_bin)+1)+'/fit/fitparams_{0}_{1}_{2}.txt'.format(correlation_type,i,j),'w')
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


                if correlation_type =='CC_D_' or correlation_type =='AC_R_D_' or correlation_type =='AC_U_D_':

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

                    save_txt=open('./output_dndz/TOMO_'+str(int(tomo_bin)+1)+'/fit/fitparams_{0}_{1}_{2}.txt'.format(correlation_type,i,j),'w')
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

    mask=(err>0.00001)
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

def save_fit(N,reference_bins_interval,correlation_optimized,correlation,label_save,verbose):
    '''
    It saves the fit to the CC and AC.
    '''

    if verbose:
        print('\n**** SAVE FIT MODULE ****')

    save_modes=['AC_R_R_fit','AC_U_fit','AC_R_P_fit','CC_A_fit','CC_P_fit']
    for i,tomo_bin in enumerate(correlation.keys()):
        for mode_k,save_mode in enumerate(save_modes):
         if save_mode in correlation_optimized[tomo_bin][correlation_optimized[tomo_bin].keys()[0]].keys():
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
                reference_bin=str(xx*n_cols+kk+1+reference_bins_interval[tomo_bin]['label_dwn'])
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

            try: 
                plt.savefig(('./output_dndz/TOMO_'+str(int(tomo_bin)+1)+'/fit/'+'{0}_{1}_'+label_save+'.pdf').format(save_mode,tomo_bin), format='pdf',dpi=1000)
                plt.close()
            except:
                print "plot not saved"



#*************************************************************************
#                 NZ computation
#*************************************************************************


def compute_Nz(correlation_optimized,jk_r,methods,bias_correction_reference_Menard,bias_correction_Newman,
                    bias_correction_reference_Schmidt,bias_correction_unknown_Menard,bias_correction_unknown_Schmidt,
                                N,reference_bins_interval,use_physical_scale_Newman,verbose):

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


    Nz_dict=dict()
    BNz_dict=dict()

    for labels in np.unique(methods):
        #try:

                Nz_method_dict=dict()
                BNz_method_dict=dict()
                bias_method_dict=dict()
                for i,tomo_bin in enumerate(correlation_optimized.keys()):
                    Nz=np.zeros((len(reference_bins_interval[tomo_bin]['z']),jk_r+1))
                    BNz=np.zeros((len(reference_bins_interval[tomo_bin]['z']),jk_r+1))
                    bias=np.zeros((len(reference_bins_interval[tomo_bin]['z']),jk_r+1))


                    for z,reference_bin in enumerate(reference_bins_interval[tomo_bin]['z']):
                        label_dwn=reference_bins_interval[tomo_bin]['label_dwn']
                        for j in range(jk_r+1):




                            # N(Z) COMPUTATION: **********************************************************************
                            if labels=='Menard':
                                Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A_']['<w>'][j]
                            elif labels=='Menard_physical_scales':
                                Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P_']['<w>'][j]
                            elif labels=='Schmidt':
                                Nz[z,j]=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_D_']['<w>'][j]

                            if labels=='Newman'  :
                                if use_physical_scale_Newman:
                                    exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P_fit']['params_1'][j])
                                    Ampl=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P_fit']['params_0'][j]
                                else:
                                    exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A_fit']['params_1'][j])
                                    Ampl=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A_fit']['params_0'][j]

                                Dc=(cosmol.hubble_distance*cosmol.inv_efunc(reference_bins_interval[tomo_bin]['z'][z])).value
                                h0=special.gamma(0.5)*special.gamma((exp-1)/0.5)/special.gamma(exp*0.5)
                                dist=((1.+reference_bins_interval[tomo_bin]['z'][z])*cosmol.angular_diameter_distance(reference_bins_interval[tomo_bin]['z'][z]).value)**(1-exp)

                                Nz[z,j]=Ampl*Dc/(dist*h0)



                            #** BIAS CORRECTION! *****************************************************************************
                            trig1= (labels=='Menard' or labels == 'Menard_physical_scales') and (bias_correction_reference_Menard=='AC_R_R_' and bias_correction_unknown_Menard =='AC_U_')
                            trig2= (labels=='Schmidt') and (bias_correction_reference_Schmidt=='AC_R_R_' and bias_correction_unknown_Schmidt =='AC_U_')
                            trig3= (labels=='Newman') and (bias_correction_Newman==1)

                            if (trig1 or trig2 or trig3):
                                r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_1'][j]
                                exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_U_fit']['params_1'][j]
                                r0_u=r0_r
                                r0_ur=np.sqrt((r0_r**exp_r)*(r0_u**exp_u))
                                BNz[z,j]=Nz[z,j]/r0_ur

                            elif labels == 'Menard':
                                mute_r=1.
                                mute_u=1.
                                b_cor_r=False
                                b_cor_u=False
                                if bias_correction_reference_Menard!='None':
                                    b_cor_r=True
                                    if bias_correction_reference_Menard=='AC_R_R_':
                                        mute_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    else:
                                        mute_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][bias_correction_reference_Menard]['<w>'][j]
                                if bias_correction_unknown_Menard!='None':
                                    b_cor_u=True
                                    if bias_correction_unknown_Menard=='AC_R_R_':
                                        mute_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    else:
                                        mute_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)][bias_correction_unknown_Menard]['<w>'][j]
                                if b_cor_r or b_cor_u:
                                    biass=np.sqrt(mute_r*mute_u)
                                    BNz[z,j]=Nz[z,j]/biass

                            elif labels == 'Schmidt':
                                mute_r=1.
                                mute_u=1.
                                b_cor_r=False
                                b_cor_u=False
                                if bias_correction_reference_Schmidt!='None':
                                    b_cor_r=True
                                    if bias_correction_reference_Schmidt=='AC_R_R_':
                                        mute_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    else:
                                        mute_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)][bias_correction_reference_Schmidt]['<w>'][j]
                                if bias_correction_unknown_Schmidt!='None':
                                    b_cor_u=True
                                    if bias_correction_unknown_Schmidt=='AC_R_R_':
                                        mute_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    else:
                                        mute_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)][bias_correction_unknown_Schmidt]['<w>'][j]
                                if b_cor_r or b_cor_u:
                                    biass=np.sqrt(mute_r*mute_u)
                                    BNz[z,j]=Nz[z,j]/biass


                    #ITERATIVE PROCEDURE **********************************************************************
                    trig1= (labels=='Menard' or labels == 'Menard_physical_scales') and (bias_correction_reference_Menard=='AC_R_R_' and bias_correction_unknown_Menard =='AC_U_')
                    trig2= (labels=='Schmidt') and (bias_correction_reference_Schmidt=='AC_R_R_' and bias_correction_unknown_Schmidt =='AC_U_')
                    trig3= (labels=='Newman') and (bias_correction_Newman==1)

                    if (trig1 or trig2 or trig3):
                        r0_new=np.zeros((Nz.shape[0],Nz.shape[1]))
                        r0_new_j=np.zeros(Nz.shape[1])

                        for j in range(jk_r+1):

                                for z,reference_bin in enumerate(reference_bins_interval[tomo_bin]['z']):


                                    label_dwn=reference_bins_interval[tomo_bin]['label_dwn']
                                    r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_1'][j]
                                    exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_U_fit']['params_1'][j]

                                    if labels=='Schmidt' or labels=='Menard' or labels=='Menard_physical_Scales':
                                        exp=(exp_r+exp_u)/2.
                                    else:
                                        if use_physical_scale_Newman:
                                            exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P_fit']['params_1'][j])
                                            mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P']['params_0'][j]
                                        else:
                                            exp=(correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A_fit']['params_1'][j])
                                            mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A']['params_0'][j]

                                    h0=special.gamma(0.5)*special.gamma((exp-1)/0.5)/special.gamma(exp*0.5)
                                    Dc=(cosmol.hubble_distance*cosmol.inv_efunc(reference_bins_interval[tomo_bin]['z'][z])).value
                                    dist=((1.+reference_bins_interval[tomo_bin]['z'][z])*cosmol.angular_diameter_distance(reference_bins_interval[tomo_bin]['z'][z]).value)**(1-exp)


                                    if labels=='Schmidt':
                                        mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_D_']['<w>'][j]
                                    elif labels =='Menard':
                                        mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_A_']['<w>'][j]
                                    elif labels =='Menard_physical_scales':
                                        mute=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['CC_P_']['<w>'][j]


                                    r0_new[z,j]=(((mute*Dc/(h0*dist*BNz[z,j]))**2)/(r0_r**exp_r))**(1./exp_u)

                                r0_new_j[j]=np.mean(r0_new[:,j])

                                for z,reference_bin in enumerate(reference_bins_interval[tomo_bin]['z']):
                                    label_dwn=reference_bins_interval[tomo_bin]['label_dwn']
                                    r0_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_0'][j]
                                    exp_r=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_R_R_fit']['params_1'][j]
                                    exp_u=correlation_optimized[tomo_bin][str(z+1+label_dwn)]['AC_U_fit']['params_1'][j]
                                    r0_u=r0_new_j[j]
                                    r0_ur=np.sqrt((r0_r**exp_r)*(r0_u**exp_u))
                                    BNz[z,j]=Nz[z,j]/r0_ur

                    Nz_method_dict.update({'{0}'.format(tomo_bin):Nz})
                    BNz_method_dict.update({'{0}'.format(tomo_bin):BNz})

                Nz_dict.update({'{0}'.format(labels):Nz_method_dict})
                BNz_dict.update({'{0}'.format(labels):BNz_method_dict})
      #  except:
            #exception: method not in list of methods.
      #      print('-> Exception: No method {0} implemented - check for spelling'.format(labels))


    return   Nz_dict,BNz_dict

def stacking(Nz_tomo,BNz_tomo,jk_r,N):

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
        for i,tomo in enumerate(Nz_tomo[method].keys()):
            norm=len(N[tomo]['zp_t'])/np.sum(Nz_tomo[method][tomo][:,0])
            norm1=len(N[tomo]['zp_t'])/np.sum(BNz_tomo[method][tomo][:,0])
            for k in range(jk_r+1):
                Nz_tomo[method][tomo][:,k]=Nz_tomo[method][tomo][:,k]*norm

                if np.sum(BNz_tomo[method][tomo][:,k])>0.:
                    BNz_tomo[method][tomo][:,k]=BNz_tomo[method][tomo][:,k]*norm1#/np.sum(BNz_tomo[method][tomo][:,k])


    return Nz_tomo,BNz_tomo


#*************************************************************************
#                 plotting & saving
#*************************************************************************
def plot(resampling,reference_bins_interval,N,Nz_tomo,label_save,jk_r,gaussian_process,set_to_zero,mcmc_negative,only_diagonal,verbose,save_fig=1):
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



    for i,tomo in enumerate(Nz_tomo.keys()):
     Nz=Nz_tomo[tomo]
    # print (jk_r,Nz[:,1:].shape,resampling)
     if (np.sum(Nz)>0.1):
        dict_2=covariance_jck(Nz[:,1:],jk_r,resampling)
        output='./output_dndz/TOMO_{0}/Nz/'.format(int(tomo)+1)
        if save_fig==1:
            try:
                plt.pcolor(dict_2['corr'])
                plt.colorbar()
                plt.savefig((output+'/cor_tot_{0}.pdf').format(label_save), format='pdf', dpi=1000)
                plt.close()
            except:
                 print "plot not saved"
                 



        if gaussian_process:
            try:
                with Silence(stdout='gaussian_log.txt', mode='w'):
                    dict_stat_gp,rec,theta,rec1,theta1,cov_gp=gaussian_process_module(z,Nz[:,0],dict_2['err'],dict_2['cov'],N[tomo]['N'],set_to_zero)


            except:
                print ("gaussian process failed")
                dict_stat_gp=None
                gaussian_process=False
        else:
            dict_stat_gp=None

        #compute statistics.
        reference_bins_interval
        dict_stat=compute_statistics(reference_bins_interval[tomo]['z_edges'],reference_bins_interval[tomo]['z'],N[tomo]['N'],Nz[:,0],dict_2['cov'],resampling,Nz[:,1:])

        if mcmc_negative:
            Nz_corrected,sigma_dwn,sigma_up,mean_z,sigma_mean_dwn,sigma_mean_up,std_z,std_dwn,std_up=negative_emcee(z,dict_2['cov'],Nz[:,0])

        if save_fig>=1:
            try:
                fig= plt.figure()
                ax = fig.add_subplot(111)
                plt.hist(N[tomo]['zp_t'],bins=reference_bins_interval[tomo]['z_edges'],color='blue',alpha=0.4,label='True distribution',histtype='stepfilled',edgecolor='None')


                plt.errorbar(reference_bins_interval[tomo]['z'],Nz[:,0],dict_2['err'],fmt='o',color='black',label='clustz')

                if gaussian_process:

                    plt.plot(rec[:,0], rec[:,1], 'k', color='#CC4F1B',label='gaussian process')
                    plt.fill_between(rec[:,0], rec[:,1]-rec[:,2], rec[:,1]+rec[:,2],
                        alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


                if mcmc_negative:
                    ytop = sigma_up-Nz_corrected
                    ybot = Nz_corrected-sigma_dwn
                    plt.errorbar(reference_bins_interval[tomo]['z'],Nz_corrected,yerr=(ybot, ytop),fmt='o',color='red',label='mcmc corrected')

                plt.xlim(min(reference_bins_interval[tomo]['z']-0.1),max(reference_bins_interval[tomo]['z']+0.4))
                plt.xlabel('$z$')
                plt.ylabel('$N(z)$')


            #put text where I want
                mute_phi=max(Nz[:,0])
                mute_z=max(reference_bins_interval[tomo]['z'])


                label_diag=''
                #if only_diagonal:
                label_diag='_diag'
                ax.text(0.8, 0.9,'<z>_pdf_bin='+str(("%.3f" % dict_stat['mean_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
                ax.text(0.8, 0.85,'<z>_clustz='+str(("%.3f" % dict_stat['mean_rec']))+'+-'+str(("%.3f" % dict_stat['mean_rec_err'+label_diag]))+' ('+str(("%.3f" % dict_stat['mean_rec_err']))+')',fontsize=11, ha='center', transform=ax.transAxes)
                ax.text(0.8, 0.8,'median_pdf_bin='+str(("%.3f" % dict_stat['median_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
                ax.text(0.8, 0.75,'median_clustz='+str(("%.3f" % dict_stat['median_rec']))+'+-'+str(("%.3f" % dict_stat['median_rec_err'])),fontsize=11, ha='center', transform=ax.transAxes)

                ax.text(0.8, 0.7,'std_pdf='+str(("%.3f" % dict_stat['std_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
                ax.text(0.8, 0.65,'std_clustz='+str(("%.3f" % dict_stat['std_rec']))+'+-'+str(("%.3f" % dict_stat['std_rec_err'+label_diag]))+' ('+str(("%.3f" % dict_stat['std_rec_err']))+')',fontsize=11 , ha='center', transform=ax.transAxes)
                ax.text(0.8, 0.6,'$\chi^2=$'+str(("%.3f" % dict_stat['chi_diag']))+' ('+str(("%.3f" % dict_stat['chi']))+') [DOF: '+str(len(Nz[:,0]))+']',fontsize=11 , ha='center', transform=ax.transAxes)

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
            except:
                print "plot not saved"



        save_wz(Nz,dict_2['cov'],reference_bins_interval[tomo]['z'],reference_bins_interval[tomo]['z_edges'],(output+'/{0}.h5').format(label_save))



        save_obj((output+'/statistics_{0}').format(label_save),dict_stat)
        if gaussian_process:
            save_obj((output+'/statistics_gauss_{0}').format(label_save),dict_stat_gp)

            save_wz(rec1,cov_gp,reference_bins_interval[tomo]['z'],reference_bins_interval[tomo]['z_edges'],(output+'/gaussian_{0}.h5').format(label_save))


            pd.DataFrame(rec[:,0]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'results')
            pd.DataFrame(rec[:,1:]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'err')

     else:
         dict_stat,dict_stat_gp=None,None
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


    dict_stat_gp=compute_statistics(z_bin,z,N,Nz_gp,cov_gp,resampling,np.zeros((10,10)))


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

def covariance_jck(TOTAL_PHI,jk_r,type_cov):
  if type_cov=='jackknife':
      fact=(jk_r-1.)/(jk_r)

  elif type_cov=='bootstrap':
      fact=1./(jk_r)
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

          cov_jck[jj,ii]=(-average[ii]*average[jj]*jk_r+cov_jck[jj,ii])*fact
          cov_jck[ii,jj]=cov_jck[jj,ii]

  for ii in range(TOTAL_PHI.shape[0]):
   err_jck[ii]=np.sqrt(cov_jck[ii,ii])
 # print err_jck

  #compute correlation
  corr=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
  for i in range(TOTAL_PHI.shape[0]):
      for j in range(TOTAL_PHI.shape[0]):
        corr[i,j]=cov_jck[i,j]/(np.sqrt(cov_jck[i,i]*cov_jck[j,j]))

  average=average*fact
  return {'cov' : cov_jck,
          'err' : err_jck,
          'corr':corr,
          'mean':average}

def covariance_scalar_jck(TOTAL_PHI,jk_r,type_cov):
  if type_cov=='jackknife':
      fact=(jk_r-1.)/(jk_r)

  elif type_cov=='bootstrap':
      fact=1./(jk_r)
  #  Covariance estimation

  average=0.
  cov_jck=0.
  err_jck=0.


  for kk in range(jk_r):
    average+=TOTAL_PHI[kk]
  average=average/(jk_r)

  for kk in range(jk_r):
    #cov_jck+=TOTAL_PHI[kk]#*TOTAL_PHI[kk]

    cov_jck+=fact*(-average+TOTAL_PHI[kk])*(-average+TOTAL_PHI[kk])


  err_jck=np.sqrt(cov_jck)


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

def compute_statistics(z_edges,ztruth,N,phi_sum,cov,resampling,Njack=np.zeros((10,10))):


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
        dict_med=covariance_scalar_jck(median_jck,Njack.shape[1],resampling)
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
                error_variance_cov_diag+=(((ztruth[k]-mean_bin)*(ztruth[i]-mean_bin)-square_root**2)**2*((cov[k,i])))
            error_variance_cov+=(((ztruth[k]-mean_bin)*(ztruth[i]-mean_bin)-square_root**2)**2*((cov[k,i])))

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
        dict_sc=covariance_scalar_jck(mean_jck,Njack.shape[1],resampling)
        mean_jck_err=dict_sc['err']


    # theory with covariance **************************************************

    N_p=Njack.shape[1]
    p_p=N.shape[0]
    f_hartlap=(N_p-1)/(N_p-p_p-2)


    cv_chol = linalg.cholesky(cov, lower=True)
    cv_sol = linalg.solve(cv_chol, N - phi_sum, lower=True)
    cov_1=copy.deepcopy(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            if i!=j:
                cov_1[i,j]=0.
    cv_chol_diag = linalg.cholesky(cov_1, lower=True)
    cv_sol_diag = linalg.solve(cv_chol_diag, N - phi_sum, lower=True)


    chi2_val_diag= np.sum(cv_sol_diag ** 2)/f_hartlap
    chi2_val= np.sum(cv_sol ** 2)/f_hartlap


    return {'chi_diag' : chi2_val_diag,
          'chi' : chi2_val,
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
