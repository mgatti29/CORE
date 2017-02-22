import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as pyint
import os
import astropy
import pandas as pd
from scipy.interpolate import interp1d
import scipy.integrate as pyint
from .dataset import save_obj, load_obj, update_progress
from astropy.cosmology import *
from treejack_clust import Jack
import  pickle
import  math
import time
import timeit
import warnings
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from .functions_nz import weight_w,estimator

def run_pairs(corr_tobecomputed =['CC_A_','CC_P_','CC_D_','AC_U_','AC_R_A_','AC_R_P_','AC_R_R_'],
                pairs=['DD','DR','RD','RR'],
                fact_dist=2,
                Nbins=[10],
                min_theta=0.01, max_theta=1., min_rp=800,max_rp=800.,max_rpar=80.,
                overwrite=True,
                cosmology=Planck15,
                w_estimator='LS',
                time0=0, number_of_cores=2,
                **kwargs):



    # **************************************************************************************************************
    '''
    This version computes the cross-correlation and the autocorrelation for a number of cases.
    1) cross corr UNK-REF in angular bin
    2) cross corr UNK-REF in angular bin corresponding to given physical distances
    3) cross corr UNK-REF density (single bin estimation)
    4) auto corr UNK in angular bin
    5) auto corr REF in angular bin
    6) auto corr REF in angular bin corresponding to given physical distances
    7) projected auto corr REF (physical distances)

    the crosscorrelations and autocorrelations are computed for different angular binnings.

    names_output:                   label :           usage - Method: Menard/Alex        Menard physical        Newman       Schmidt
    #1) w_cross_angular_*            CC_A_                                 x       |           -           |      x     |       -
    #2) w_cross_physical_*           CC_P_                                 -       |           x           |      -     |       -
    #3) w_cross_density_*            CC_D_                                 -       |           -           |      -     |       x
    #4) w_auto_UNK_*                 AC_U_                                 -       |           -           |      x     |       x
    #5) w_auto_REF_angular_*         AC_R_A_                               x       |           -           |      -     |       -
    #6) w_auto_REF_physical_*        AC_R_P_                               -       |           x           |      x     |       x
    #7) w_auto_REF_rp_*              AC_R_R_                               -       |           -           |      x     |       x



    '''
    #TODO: make the update progress working also with the parallelization
    #*************************************************************************************************************


    if time0>0.:
        verbose=True
    else:
        verbose=False

    cosmol=impose_cosmology(cosmology)

    cosmol=cosmol

    # load catalogs
    hdf = pd.HDFStore('./pairscount/dataset.h5')
    reference = hdf['ref']
    reference_rndm = hdf['ref_random']
    unknown = hdf['unk']
    unknown_rndm = hdf['unk_random']
    hdf.close()

    # load redshift arrays
    unknown_bins_interval=load_obj('./pairscount/unknown_bins_interval')
    reference_bins_interval=load_obj('./pairscount/reference_bins_interval')

    # load jackknives centers
    centers=np.loadtxt('./pairscount/pairscounts_centers.txt')

    start_time=timeit.default_timer()
    #update_progress(0.)

    # save r_p and pass it to the dndz module
    save_obj('./pairscount/max_rpar',max_rpar)

    njk=len(np.unique(reference_rndm['HPIX']))

    # cycle over tomographic bins *****************************************************************

    for i,z_unk_value in enumerate(unknown_bins_interval['z']):

        ra_unk=np.array(unknown['RA'][unknown['bins']==i+1])
        dec_unk=np.array(unknown['DEC'][unknown['bins']==i+1])
        w_unk=np.array(unknown['W'][unknown['bins']==i+1])
        jck_unk=np.array(unknown['HPIX'][unknown['bins']==i+1])


        ra_unk_rndm=np.array(unknown_rndm['RA'][unknown_rndm['bins']==i+1])
        dec_unk_rndm=np.array(unknown_rndm['DEC'][unknown_rndm['bins']==i+1])
        w_unk_rndm=np.array(unknown_rndm['W'][unknown_rndm['bins']==i+1])
        jck_unk_rndm=np.array(unknown_rndm['HPIX'][unknown_rndm['bins']==i+1])




        #PARALLELIZATION OF THE REDSHIFT SLICES.

        pool  = mp.Pool(number_of_cores)
        iterable = range(reference_bins_interval['z'].shape[0])
        func = partial(redshift_slice,reference,reference_rndm,cosmol, corr_tobecomputed,overwrite, verbose,pairs,
                        ra_unk,dec_unk,w_unk,jck_unk,ra_unk_rndm,dec_unk_rndm,w_unk_rndm,jck_unk_rndm,
                        min_rp,max_rp,Nbins,min_theta,max_theta,max_rpar,fact_dist, centers,njk,w_estimator,reference_bins_interval['z'],i)

        pool.map(func, iterable)

        pool.close()
        pool.join()

        '''
        for j,mutee_j in enumerate(reference_bins_interval['z']):
            redshift_slice(reference,reference_rndm,cosmol, corr_tobecomputed,overwrite, verbose, pairs,
                        ra_unk,dec_unk,w_unk,jck_unk,ra_unk_rndm,dec_unk_rndm,w_unk_rndm,jck_unk_rndm,
                        min_rp,max_rp,Nbins,min_theta,max_theta,max_rpar,fact_dist, centers,njk,w_estimator,reference_bins_interval['z'],0,j)
        '''
    # plotting figures **************************************************

    for i,z_unk in enumerate (unknown_bins_interval['z']):
        plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'w',w_estimator)
        '''
        plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'DD')
        plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'DR')
        plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'RD')
        plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'RR')
        '''
        plot_bias(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,'w',w_estimator)
#****************************************************************************************************
#                       multiproccesing on each redshift slice
#****************************************************************************************************
def redshift_slice(reference,reference_rndm, cosmol,corr_tobecomputed,overwrite, verbose,pairs,
                    ra_unk,dec_unk,w_unk,jck_unk,
                    ra_unk_rndm,dec_unk_rndm,w_unk_rndm,jck_unk_rndm,
                    min_rp,max_rp,Nbins,min_theta,max_theta ,max_rpar, fact_dist, centers,njk,w_estimator,reference_bins_interval,i,j):

            # redshift of the slice
            if verbose:
                print('Running slice {0}'.format(j+1))
            z_ref_value=reference_bins_interval[j]

            ra_ref=np.array(reference['RA'][reference['bins']==j+1])
            dec_ref=np.array(reference['DEC'][reference['bins']==j+1])
            w_ref=np.array(reference['W'][reference['bins']==j+1])
            z_ref=np.array(reference['Z'][reference['bins']==j+1])
            jck_ref=np.array(reference['HPIX'][reference['bins']==j+1])

            '''
            ra_ref_rndm=np.array(reference_rndm['RA'][reference_rndm['bins']==j+1])
            dec_ref_rndm=np.array(reference_rndm['DEC'][reference_rndm['bins']==j+1])
            w_ref_rndm=np.array(reference_rndm['W'][reference_rndm['bins']==j+1])
            z_ref_rndm=np.array(reference_rndm['Z'][reference_rndm['bins']==j+1])
            jck_ref_rndm=np.array(reference_rndm['HPIX'][reference_rndm['bins']==j+1])
            '''
            ra_ref_rndm=np.array(reference_rndm['RA'][reference_rndm['bins']==j+1])

            dec_ref_rndm=np.array(reference_rndm['DEC'][reference_rndm['bins']==j+1])
            w_ref_rndm=np.array(reference_rndm['W'][reference_rndm['bins']==j+1])
            z_ref_rndm=np.array(reference_rndm['Z'][reference_rndm['bins']==j+1])
            jck_ref_rndm=np.array(reference_rndm['HPIX'][reference_rndm['bins']==j+1])

            #compute the angular equivalent to physical extrema depending on redshift
            min_theta_rp=(min_rp/(1000*(1.+z_ref_value)*cosmol.angular_diameter_distance(z_ref_value).value*(2*math.pi)/360))
            max_theta_rp=(max_rp/(1000*(1.+z_ref_value)*cosmol.angular_diameter_distance(z_ref_value).value*(2*math.pi)/360))


            # we shall check for this. It converts redshift into distances, needed to compute
            # the projected autocorrelation #TODO: put it in the dataset module.

            for ih in range(z_ref_rndm.shape[0]):
                z_ref_rndm[ih]=cosmol.angular_diameter_distance(z_ref_rndm[ih]).value
            for ih in range(z_ref.shape[0]):
                z_ref[ih]=cosmol.angular_diameter_distance(z_ref[ih]).value


            # cycle over the angular bins
            for nnn in range(len(Nbins)):

                conf = {'nbins': Nbins[nnn],
                    'min_sep': min_theta,
                    'max_sep': max_theta,
                    'sep_units':'degrees',
                    'bin_slop': 0.03#,
                    #'nodes': 2  #parameter for treecorr
                    }


                conf_physical = {'nbins': Nbins[nnn],
                            'min_sep': min_theta_rp,
                            'max_sep': max_theta_rp,
                            'sep_units':'degrees',
                            'bin_slop': 0.03#,
                            #'nodes': 10
                            }

                conf_density = {'nbins': Nbins[nnn],
                            'min_sep': min_rp,
                            'max_sep': max_rp,
                            'sep_units':'kpc',
                            'bin_slop': 0.03#,
                            #'nodes': 10
                            }

                conf_rp = {'nbins': Nbins[nnn],
                        'min_sep': min_rp/1000.,
                        'max_sep': max_rp/1000.,
                        'max_rpar' : max_rpar, #in Mpc
                        'min_rpar' : -max_rpar, #in Mpc
                        'bin_slop': 0.03#,
                        #'nodes': 10
                        }


                if 'CC_A_' in corr_tobecomputed :
                # 1) cross corr angular bins **********************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('CC_A_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('CC_A_',Nbins[nnn])
                        J = Jack(w_estimator,conf,pairs, ra_unk,dec_unk,ra_unk_rndm,dec_unk_rndm,ra_ref,dec_ref,
                        ra_ref_rndm,dec_ref_rndm,jck_unk,jck_ref,jck_unk_rndm, jck_ref_rndm,
                        w_unk,w_ref,w_unk_rndm,w_ref_rndm,fact_dist,z_ref_value,
                        'cross',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('CC_A_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)


                if 'CC_P_' in corr_tobecomputed :
                    # 2) cross corr physical   ************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('CC_P_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('CC_P_',Nbins[nnn])

                        J = Jack(w_estimator,conf_physical,pairs, ra_unk,dec_unk,ra_unk_rndm,dec_unk_rndm,ra_ref,dec_ref,
                        ra_ref_rndm,dec_ref_rndm,jck_unk,jck_ref,jck_unk_rndm, jck_ref_rndm,
                        w_unk,w_ref,w_unk_rndm,w_ref_rndm,fact_dist,z_ref_value,'cross',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('CC_P_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)


                if 'CC_D_' in corr_tobecomputed :
                    # 3) cross density #to be modified **************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('CC_D_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('CC_D_',Nbins[nnn])

                        J = Jack(w_estimator,conf_density,pairs,ra_unk,dec_unk,ra_unk_rndm,dec_unk_rndm,ra_ref,dec_ref,ra_ref_rndm,dec_ref_rndm,jck_unk,
                        jck_ref,jck_unk_rndm, jck_ref_rndm,w_unk,w_ref,w_unk_rndm,w_ref_rndm,fact_dist,z_ref_value,'density', centers=centers,
                        njk=njk,verbose=verbose)
                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('CC_D_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)

                if 'AC_R_D_' in corr_tobecomputed:
                    # 6) w_auto_REF_physical_*  **********************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('AC_R_D_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('AC_R_D_',Nbins[nnn])
                        J = Jack(w_estimator,conf_density, pairs, ra_ref,dec_ref,ra_ref_rndm,dec_ref_rndm,ra_ref,dec_ref,ra_ref_rndm,
                        dec_ref_rndm,jck_ref,jck_ref,jck_ref_rndm, jck_ref_rndm,w_ref,w_ref,w_ref_rndm,
                        w_ref_rndm,fact_dist,z_ref_value,'density',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_D_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)

                if 'AC_U_' in corr_tobecomputed :
                    # 4) w_auto_UNK ********************************************************************
                    # this has to be computed only once.
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('AC_U_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('AC_U_',Nbins[nnn])
                        if j==0:
                            J_auto = Jack(w_estimator,conf,pairs, ra_unk,dec_unk,ra_unk_rndm,dec_unk_rndm,ra_unk,dec_unk,
                            ra_unk_rndm,dec_unk_rndm,jck_unk,jck_unk,jck_unk_rndm,jck_unk_rndm,w_unk,w_unk,
                            w_unk_rndm,w_unk_rndm,fact_dist,z_ref_value,'auto',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                            pairs = J_auto.NNCorrelation()
                            path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_U_',Nbins[nnn],i+1,j+1)
                            save_obj(path,pairs)

                if 'AC_R_A_' in corr_tobecomputed:
                    # 5) w_auto_REF_angular  ************************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('AC_R_A_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('AC_R_A_',Nbins[nnn])
                        J = Jack(w_estimator,conf,pairs,  ra_ref,dec_ref,ra_ref_rndm,dec_ref_rndm,ra_ref,dec_ref,ra_ref_rndm,
                        dec_ref_rndm,jck_ref,jck_ref,jck_ref_rndm, jck_ref_rndm,w_ref,w_ref,w_ref_rndm,
                        w_ref_rndm,fact_dist,z_ref_value,'auto',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                        pairs = J.NNCorrelation()

                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_A_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)

                if 'AC_R_P_' in corr_tobecomputed:
                    # 6) w_auto_REF_physical_*  **********************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('AC_R_P_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('AC_R_P_',Nbins[nnn])
                        J = Jack(w_estimator,conf_physical,pairs,  ra_ref,dec_ref,ra_ref_rndm,dec_ref_rndm,ra_ref,dec_ref,ra_ref_rndm,
                        dec_ref_rndm,jck_ref,jck_ref,jck_ref_rndm, jck_ref_rndm,w_ref,w_ref,w_ref_rndm,
                        w_ref_rndm,fact_dist,z_ref_value,'auto',corr = 'NN', centers=centers, njk=njk,verbose=verbose)

                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_P_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)




                if 'AC_R_R_' in corr_tobecomputed:
                    #7) w_auto_REF_rp_*  *************************************************************
                    if not overwrite and os.path.exists(('./pairscount/pairs/{0}_{1}_{2}_{3}.pkl').format('AC_R_R_',Nbins[nnn],i+1,j+1)):
                        pass
                    else:
                        if verbose:
                            print'\n---> Method: {0}  - bins: {1}'.format('AC_R_R_',Nbins[nnn])
                        J = Jack(w_estimator,conf_rp, pairs,ra_ref,dec_ref,ra_ref_rndm,dec_ref_rndm,ra_ref,dec_ref,ra_ref_rndm,
                        dec_ref_rndm,jck_ref,jck_ref,jck_ref_rndm, jck_ref_rndm,w_ref,w_ref,w_ref_rndm,
                        w_ref_rndm,fact_dist,z_ref_value,'auto_rp',corr = 'NN', zu=z_ref,zr=z_ref,zur=z_ref_rndm,zrr=z_ref_rndm,centers=centers, njk=njk,verbose=verbose)


                        pairs = J.NNCorrelation()
                        path=('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_R_',Nbins[nnn],i+1,j+1)
                        save_obj(path,pairs)

            #counter+=1

            #if not verbose:
            #    update_progress(np.float((i*len(reference_bins_interval['z'])+counter+1))/(len(unknown_bins_interval['z'])*len(reference_bins_interval['z'])),timeit.default_timer(),start_time)
            #return counter

#****************************************************************************************************
#                                  plotting routine
#****************************************************************************************************
def plot(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,w_p,w_estimator):
        for nnn in range(len(Nbins)):
            for modes in corr_tobecomputed:

                n_rows=int(math.ceil(len(reference_bins_interval['z'])/4.))

                n_cols=4
                if n_rows==1:
                    n_cols=2
                    n_rows=2
                fig, ax = plt.subplots(n_rows,n_cols,sharex=True, sharey=True, figsize=(11,10))
                fig.subplots_adjust(wspace=0.,hspace=0.)
                    #fig = plt.figure()
                k=0
                x=0

                for j in range(0,reference_bins_interval['z'].shape[0]):

                    if modes=='AC_U_':
                        dict=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format(modes,Nbins[nnn],i+1,1))
                    else:
                        dict=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format(modes,Nbins[nnn],i+1,j+1))
                    theta=dict['theta']
                    w=estimator(w_estimator,dict['DD'][:,0],dict['DR'][:,0],dict['RD'][:,0],dict['RR'][:,0])
                    wjk=estimator(w_estimator,dict['DD'][:,1:],dict['DR'][:,1:],dict['RD'][:,1:],dict['RR'][:,1:])
                    #
                    #compute covariance
                    average=np.zeros(w.shape[0])
                    cov_jck=np.zeros((w.shape[0],w.shape[0]))
                    for jj in range(njk):
                        average+=wjk[:,jj]
                    average=average/(njk)

                    for ii in range(len(average)):
                        for jj in range(ii+1):
                            for kk in range(njk):
                                cov_jck[ii,jj]+=wjk[ii,kk]*wjk[jj,kk]

                            cov_jck[ii,jj]=(-average[ii]*average[jj]*njk+cov_jck[ii,jj])*(njk-1)/(njk)
                            cov_jck[jj,ii]=cov_jck[ii,jj]

                    err=np.zeros(len(cov_jck[:,0]))
                    for ss in range(len(average)):
                        err[ss]=np.sqrt(cov_jck[ss,ss])


                    #vars()['ax'+str(j)]= fig.add_subplot(6,4,j+1)

                    #ax[x,k].set_ylim([-0.4,1])


                    ax[x,k].set_xlim([0.001,10])

                    '''
                    CROSSCHECK FACT_DIST
                    #total=dict['w_total']
                    #ax[x,k].plot(theta,total,color='black')
                    '''
                    ax[x,k].errorbar(theta,w,err,fmt='o',color='black',markersize='3',elinewidth='0.5')
                    ax[x,k].xaxis.set_tick_params(labelsize=8)
                    ax[x,k].yaxis.set_tick_params(labelsize=8)

                    if w_p != 'w':
                        ax[x,k].set_yscale("log")

                    ax[x,k].set_xscale("log")

                    ax[x,k].text(0.8, 0.8, 'z_bin {0}'.format(j+1), verticalalignment='bottom', horizontalalignment='left',  transform=ax[x,k].transAxes,fontsize=15)


                    #cancel 1st and last number
                    '''
                    if k==0 and x==0:
                        yticks = ax[x ,k ].xaxis.get_major_ticks()
                        yticks[0].label1.set_visible(False)

                    xticks = ax[x ,k ].yaxis.get_major_ticks()
                    xticks[0].label1.set_visible(False)
                    xticks[-1].label1.set_visible(False)
                    xticks[-2].label1.set_visible(False)
                        #xticks[-1].label1.set_visible(False)

                        #yticks = ax[x ,k ].yaxis.get_major_ticks()
                        #yticks[0].label1.set_visible(False)
                        #yticks[-1].label1.set_visible(False)
                    '''
                    k+=1
                    if k==n_cols:
                        k=0
                        x+=1

                if w_p =='w':
                    plt.savefig(('./pairscount/{0}_tomobin_{1}_angularbins_{2}.pdf').format(modes,i+1,Nbins[nnn]), format='pdf', dpi=1000)
                else:
                    plt.savefig(('./pairscount/pairs_plot/{0}_{1}_tomobin_{2}_angularbins_{3}.pdf').format(modes,w_p,i+1,Nbins[nnn]), format='pdf', dpi=1000)
                plt.close()


def plot_bias(Nbins,corr_tobecomputed,reference_bins_interval,i,z_unk,njk,w_p,w_estimator):
        for nnn in range(len(Nbins)):
            if 'AC_R_P_' in corr_tobecomputed:

                bias=np.zeros((len(reference_bins_interval['z']),njk+1))
                z_bias=reference_bins_interval['z']

                for j in range(0,reference_bins_interval['z'].shape[0]):

                    dict=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_P_',Nbins[nnn],i+1,j+1))
                    theta=dict['theta']
                    #w=dict['w'][:,0]
                    w=estimator(w_estimator,dict['DD'][:,0],dict['DR'][:,0],dict['RD'][:,0],dict['RR'][:,0])

                    #wjk=dict['w'][:,1:]
                    wjk=estimator(w_estimator,dict['DD'][:,1:],dict['DR'][:,1:],dict['RD'][:,1:],dict['RR'][:,1:])
                    average=np.zeros(w.shape[0])
                    cov_jck=np.zeros((w.shape[0],w.shape[0]))
                    for jj in range(njk):
                        average+=wjk[:,jj]
                    average=average/(njk)

                    for ii in range(len(average)):
                        for jj in range(ii+1):
                            for kk in range(njk):
                                cov_jck[ii,jj]+=wjk[ii,kk]*wjk[jj,kk]

                            cov_jck[ii,jj]=(-average[ii]*average[jj]*njk+cov_jck[ii,jj])*(njk-1)/(njk)
                            cov_jck[jj,ii]=cov_jck[ii,jj]

                    err=np.zeros(len(cov_jck[:,0]))
                    for ss in range(len(average)):
                        err[ss]=np.sqrt(cov_jck[ss,ss])

                    bias[j,0]=weight_w(w,theta,err,False,1)['integr']
                    for ik in range(njk):
                        bias[j,ik+1]=weight_w(wjk[:,ik],theta,err,False,1)['integr']



                    average=np.zeros(bias.shape[0])
                    cov_jck=np.zeros((bias.shape[0],bias.shape[0]))
                    for jj in range(njk):
                        average+=bias[:,jj+1]
                    average=average/(njk)

                    for ii in range(len(average)):
                        for jj in range(ii+1):
                            for kk in range(njk):
                                cov_jck[ii,jj]+=bias[ii,kk+1]*bias[jj,kk+1]

                            cov_jck[ii,jj]=(-average[ii]*average[jj]*njk+cov_jck[ii,jj])*(njk-1)/(njk)
                            cov_jck[jj,ii]=cov_jck[ii,jj]

                    err_bias=np.zeros(len(cov_jck[:,0]))
                    for ss in range(len(average)):
                        err_bias[ss]=np.sqrt(cov_jck[ss,ss])



                plt.errorbar(z_bias,bias[:,0],err_bias,fmt='o',color='black',markersize='3',elinewidth='0.5')
                plt.savefig(('./pairscount/bias_{0}_tomobin_{1}_angularbins_{2}.pdf').format('AC_R_P_',i+1,Nbins[nnn]), format='pdf', dpi=1000)

                plt.close()

            if 'AC_R_D_' in corr_tobecomputed:

                bias=np.zeros(len(reference_bins_interval['z']))
                err_bias=np.zeros(len(reference_bins_interval['z']))
                z_bias=reference_bins_interval['z']

                for j in range(0,reference_bins_interval['z'].shape[0]):

                    pairs=load_obj(('./pairscount/pairs/{0}_{1}_{2}_{3}').format('AC_R_D_',Nbins[nnn],i+1,j+1))

                    DD_summed=np.zeros(pairs['DD'].shape[1])
                    DR_summed=np.zeros(pairs['DR'].shape[1])
                    RD_summed=np.zeros(pairs['RD'].shape[1])
                    RR_summed=np.zeros(pairs['RR'].shape[1])
                    w_summed=np.zeros(pairs['RR'].shape[1])
                    for jck in range(w_summed.shape[0]):
                            DD_summed[jck]=np.sum(pairs['DD'][:,jck])
                            DR_summed[jck]=np.sum(pairs['DR'][:,jck])
                            RD_summed[jck]=np.sum(pairs['RD'][:,jck])
                            RR_summed[jck]=np.sum(pairs['RR'][:,jck])


                            w_summed[jck]=estimator(w_estimator,DD_summed[jck],DR_summed[jck],RD_summed[jck],RR_summed[jck])

                    average=0.
                    cov_jck=0.
                    err_jck=0.
                    for kk in range(njk):
                        average+=w_summed[kk+1]
                    average=average/(njk)

                    for kk in range(njk):
                        cov_jck+=(-average+w_summed[kk+1])*(-average+w_summed[kk+1])
                    err_jck=np.sqrt(cov_jck*(njk-1)/(njk))

                    bias[j]=w_summed[0]
                    err_bias[j]=err_jck
                plt.errorbar(z_bias,bias,err_bias,fmt='o',color='black',markersize='3',elinewidth='0.5')
                plt.savefig(('./pairscount/bias_{0}_tomobin_{1}_angularbins_{2}.pdf').format('AC_R_D_',i+1,Nbins[nnn]), format='pdf', dpi=1000)

                plt.close()


def impose_cosmology(cosmology):
    #default:
    cosmo=Planck15

    if 'Planck' in cosmology:
        if '15' in cosmology:
            cosmo=Planck15
        elif '13' in cosmology:
            cosmo=Planck13


    if 'WMAP' in cosmology:
        if '5' in cosmology:
            cosmo=WMAP5
        elif '7' in cosmology:
            cosmo=WMAP7
        elif '9' in cosmology:
            cosmo=WMAP9

    if 'FlatLambdaCDM' in cosmology:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    save_obj('./pairscount/cosmology',cosmo)
    '''

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    WMAP5 	Komatsu et al. 2009 	70.2 	0.277 	Yes
    WMAP7 	Komatsu et al. 2011 	70.4 	0.272 	Yes
    WMAP9 	Hinshaw et al. 2013 	69.3 	0.287 	Yes
    Planck13 	Planck Collab 2013, Paper XVI 	67.8 	0.307 	Yes
    Planck15 	Planck Collab 2015, Paper XIII 	67.7 	0.307 	Yes
    '''
    # saving the cosmology
    return cosmo
