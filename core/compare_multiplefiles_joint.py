from __future__ import print_function, division

import numpy as np
import pandas as pd
import pickle
import yaml
import matplotlib.pyplot as plt
from os import path, makedirs
import copy
from scipy.interpolate import UnivariateSpline
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import os
import timeit
import sys
from scipy import linalg
from .dataset import save_obj, load_obj, update_progress
from .functions_nz import covariance_jck
from routine_compare import update_progress,compute_mean1,compute_statistics,covariance_scalar_jck,Silence



def compare(photo_z_columns,true_column,resampling,label_output,path_wz_samples,path_datasets,priors,tomo_bins,sigma='ALL',add_noise=False,shift_pz_1= None, model_kwargs = {'z0': 0.5},
                cov_mode='diag',zmin='None',zmax='None',nwalkers=20.,nburnin=200,nrun=700,live_dangerously= False,match ='chi2'):

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    compare_path= './compare/'
    label_output=label_output+'_'+photo_z_columns[0]+'_'+match+'_'+cov_mode+'_'+str(sigma)+'_'
    time0=0.


    #load clustering-z results
    wz, wz_cov, wz_cov_mean, wz_mean = load_wz(zmin,zmax,label_output,path_wz_samples, sigma,resampling,cov_mode=cov_mode)

    # load true redshift distribution in finer broad bins
    pz = load_pz(wz,shift_pz_1,path_datasets,photo_z_columns,true_column,tomo_bins,add_noise)

    # we'll use the priors to create p0
    p0 = []

    # create the redshift bias object
    sampler = RedshiftBiasSampler(
        wz=wz,wz_cov=wz_cov, wz_cov_mean=wz_cov_mean,  wz_mean= wz_mean,
        pz=pz, cov_mode=cov_mode,
        nwalkers=nwalkers, nburnin=nburnin, nrun=nrun,
        live_dangerously=live_dangerously,
        p0=p0, mode=match, model_kwargs=model_kwargs, priors=priors, time0=time0)

    # run sampler
    sampler.walk()

    # save sampler
    save_sampler(sampler, label_output, compare_path)

    # correct samples *****************************************
    p = sampler.median_sample()
    ndim=len(path_wz_samples)
    nphoto=len(photo_z_columns)
    spreads=np.zeros(ndim*nphoto)
    biases=np.zeros(ndim*nphoto)
    amplitudes=np.zeros(ndim*nphoto)



    for i in range(int(ndim)):
        for j in range(int(nphoto)):
            biases[nphoto*i+j] = p[3*nphoto*i+3*j]
            amplitudes[nphoto*i+j] = np.exp(-p[3*nphoto*i+3*j+1])
            spreads[nphoto*i+j] = p[3*nphoto*i+3*j+2]

    for i in pz.keys():
        for j in range(int(nphoto)):
            index=j+nphoto*int(i)
            mean_pz=np.mean(pz[i][str(j)]['z'])
            #print (mean_pz,biases[index],spreads[index])
            pz[i][str(j)].update({'z_corrected':(mean_pz+spreads[index]*(pz[i][str(j)]['z']-mean_pz) - biases[index])})
            pz[i][str(j)].update({'amplitude':amplitudes[nphoto*int(i)+j]})

    wz_corrected=copy.deepcopy(wz)
    wz_corrected=sampler.correct_wz(wz_corrected, p,False)

    wz_cov_corrected=make_covariance(wz_corrected,cov_mode,False,resampling)
    wz_cov=make_covariance(wz,cov_mode,False,resampling)

    
    
    try:
        fig = sampler.plot_flatchain()
        fig.savefig('{0}countour{1}.pdf'.format(compare_path,label_output))
        plt.close()
    except:
        pass
    # save chains *************************
    flatchain = sampler.flatchain.copy()
    for i in range(ndim):
        percent=34
        new_chain_final=sampler.mean_z_chain[str(i)]
        vals = np.percentile(new_chain_final,[50 - percent, 50, 50 + percent])
        print (np.mean(pz[str(i)]['z_true_residual'])-vals[1],vals[1]-vals[0],vals[2]-vals[1])
        #print (new_chain_final)
        import corner
        try:
            fig=corner.corner(new_chain_final)
            fig.savefig('./compare/mean_{0}.png'.format(i+1))
        except:
            pass
    '''
        if mad_cut > 0:
            # remove median absolute deviations
            med = np.median(flatchain, axis=0)
            mad = np.median(np.abs(flatchain - med), axis=0)
            flatchain = flatchain[np.all(np.abs(flatchain - med) < mad_cut * mad, axis=1)]

        return corner.corner(flatchain[:,self.mask_labels], labels=self.labels, **kwargs)
    # compute mean and errors based on the chains (only if nphoto>1)
    '''
    # reading corrections *****************
    dict=load_obj(compare_path+label_output+'_results')
    keys=dict.keys()
    shift=np.zeros(ndim*nphoto)
    errp=np.zeros(ndim*nphoto)
    errm=np.zeros(ndim*nphoto)
    errc=np.zeros(ndim*nphoto)
    if nphoto==1:
        for i in range(ndim):
            for j in range(int(nphoto)):
                shift[i]=np.float(dict['$\\Delta z_{'+str(i+1)+'_'+str(j+1)+'}$'])
                errp[i]=np.float(dict['$\\Delta z_{'+str(i+1)+'_'+str(j+1)+'}$_err+'])
                errm[i]=np.float(dict['$\\Delta z_{'+str(i+1)+'_'+str(j+1)+'}$_err-'])
                errc[i]=np.float(dict['$\\Delta z_{'+str(i+1)+'_'+str(j+1)+'}$_err='])

    # plot *********************
    try:
        plot(pz,wz,wz_corrected,wz_cov,wz_cov_corrected ,label_output,compare_path,cov_mode,shift,errp,errm,errc,nphoto,amplitudes)
    except:
        pass


# ********************************************************************************
#                                   LOAD WZ AND PZ
# ********************************************************************************

def load_wz(zmin,zmax,label_output,label, sigma,resampling,cov_mode='diag'):
    '''
    [str(i)]
        -z_centers
        -z_edges
        -wz
        -wz_jack
    '''

    gaussian_process=False
    wz=dict()

    for i in range(len(label)):
        tomobin=dict()


        hpath = '{0}.h5'.format(label[i])

        # reading z ****************************
    
        tomobin.update({'z_centers':pd.read_hdf(hpath, 'z').values.T[0][:]})
        tomobin.update({'z_edges':pd.read_hdf(hpath, 'z_edges').values.T[0][:]})

        ww=pd.read_hdf(hpath, 'results').values.T[0][:]
        norm_wz = normalize(ww, tomobin['z_edges'])
        norm_wz=1.
        tomobin.update({'norm':norm_wz})

        # reading wz  **************************
        tomobin.update({'wz':pd.read_hdf(hpath, 'results').values.T[0][:]/norm_wz})

        # reading jackknives *******************
        try:
            wz1=pd.read_hdf(hpath, 'jackknife').values.T[:, :]
            for jk in range(wz1.shape[0]):
                #norm_wz_samples = normalize(wz1[jk,:], tomobin['z_edges'])
                wz1[jk,:] = wz1[jk,:]/norm_wz

            tomobin.update({'wz_jack':wz1})

        except:
            tomobin.update({'err':pd.read_hdf(hpath, 'err').values/norm_wz})
            gaussian_process=True



        # restrict *********************************
        mean_wz,std_w=compute_mean1(tomobin['wz'],tomobin['z_centers'])
        if zmin=='None':
            zmin1=0.
        else:
            zmin1=zmin
        if zmax=='None':
            zmax1=5.
        else:
            zmax1=zmax
        if sigma=='all':

            mask= (tomobin['z_centers']>(mean_wz-1.)) & (tomobin['z_centers']<(mean_wz+1.)) & ((tomobin['z_centers']>(zmin1)) & (tomobin['z_centers']<(zmax1)))
        else:
            if sigma*std_w<0.001:
                std_w=0.001/sigma
            mask= (tomobin['z_centers']>(mean_wz-sigma*std_w)) & (tomobin['z_centers']<(mean_wz+sigma*std_w)) &  ((tomobin['z_centers']>(zmin1)) & (tomobin['z_centers']<(zmax1)))


        delta=(-(tomobin['z_centers'][0])+(tomobin['z_centers'][1]))/1.9


        tomobin['z_centers']=tomobin['z_centers'][mask]
        tomobin['wz']=tomobin['wz'][mask]
        tomobin['wz_jack']=tomobin['wz_jack'][:,mask]

        #print (tomobin['z_centers'])
        mask2=(tomobin['z_edges']>tomobin['z_centers'][0]-delta) & (tomobin['z_edges']<(tomobin['z_centers'][-1]+delta))
        tomobin['z_edges']=tomobin['z_edges'][mask2]


        wz.update({str(i):tomobin})

    wz_cov=make_covariance(wz,cov_mode,gaussian_process,resampling,label_output)
    wz_cov_mean,means=make_covariance_mean(wz,wz_cov,label_output)
    return wz, wz_cov,wz_cov_mean,means


def rebin2(z_old, pdf_old, zbins):
        # spline
        kwargs_spline = {'s': 0,  # force spline to go through data points
                         'ext': 'zeros',  # ext=0 means extrapolate, =1 means return 0
                         'k': 3,
                        }
        spline = UnivariateSpline(z_old, pdf_old, **kwargs_spline)
        pdf = np.zeros(len(zbins) - 1)
        for i in xrange(len(zbins) - 1):
            zmin = zbins[i]
            zmax = zbins[i + 1]
            pdf[i] = spline.integral(zmin, zmax) #/ (zmax - zmin)
        return pdf

def load_pz(wz,shift_pz_1,path_datasets,photo_z_columns,true_column,tomo_bins_label,add_noise):
    '''
    [str(i)]
        z_true_residual
        [photoz_j]
            -z
            -pz
            -pz_z_centers
            -pz_z_edges
    '''

    pz=dict()

    for i in range(len(path_datasets)):
        pz_tomo=dict()
        for j,Z_T in enumerate(photo_z_columns):
            pz_tomo_j=dict()
            #print (pd.read_hdf(path_datasets[i]+'dataset.h5', 'unk').keys())
            z = pd.read_hdf(path_datasets[i]+'dataset.h5', 'unk')[Z_T].values
            z=z+shift_pz_1[i]
            mask_z=z>=0.
            z=z[mask_z]
            tomo_bins = pd.read_hdf(path_datasets[i]+'dataset.h5', 'unk')['bins'].values

            if add_noise:
                rndm=(np.random.randint(10000,size=len(z))-5000)/1000000.
                z=z+rndm
            # cuts the tomographic bin *************

            unknown_bins_interval=load_obj(path_datasets[i]+'unknown_bins_interval')
            unkn_z=unknown_bins_interval['z']
            tomo_bins=tomo_bins[mask_z]
            mask=(int(tomo_bins_label[i])-0.5<tomo_bins) & (tomo_bins<int(tomo_bins_label[i])+0.5)
            z=z[mask]

            pz_tomo_j.update({'z':z})
            #pz_tomo.update({'tomo_bins':tomo_bins})

            # coarser spacing   ********************
            norm_tomo1=z.shape[0]
            pz_zbins = np.linspace(-0.5, 2.5, 15001)

            pz_zcenters = 0.5 * (pz_zbins[1:] + pz_zbins[:-1])

            pz1, centers_out, bins_out, norm_out = catalog_to_histogram(z, pz_zbins, cut_to_range=False)
            pz1=pz1*len(z)


            pz_tomo_j.update({'pz':pz1})
            pz_tomo_j.update({'pz_z_centers':pz_zcenters})
            pz_tomo_j.update({'pz_z_edges':pz_zbins})
            pz_tomo.update({str(j):pz_tomo_j})
        # read true distribution ************************
        z_comparison = pd.read_hdf(path_datasets[i]+'dataset.h5', 'unk')[true_column].values
        mask_z=z_comparison>=0.
        z_comparison=z_comparison[mask_z]
        tomo_bins = pd.read_hdf(path_datasets[i]+'dataset.h5', 'unk')['bins'].values

        unknown_bins_interval=load_obj(path_datasets[i]+'unknown_bins_interval')
        unkn_z=unknown_bins_interval['z']
        tomo_bins=tomo_bins[mask_z]
        mask=(int(tomo_bins_label[i])-0.5<tomo_bins) & (tomo_bins<int(tomo_bins_label[i])+0.5)
        z_comparison=z_comparison[mask]


        pz_tomo.update({'z_true_residual':z_comparison})

        pz.update({str(i):pz_tomo})

    return pz

def catalog_to_histogram(z, bins, weights=None, z_min=0, z_max=10, cut_to_range=True):
    conds = (z >= z_min) & (z < z_max)
    if sum(conds) == 0:
        bins_out = bins
        pdf = np.zeros(len(bins) - 1)
    else:
        # bin
        pdf, bins_out = np.histogram(z, bins, weights=weights)
    if np.any(bins != bins_out):
        print('bins:')
        print(bins)
        print('bins_out:')
        print(bins_out)
        raise Exception('bins not equal to bins_out?!')

    centers = 0.5 * (bins[1:] + bins[:-1])
    conds = (bins[:-1] >= z_min) & (bins[1:] < z_max)
    conds_bins = (bins >= z_min) & (bins < z_max)

    # normalize with trapezoidal
    norm = normalize(pdf, bins, z_min=z_min, z_max=z_max)
    if norm == 0:
        norm = 1

    if cut_to_range:
        centers = centers[conds]
        pdf = pdf[conds]
        bins = bins[conds_bins]
    return pdf / norm, centers, bins, norm

def normalize(pdf, bins, z_min=0, z_max=10):
    # integrate via trapezoidal rule

    centers = 0.5 * (bins[1:] + bins[:-1])
    conds = (bins[:-1] >= z_min) & (bins[1:] < z_max)
    norm = np.trapz(pdf[conds], centers[conds])
    return norm


def make_covariance_mean(wz,wz_cov,path):
    mean_cov_err=np.zeros((len(wz.keys()),len(wz.keys())))
    means=np.zeros(len(wz.keys()))
    len1=0

    for i in wz.keys():
        len2=0
        for j in wz.keys():

            norm_mean_bin1=0.
            norm_mean_bin2=0.
            mean_bin1=0.
            mean_bin2=0.

            for k in range(wz[str(j)]['wz'].shape[0]):
                mean_bin2 += wz[str(j)]['wz'][k]*wz[str(j)]['z_centers'][k]
                norm_mean_bin2 += wz[str(j)]['wz'][k]
            mean_bin2=mean_bin2/norm_mean_bin2
            means[int(j)]=mean_bin2
            for k in range(wz[str(i)]['wz'].shape[0]):
                mean_bin1 += wz[str(i)]['wz'][k]*wz[str(i)]['z_centers'][k]
                norm_mean_bin1 += wz[str(i)]['wz'][k]
            mean_bin1=mean_bin1/norm_mean_bin1

            #print (mean_bin1,mean_bin2)
            mean_cov=0.

            for k in range(wz[str(i)]['wz'].shape[0]):
                for w in range(wz[str(j)]['wz'].shape[0]):

                    mean_cov+=((wz_cov[len1+k,len2+w])*(norm_mean_bin1*wz[str(i)]['z_centers'][k]-mean_bin1*norm_mean_bin1)*(norm_mean_bin2*wz[str(j)]['z_centers'][w]-mean_bin2*norm_mean_bin2))

            mean_cov=mean_cov/(norm_mean_bin1*norm_mean_bin2)**2.
            mean_cov_err[int(i),int(j)]=mean_cov

            len2+=wz[str(j)]['wz'].shape[0]
        len1+=wz[str(i)]['wz'].shape[0]
    if path:
        wz_cov_1=copy.deepcopy(mean_cov_err)
        for i in range(mean_cov_err.shape[0]):
            for j in range(mean_cov_err.shape[0]):
                wz_cov_1[i,j]=mean_cov_err[i,j]/np.sqrt(mean_cov_err[i,i]*mean_cov_err[j,j])
        
        try:
            plt.pcolor(wz_cov_1)
            plt.colorbar()
            plt.savefig('./compare/cov_mean_'+path+'.pdf' , format='pdf', dpi=1000)
            plt.close()
        except:
            pass
    return mean_cov_err,means


def make_covariance(wz,cov_mode,gaussian_process,resampling,path=False):


    len=0
    for i in wz.keys():
        len+=wz[str(i)]['wz'].shape[0]
        jck=wz[str(i)]['wz_jack'].shape[0]
    wz_cov=np.zeros((len,len))
    wz_tot=np.zeros((jck,len))
    len=0
    for i in wz.keys():

        wz_tot[:,len:(len+wz[str(i)]['wz_jack'].shape[1])]=wz[str(i)]['wz_jack']
        len+=wz[str(i)]['wz'].shape[0]

    dict_cov=covariance_jck(wz_tot[:,:].T,jck,resampling)


    if not gaussian_process:

        wz_cov = dict_cov['cov']

    else:
        for i in range(wz_samples.shape[1]):
            cov_full=np.zeros((wz_samples_err.shape[0],wz_samples_err.shape[0]))
            for ii in range(wz_samples_err.shape[0]):
                cov_full[ii,ii]=wz_samples_err[ii]**2.
            wz_cov[i*wz_samples.shape[2]:(i+1)*wz_samples.shape[2],i*wz_samples.shape[2]:(i+1)*wz_samples.shape[2]]=cov_full


    if cov_mode=='diag':
        for g in range(wz_cov.shape[0]):
            for h in range(wz_cov.shape[0]):
                if h!=g:
                    wz_cov[h,g]=0.

    if cov_mode=='2off_diag':
        for g in range(wz_cov.shape[0]):
            for h in range(wz_cov.shape[0]):
                if (h-g)*(h-g)>4.:
                    wz_cov[h,g]=0.

    #plot_covariance!#
    if path:
        wz_cov_1=copy.deepcopy(wz_cov)
        for i in range(wz_cov.shape[0]):
            for j in range(wz_cov.shape[0]):
                wz_cov_1[i,j]=wz_cov[i,j]/np.sqrt(wz_cov[i,i]*wz_cov[j,j])
        try:
            plt.pcolor(wz_cov_1)
            plt.colorbar()
            plt.savefig('./compare/cov_'+path+'.pdf' , format='pdf', dpi=1000)
            plt.close()
        except:
            print ("not saved")
        inv_cov=linalg.inv(wz_cov)
        wz_cov_1=copy.deepcopy(linalg.inv(wz_cov))
        for i in range(wz_cov.shape[0]):
            for j in range(wz_cov.shape[0]):
                wz_cov_1[i,j]=inv_cov[i,j]/np.sqrt(inv_cov[i,i]*inv_cov[j,j])
        try:
            plt.pcolor(wz_cov_1)
            plt.colorbar()
            plt.savefig('./compare/cov_'+path+'_inv.pdf' , format='pdf', dpi=1000)
            plt.close()
        except:
            print ("not saved")
    return (wz_cov)


def save_sampler(sampler, label, sample_path):
    h5_path = '{0}/{1}.h5'.format(sample_path,label)
    priors_pickle_path = '{0}/priors_{1}.pkl'.format(sample_path, label)
    # save the chain
    chain = sampler.sampler.chain.copy()
    pd.Panel(chain).to_hdf(h5_path, '/{0}/chain'.format(label))

    kwargs = {'nwalkers': sampler.nwalkers,
              'nburnin': sampler.nburnin,
              'nrun': sampler.nrun,
              'live_dangerously': sampler.live_dangerously,
              'time0': sampler.time0}
    pd.Series(kwargs).to_hdf(h5_path, '/{0}/kwargs'.format(label))

    pd.Series(sampler.model_kwargs).to_hdf(h5_path, '/{0}/model_kwargs'.format(label))

    save_pickle(sampler.priors, priors_pickle_path)

    # write human readable output, too
    text_path = '{0}/{1}_results.txt'.format(sample_path, label)
    results = {key: {} for key in sampler.labels}
    # median, error, asymmetric error, argmax, two samples
    N_samples = 5
    results_str = '{0:+.3e}'
    sample_median = sampler.median_sample()[sampler.mask_labels]
    sample_error = sampler.error_sample()[sampler.mask_labels]
    sample_asymmetric_error = sampler.error_asymmetric_sample()[sampler.mask_labels]
    sample_argmax = sampler.best_sample()[sampler.mask_labels]
    sample_samples = sampler.sample(N_samples)[:,sampler.mask_labels]
    lnprior = np.array(sampler.lnprior(sampler.median_sample()))[sampler.mask_labels]
    for key, med, err, asym, arg, sam, lnpri in zip(sampler.labels, sample_median, sample_error, sample_asymmetric_error, sample_argmax, sample_samples.T, lnprior):
        # put in as strings
        results[key]['median'] = results_str.format(med)
        results[key]['error='] = results_str.format(err)
        results[key]['error+'] = results_str.format(asym[1])
        results[key]['error-'] = results_str.format(asym[0])
        results[key]['argmax'] = results_str.format(arg)
        results[key]['sample'] = [results_str.format(sam_i) for sam_i in sam]
        results[key]['lnprior'] = results_str.format(lnpri)

    # get the tomo chi2 and dof as well
    '''
    chi2 = -2 * sampler.lnlike(sample_median)  # for each tomo
    lnprob = sampler.lnprob(sample_median)
    lnprior_tot = np.sum(lnprior)
    results['chi2'] = {}
    results['chi2']['lnprior'] = results_str.format(lnprior_tot)
    results['chi2']['lnprob'] = results_str.format(lnprob)


    for chi2_i, chi2_val in enumerate(chi2):
        results['chi2']['tomo_{0}'.format(chi2_i)] = results_str.format(chi2_val)
        results['chi2']['dof_{0}'.format(chi2_i)] = '{0}'.format(len(sampler.wz[chi2_i]))
    '''
    with open(text_path, 'w') as f:
        f.write(yaml.dump(results, default_flow_style=False))

    results_output=dict()
    for i in range(len(sampler.labels)):
        results_output.update({sampler.labels[i]:results[sampler.labels[i]]['median']})
        results_output.update({'{0}_err+'.format(sampler.labels[i]):results[sampler.labels[i]]['error+']})
        results_output.update({'{0}_err-'.format(sampler.labels[i]):results[sampler.labels[i]]['error-']})
        results_output.update({'{0}_err='.format(sampler.labels[i]):results[sampler.labels[i]]['error=']})
    save_obj('{0}/{1}_results'.format(sample_path, label),results_output)


def load_sampler(label, sample_path):
    h5_path = '{0}/compare.h5'.format(sample_path)
    priors_pickle_path = '{0}/compare_priors_{1}.pkl'.format(sample_path, label)

    kwargs = pd.read_hdf(h5_path, '/{0}/kwargs'.format(label))

    model_kwargs = pd.read_hdf(h5_path, '/{0}/model_kwargs'.format(label)).to_dict()

    priors = load_pickle(priors_pickle_path)


    sampler = RedshiftBiasSampler(model_kwargs=model_kwargs, priors=priors,
                                  **kwargs)



    # add chain to sampler
    chain_key = '/{0}/chain'.format(label)
    chain = pd.read_hdf(h5_path, chain_key).values

    sampler.sampler._chain = chain

    return sampler

def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)





class RedshiftBiasSampler(object):
    """Model a set of pz and wz as redshift bias
    """

    def __init__(self, wz=None,wz_cov=None,wz_cov_mean=None, wz_mean= None,
                 pz=None,
                 cov_mode='full',
                 nwalkers=64, nburnin=100, nrun=500, live_dangerously=False,
                 p0=[],
                 mode='chi2',
                 model_kwargs={'z0': 0.8},
                 priors={
                     'gamma': {'kind': 'uniform', 'weight': 1,
                               'kwargs': {'loc': -2, 'scale': 4}},
                     'deltaz': {'kind': 'truncnorm', 'weight': 1,
                                'kwargs': {'a': -8.0, 'b': 8.0,
                                           'loc': 0, 'scale': 0.05}},
                     'amplitude': {'kind': 'truncnorm', 'weight': 1,
                                   'kwargs': {'a': -5.0, 'b': 5.0,
                                              'loc': 0, 'scale': 1.0}},
                     },
                 time0=0,
                 **kwargs):

        """Combine PZ and WZ estimates

        Parameters
        ----------
        wz :                Estimation of redshift distribution from clustering
                            methods. A list of arrays for the different
                            samples.
        wz_cov :            Estimation of the covariance of the wz
                            measurements.  These can be provided in three
                            formats:
                                -   If cov is a single 2d array, then it is
                                    assumed to be the full covariance of all
                                    the wz values flattened to one list
                                -   If the cov is a list of 2d arrays, then it
                                    is assumed to be the covariance matrix of
                                    each wz sample, so each wz sample is
                                    treated as independent of each other
                                -   If the cov is a list of 1d arrays, then it
                                    is assumed to be the diag component of
                                    the covariance, sigma2
        wz_zbins :          The redshift bins used for the wz estimates. If it
                            is a single array, it is assumed that all wz
                            measurements share the same redshift bins. If it is
                            a list of arrays then each entry corresponds to the
                            redshift bins for the corresponding wz
        wz_zcenters :       The center location of the redshift bins
        pz:                 List of array of redshifts from the photoz code for
                            each sample.
        pz_zbins :          The redshift bins used for the pz estimates.
                            Note: Comparisons are done in the wz basis.
        pz_zcenters :       The center location of the redshift bins
        nwalkers :          int [default 32] number of walkers emcee uses
        nburnin :           int [default 200] number of initial steps to take
                            in emcee fitter
        nrun :              int [default 2000] number of steps each walker will
                            take in sampling the parameter space
        live_dangerously :  override emcee's concerns about size of nwalkers
                            relative to ndims of the model
        p0:                 list [Default: empty] List of initial guesses. Must
                            be nwalkers length. If it is not, then the initial
                            guesses will come from samples of the prior
                            distribution.
        priors :            dict of dicts. Each key corresponds to a different
                            term in equation. kind is the kind of scipy.stats
                            function.  Weight is the relative weight of the
                            prior (higher = weight heavier). kwargs is a
                            dictionary of kwargs passed to construction of the
                            stats function.
        z0:                 Float [Default: 0.5] Redshift pivot for bias
                            evolution function.

        """
        self.time0 = time0

        self.wz = wz
        self.pz = pz
        self.wz_cov = wz_cov
        self.wz_cov_mean = wz_cov_mean
        self.wz_mean= wz_mean

        self.start=timeit.default_timer()
        self.mode=mode
        self.ndim=len(self.wz.keys())
        self.nphoto=len(self.pz[pz.keys()[0]].keys())-1
        mean_z_chain=dict()
        for i in range(self.ndim):
            chain=[]
            mean_z_chain.update({str(i):chain})
        self.mean_z_chain=mean_z_chain
        self.cov_mode=cov_mode


        # set up model kwargs
        self.model_kwargs = self.setup_model_kwargs(**model_kwargs)


        # set up labels
        #self.labels = self.setup_labels()

        # set up priors
        self.priors = priors
        self.prior_function, self.prior_weight,self.labels,self.mask_labels = self.setup_prior(priors)
        self.nparams = len(self.prior_function)

        # set up emcee
        self.nwalkers = nwalkers
        self.nburnin = nburnin
        self.nrun = nrun
        self.live_dangerously = live_dangerously
        self.p0, self.sampler = self.setup_sampler(nwalkers, p0)

    def lnlike(self, p):
        """Function that evaluates the goodness of fit of parameters

        Parameters
        ----------
        p : array
            Parameter values

        Returns
        -------
        chi squares
        """
        # we use wz_zbins (instead of pz_zbins) to put pz_corrected in same shape as output wz

        self.correct_pz_full(self.pz, self.wz, self.wz_cov, -p)

        if self.mode=='chi2':
            chi2 = self.evaluate(self.wz, self.pz, self.wz_cov)
        elif self.mode =='mean':
            chi2 = self.evaluate_mean(self.wz_mean, self.wz,self.pz, self.wz_cov_mean)
        return -0.5 * chi2

    def lnprior(self, p):

        logprior = [prior.logpdf(pi) * wi for pi, prior, wi in zip(p, self.prior_function, self.prior_weight)]
        return logprior

    def lnprob(self, p):

        self.nsteps += 1

        update_progress(float(self.nsteps / self.nwalkers / (self.nburnin + self.nrun)),timeit.default_timer(),self.start)


        lp = np.sum(self.lnprior(p))
        if not np.isfinite(lp):
            return -np.inf

        ll = np.sum(self.lnlike(p))

        return lp + ll

    def evaluate_mean(self, wz_mean, wz, pz, wz_cov_mean):
        pz_mean=np.zeros(len(pz.keys()))
        for i in pz.keys():
            mean_bin=0.
            norm=0.
            for k in range(wz[i]['wz'].shape[0]):
                mean_bin +=pz[i]['pz_binned'][k]*wz[i]['z_centers'][k]
                norm+= pz[i]['pz_binned'][k]
            mean_bin=mean_bin/norm
            pz_mean[int(i)]=mean_bin

        cv_chol = linalg.cholesky(wz_cov_mean, lower=True)
        cv_sol = linalg.solve(cv_chol, wz_mean - pz_mean, lower=True)
        chi2_val = np.sum(cv_sol ** 2)
        return np.array(chi2_val)

    def evaluate(self, wz, pz, cov):



        if self.cov_mode=='diag':

            chi2_val=0.
            len=0
            for i in wz.keys():
                cv_chol = linalg.cholesky(cov[len:(len+wz[i]['wz'].shape[0]),len:(len+wz[i]['wz'].shape[0])], lower=True)
                cv_sol = linalg.solve(cv_chol, wz[i]['wz'] - pz[i]['pz_binned'], lower=True)

                #Hartlap_correction
                N_p=(wz[str(0)]['wz_jack'].shape[0])
                p_p=wz[i]['wz'].shape[0]
                f_hartlap=(N_p-1)/(N_p-p_p-2)
                chi2_val+= np.sum(cv_sol ** 2)/f_hartlap
                len+=wz[i]['wz'].shape[0]

        else:
            len=0
            for i in wz.keys():
                len+=wz[i]['wz'].shape[0]

            wz_tot=np.zeros(len)
            pz_tot=np.zeros(len)

            len=0
            for i in wz.keys():

                wz_tot[len:(len+wz[i]['wz'].shape[0])]=wz[i]['wz']
                pz_tot[len:(len+wz[i]['wz'].shape[0])]=pz[i]['pz_binned']

                len+=wz[i]['wz'].shape[0]

            cv_chol = linalg.cholesky(cov, lower=True)
            cv_sol = linalg.solve(cv_chol, wz_tot - pz_tot, lower=True)
            chi2_val = np.sum(cv_sol ** 2)

            #Hartlap_correction
            N_p=(wz[str(0)]['wz_jack'].shape[0])
            p_p=cov.shape[0]
            f_hartlap=(N_p-1)/(N_p-p_p-2)
            chi2_val=chi2_val/f_hartlap


        return np.array(chi2_val)


        #try to test with off diagonal set to zero and only with the cross terms.



    @classmethod
    def rebin(cls, z_old, pdf_old, zbins):
        # spline
        kwargs_spline = {'s': 0,  # force spline to go through data points
                         'ext': 'zeros',  # ext=0 means extrapolate, =1 means return 0
                         'k': 3,
                        }
        spline = UnivariateSpline(z_old, pdf_old, **kwargs_spline)
        pdf = np.zeros(len(zbins) - 1)
        for i in xrange(len(zbins) - 1):
            zmin = zbins[i]
            zmax = zbins[i + 1]
            pdf[i] = spline.integral(zmin, zmax) #/ (zmax - zmin)
        return pdf

    def setup_sampler(self, nwalkers=32,  p0=[]):
        import emcee

        if len(p0) != nwalkers:
            # set p0 from the prior if no p0 given
            p0 = self.prior_sample(nwalkers)

        # set up the sampler
        sampler = emcee.EnsembleSampler(nwalkers, self.nparams, self.lnprob,
            live_dangerously=self.live_dangerously)

        return p0, sampler

    def walk(self, nburnin=-1, nrun=-1):

        if nburnin < 0:
            nburnin = self.nburnin
        else:
            self.nburnin = nburnin
        if nrun < 0:
            nrun = self.nrun
        else:
            self.nrun = nrun

        self.nsteps = 0
        # burn-in
        stuff = self.sampler.run_mcmc(self.p0, nburnin)
        #print (list(stuff)[0])
        pos = (list(stuff)[0])
        # run
        self.sampler.reset()
        stuff = self.sampler.run_mcmc(pos,nrun)

        # update timer to remove progress bar
        if self.time0:
            self.time0 = 0

        return stuff

    def plot_flatchain(self, mad_cut=0, **kwargs):
        # a useful kwarg is labels
        # plot with dfm's corner.py, potentially cutting on median absolute deviation
        flatchain = self.sampler.flatchain.copy()
        import corner
        if mad_cut > 0:
            # remove median absolute deviations
            med = np.median(flatchain, axis=0)
            mad = np.median(np.abs(flatchain - med), axis=0)
            flatchain = flatchain[np.all(np.abs(flatchain - med) < mad_cut * mad, axis=1)]

        return corner.corner(flatchain[:,self.mask_labels], labels=self.labels, **kwargs)

    def sample(self, size=1, **kwargs):
        if hasattr(self, 'sampler'):
            # if we have run the chain, sample from there
            return self.sampler.flatchain[
                np.random.choice(len(self.sampler.flatchain), size=size, **kwargs)]
        else:
            # else, sample from the prior
            return self.prior_sample(size=size)

    def median_sample(self):
        return np.median(self.sampler.flatchain, axis=0)

    def error_sample(self, percent=34):
        # default is 1sigma bounds
        # average, upper, lower
        vals = np.array(map(lambda v: 0.5 * (v[2] - v[0]),
            zip(*np.percentile(self.sampler.flatchain,
                               [50 - percent, 50, 50 + percent], axis=0))))
        return vals

    def error_asymmetric_sample(self, percent=34):
        # default is 1sigma bounds
        # upper, lower
        vals = np.array(map(lambda v: (v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(self.sampler.flatchain,
                               [50 - percent, 50, 50 + percent], axis=0))))
        return vals

    def best_sample(self, size=1):
        if size != 1:
            indx = np.argsort(self.sampler.flatlnprobability)[::-1]
            return self.sampler.flatchain[indx[:size]]
        else:
            return self.sampler.flatchain[self.sampler.flatlnprobability.argmax()]

    def prior_sample(self, size=1):
        if size != 1:
            # transpose to put in (nwalkers, ndim)
            samples = np.array([prior.rvs(size) for prior in self.prior_function]).T
        else:
            samples = np.array([prior.rvs() for prior in self.prior_function])
        return samples

    @property
    def chain(self):
        return self.sampler.chain

    @property
    def flatchain(self):
        return self.sampler.flatchain

    @classmethod
    def interpret_prior(cls, kind, kwargs):
        from scipy import stats
        return getattr(stats, kind)(**kwargs)

    # ****************************************************************************************
    # ****************************************************************************************

    def correct_pz_full(self, pz, wz, wz_cov, p, **kwargs):

        """
        NOTE: zbins -> final binning scheme
        NOTE: zcenters -> initial redshifts of pz (shape should match pz)
        """

        # if zcats are already binned pdfs, apply bias and rebin
        params = self.params(p,self.ndim,self.nphoto)
        biases = params['bias']
        amplitudes = params['amplitude']
        spread= -params['spread'] #(it s the only one that does not need to be inverted)
        gamma = params['gamma']
        z0 = self.model_kwargs['z0']


        for i in pz.keys():

            pdf=np.zeros(len(wz[i]['z_centers']))
            mute_mean=0.
            norm_mean=0.
            for j in range(self.nphoto):
                mean_pz=np.mean(pz[i][str(j)]['z'])

                #pdf=self.rebin(pz[i]['pz_z_centers'] + biases[int(i)], pz[i]['pz'] , wz[i]['z_edges'])
                index=j+self.nphoto*int(i)
                mute=(1./self.nphoto)*(1./spread[index])*self.rebin(mean_pz+spread[index]*(pz[i][str(j)]['pz_z_centers']-mean_pz) + biases[index], pz[i][str(j)]['pz'],wz[i]['z_edges'])

                # now apply the wz correction to pz (but with opposite params!)
                pdf += ((1 + wz[i]['z_centers']) / (1 + z0)) ** gamma * amplitudes[index] * mute
                mute_mean+=amplitudes[index]*(mean_pz+ biases[index])
                norm_mean+=amplitudes[index]
            self.mean_z_chain[i].append(mute_mean/norm_mean)

            self.pz[i].update({'pz_binned':pdf})



    # clustering
    def correct_wz(self, wz, p,gaussian_process, **kwargs):
        """
        NOTE: zbins -> not used
        NOTE: zcenters -> should match with input wz shape
        """
        params = self.params(p,self.ndim,self.nphoto)
        amplitudes = params['amplitude']
        gamma = params['gamma']
        z0 = self.model_kwargs['z0']


        for i in wz.keys():
            wz[i]['wz'] = ((1 + wz[i]['z_centers']) / (1 + z0)) ** gamma  * wz[i]['wz']
            for jk in range(wz[i]['wz_jack'].shape[0]):
                wz[i]['wz_jack'][jk,:] = ((1 + wz[i]['z_centers']) / (1 + z0)) ** gamma  * wz[i]['wz_jack'][jk,:]

        return wz


    # ****************************************************************************************
    # ****************************************************************************************

    @classmethod
    def params(cls, p,ndim,nphoto):
        #boia
        bias=np.zeros(ndim*nphoto)
        amplitude=np.zeros(ndim*nphoto)
        spread=np.zeros(ndim*nphoto)


        for i in range(int(ndim)):
            for j in range(int(nphoto)):
                bias[nphoto*i+j] = p[3*nphoto*i+3*j]
                amplitude[nphoto*i+j] = np.exp(p[3*nphoto*i+3*j+1])
                spread[nphoto*i+j] = p[3*nphoto*i+3*j+2]
        gamma=p[3*ndim*nphoto]

        return {'bias': bias, 'spread': spread, 'amplitude': amplitude,'gamma':gamma}


    def setup_model_kwargs(self, z0):
        return {'z0': z0}

    def setup_prior(self, priors):
        # convert prior from dictionary to appropriate list
        #TO BE MODIFIED TO ACCOUNT FOR DIFFERENT PRIORS!
        prior_function = []
        prior_weight = []
        labels = []


        mask_labels=[]
        for i in range(self.ndim):
            for j in range(self.nphoto):
                for key in ['deltaz', 'amplitude', 'spread']:
                    try:

                        prior_function.append(self.interpret_prior(priors['photo_z'+str(j+1)]['tomo'+str(i+1)][key]['kind'], priors['photo_z'+str(j+1)]['tomo'+str(i+1)][key]['kwargs']))
                        prior_weight.append(priors['photo_z'+str(j+1)]['tomo'+str(i+1)][key]['weight'])
                        if key == 'deltaz':
                            labels+=['$\Delta z_{{{0}_{1}}}$'.format(i + 1,j + 1)]
                        elif key =='amplitude':
                            labels+=['$k_{{{0}_{1}}}$'.format(i + 1 ,j + 1)]
                        else:
                            labels+=['$A_{{{0}_{1}}}$'.format(i + 1 ,j + 1)]
                        mask_labels.append(True)
                    except:
                        mask_labels.append(False)
                        if key=='spread':

                            prior_function.append(self.interpret_prior('uniform', {'loc': 1., 'scale': 0.00000001}))
                        else:
                            prior_function.append(self.interpret_prior('uniform', {'loc': 0., 'scale': 0.00000001}))
                        prior_weight.append(1)

        try:

            prior_function.append(self.interpret_prior(priors['gamma']['kind'], priors['gamma']['kwargs']))
            prior_weight.append(priors['gamma']['weight'])
            labels +=  ['$\gamma$']
            mask_labels.append(True)
        except:
            mask_labels.append(False)
            prior_function.append(self.interpret_prior('uniform', {'loc': 0., 'scale': 0.00000001}))
            prior_weight.append(1)


        return prior_function, prior_weight,labels,mask_labels





# ****************************************************************************************
def plot(pz,wz,wz_corrected,wz_cov,wz_cov_corrected ,label_save,output,cov_mode,shift,errp,errm,errc,nphoto,amplitudes):

    len1=0
    results_output=dict()
    for i in wz.keys():

        err=np.zeros(wz[i]['wz'].shape[0])
        err_corrected=np.zeros(wz[i]['wz'].shape[0])
        for h in range(wz[i]['wz'].shape[0]):
            err[h]=np.sqrt(wz_cov[len1+h,len1+h])
            err_corrected[h]=np.sqrt(wz_cov_corrected[len1+h,len1+h])


        fig= plt.figure()
        ax = fig.add_subplot(111)
        # stack the pz ***************************

        p_z=copy.deepcopy(pz[i][str(0)]['z'])
        #print (pz[i][str(0)]['z'].shape)
        weight_corr1=(1./nphoto)*np.ones(len(p_z))
        for j in range(1,nphoto):
                #print (pz[i][str(j)]['z'].shape)
                #print (pz[i][str(j)]['z_corrected'].shape)
                p_z=np.hstack((p_z.T,pz[i][str(j)]['z'].T))

                weight_mute=(1./nphoto)*np.ones(len(pz[i][str(j)]['z']))
                weight_corr1=np.hstack((weight_corr1.T,weight_mute.T))

        p_z_corr=copy.deepcopy(pz[i][str(0)]['z_corrected'])
        index=nphoto*int(i)
        weight_corr2=(1./nphoto)*amplitudes[index]*np.ones(len(p_z_corr))
        for j in range(1,nphoto):
                index=j+nphoto*int(i)

                weight_mute=(1./nphoto)*amplitudes[index]*np.ones(len(pz[i][str(j)]['z_corrected']))
                p_z_corr=np.hstack((p_z_corr.T,pz[i][str(j)]['z_corrected'].T))
                weight_corr2=np.hstack((weight_corr2.T,weight_mute.T))
        #print (len(weight_corr2),len(weight_corr1),len(p_z),len(p_z_corr))

        #print (p_z.shape,weight_corr1.shape)
        plt.hist(p_z,bins=wz[i]['z_edges'],weights=weight_corr1,color='blue',alpha=0.4,label='photo_z',histtype='step',edgecolor='r')
        #plt.hist(p_z,bins=wz[i]['z_edges'],weights=weight_corr2,color='blue',alpha=0.4,label='photo_z_corrected',histtype='step',edgecolor='b')
        plt.hist(p_z_corr,bins=wz[i]['z_edges'],weights=weight_corr2,color='blue',alpha=0.4,label='photo_z_corrected',histtype='step',edgecolor='b')




        plt.errorbar(wz[i]['z_centers'],wz[i]['wz'],err,fmt='o',color='red',label='clustz')
    #    plt.errorbar(wz[i]['z_centers'],wz_corrected[i]['wz'],err_corrected,fmt='o',color='blue',label='clustz corrected')


        plt.xlim(min(wz[i]['z_centers']-0.1),max(wz[i]['z_centers']+0.4))
        plt.xlabel('$z$')
        plt.ylabel('$N(z)$')


        N,_=np.histogram(p_z,bins=wz[i]['z_edges'],weights=weight_corr1)
        N1,_=np.histogram(p_z_corr,bins=wz[i]['z_edges'],weights=weight_corr2)


        dict_stat1=compute_statistics(wz[i]['z_edges'],wz[i]['z_centers'],N,wz[i]['wz'],wz_cov[len1:(len1+wz[i]['wz'].shape[0]),len1:(len1+wz[i]['wz'].shape[0])],wz[i]['wz_jack'].T)
        dict_stat2=compute_statistics(wz[i]['z_edges'],wz[i]['z_centers'],N1,wz_corrected[i]['wz'],wz_cov_corrected[len1:(len1+wz[i]['wz'].shape[0]),len1:(len1+wz[i]['wz'].shape[0])],wz[i]['wz_jack'].T)

        len1+=wz[i]['wz'].shape[0]
        #put text where I want
        mute_phi=max(wz[i]['wz'])
        mute_z=max(wz[i]['z_centers'])


        #label_diag=''
        #if cov_mode=='diag':
        label_diag='_diag'

        ax.text(0.8, 0.95,'<z>_pdf_bin='+str(("%.3f" % dict_stat1['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.9,'<z>_pdf_bin_corr='+str(("%.3f" % dict_stat2['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.85,'median_pdf_bin='+str(("%.3f" % dict_stat1['median_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.8,'median_pdf_bin_corr='+str(("%.3f" % dict_stat2['median_true'])),fontsize=11, ha='center', transform=ax.transAxes)


        ax.text(0.8, 0.75,'<z>_clustz='+str(("%.3f" % dict_stat1['mean_rec']))+'+-'+str(("%.3f" % dict_stat1['mean_rec_err'+label_diag])),fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.7,'<z>_clustz_corr='+str(("%.3f" % dict_stat2['mean_rec']))+'+-'+str(("%.3f" % dict_stat2['mean_rec_err'+label_diag])),fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.65,'median_clustz='+str(("%.3f" % dict_stat1['median_rec']))+'+-'+str(("%.3f" % dict_stat1['median_rec_err'])),fontsize=12, ha='center', transform=ax.transAxes)


        ax.text(0.8, 0.6,'$\chi^2=$'+str(("%.3f" % dict_stat1['chi_diag']))+' ('+str(("%.3f" % dict_stat1['chi']))+') [DOF: '+str(wz[i]['wz'].shape[0])+']',fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.55,'$\chi^2=$'+str(("%.3f" % dict_stat2['chi_diag']))+' ('+str(("%.3f" % dict_stat2['chi']))+') [DOF: '+str(wz[i]['wz'].shape[0])+']',fontsize=11 , ha='center', transform=ax.transAxes)

        if nphoto==1:
            mask=pz[i][str(0)]['z_corrected']>0.
            mask1=pz[i]['z_true_residual']>0.
            residual=-np.mean(pz[i][str(0)]['z_corrected'][mask])+np.mean(pz[i]['z_true_residual'][mask1])
            ax.text(0.8, 0.5,'res shift='+str(("%.3f" % residual))+'+'+str(("%.3f" % errp[int(i)]))+'-'+str(("%.3f" % errm[int(i)])),fontsize=12 , ha='center', transform=ax.transAxes)

            results_output.update({'{0}_shift'.format(i):residual})
            results_output.update({'{0}_err+'.format(i):errp})
            results_output.update({'{0}_err-'.format(i):errm})
            results_output.update({'{0}_err='.format(i):errc})


        '''
    ax.text(0.8, 0.8,'std_pdf='+str(("%.3f" % dict_stat['std_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.75,'std_clustz='+str(("%.3f" % dict_stat['std_rec']))+'+-'+str(("%.3f" % dict_stat['std_rec_err'+label_diag])),fontsize=12 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.7,'$\chi^2/dof=$'+str(("%.3f" % dict_stat['chi_reduced'])),fontsize=12 , ha='center', transform=ax.transAxes)
        '''
        plt.legend(loc=2,prop={'size':10},fancybox=True)

        try:
            plt.savefig((output+'/{0}_{1}.pdf').format(label_save,int(i)+1), format='pdf', dpi=100)
            plt.close()
        except:
            pass
    save_obj(output+'/{0}_finalshifts'.format(label_save),results_output)


# save and load python objects ***************************************

def save_obj( name,obj ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
