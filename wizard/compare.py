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

from .dataset import save_obj, load_obj, update_progress
from .functions_nz import compute_statistics,covariance_jck,Silence







def compare(labels=['BNZ_Newman'],
            shift_pz=[0.],
            cov_mode='diag',
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
            nwalkers=32, nburnin=100, nrun=500, live_dangerously=False,
            time0=0,number_of_cores=1,overwrite='True'):

            update_progress(0.)

            start_time=timeit.default_timer()
            if labels==['ALL_NZ']:
                if not os.path.exists('./output_dndz/compare_Nz/'):
                    os.makedirs('./output_dndz/compare_Nz/')
                for file in os.listdir('./output_dndz/Nz/'):
                    if file.endswith(".h5") and not ('BNz' in file):
                        labels=file.replace(".h5", "")
                        if ((os.path.exists('./output_dndz/compare_Nz/compare_'+labels+'_shifts.pkl')) and (overwrite=='False')):
                            pass
                        else:
                            compare_single(labels,0.,cov_mode,model_kwargs,priors,nwalkers,nburnin,nrun,live_dangerously,time0,0, './output_dndz/compare_Nz/','./output_dndz/Nz/')
            else:
                for i in range(len(labels)):
                #with Silence(stdout='./compare/compare_log.txt', mode='w'):
                    label=labels[i]
                    shift_pz_1=shift_pz[i]
                    compare_single(label,shift_pz_1,cov_mode,model_kwargs,priors,nwalkers,nburnin,nrun,live_dangerously,time0,i, './compare/','./output_dndz/best_Nz/')
                    update_progress(np.float((i+1.)/len(labels)),timeit.default_timer(),start_time)

            '''
            pool  = mp.Pool(number_of_cores)
            iterable = range(len(labels))
            func = partial(compare_single,labels,shift_pz,cov_mode,model_kwargs,priors,nwalkers,nburnin,nrun,live_dangerously,time0)

            pool.map(func, iterable)

            pool.close()
            pool.join()
            '''

            #for i,label_1 in enumerate(label):

                #shift_pz_1=shift_pz[i]

                #update_progress(np.float((i+1.)/len(label)),timeit.default_timer(),start_time)
                #return

def compare_single(label_multi='BNZ_Newman',
            shift_multi=0.,
            cov_mode='diag',
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
            nwalkers=32, nburnin=100, nrun=500, live_dangerously=False,
            time0=0,ik=1,compare_path='./compare/',path_w='./output_dndz/best_Nz/'):
    """Open up catalog and create dndz from it

    Parameters
    ----------
    label :             Where we extract the N(z)
    cov_mode :          string [diag, tomo, full] The kind of covariance
                        matrix we will use for the fits from the WZ
    nwalkers :          int [default 32] number of walkers emcee uses
    nburnin :           int [default 200] number of initial steps to take in
                        emcee fitter
    nrun :              int [default 2000] number of steps each walker will
                        take in sampling the parameter space
    live_dangerously :  override emcee's concerns about size of nwalkers
                        relative to ndims of the model
    priors :            dict of dicts. Each key corresponds to a different term
                        in equation. kind is the kind of scipy.stats function.
                        Weight is the relative weight of the prior (higher =
                        weight heavier). kwargs is a dictionary of kwargs
                        passed to construction of the stats function.
    model_kwargs :      dictionary of things to pass to the specific model for
                        fitting.

                        Here we have one kwarg: z0.  float [default 0.8] pivot
                        point for clustering bias term. Multiplies WZ by ((1 +
                        z) / (1 + z0)) ** gamma
    time0 :             float [default: 0] If non-zero, will give updates on
                        progress

    Notes
    -----

    """

    label=label_multi
    shift_pz_1=shift_multi
    print('{0} sample running'.format(label))



    if time0:
        update_progress(0, timeit.default_timer(), time0)


    compare_label = 'compare_{0}'.format(label)

    if not path.exists(compare_path):
        makedirs(compare_path)
    # load wz
    wz, wz_zbins, wz_zcenters, wz_cov, wz_jackknives,gaussian_process = load_wz(label, cov_mode=cov_mode,path_w=path_w)

    # load true redshift distribution in finer broad bins
    pz, pz_zbins, pz_zcenters,pz_binned,norm,zp_t_TOT = load_pz(wz_zbins,shift_pz_1)

    # we'll use the priors to create p0
    p0 = []

    # create the redshift bias object
    sampler = RedshiftBiasSampler(
        wz=wz, wz_zbins=wz_zbins, wz_zcenters=wz_zcenters, wz_cov=wz_cov,
        pz=pz, pz_zbins=pz_zbins, pz_zcenters=pz_zcenters,
        cov_mode=cov_mode,
        nwalkers=nwalkers, nburnin=nburnin, nrun=nrun,
        live_dangerously=live_dangerously,
        p0=p0, model_kwargs=model_kwargs, priors=priors, time0=time0)

    # run sampler
    sampler.walk()

    # save sampler
    save_sampler(sampler, compare_label, compare_path)

    # correct samples
    p = sampler.median_sample()
    biases = p[:-1:2]
    zp_t_TOT_corrected=zp_t_TOT-biases

    ntomo = len(wz)

    # correct PZ to be the same binning as WZ
    pz_corrected = sampler.correct_pz(pz,
        # [pz_zbins] * ntomo, [pz_zcenters] * ntomo,
        [wz_zbins] * ntomo, [pz_zcenters] * ntomo,
        p)


    wz_corrected,wz_jackknives_corrected  = sampler.correct_wz_jck(wz,wz_jackknives,
        [wz_zbins] * ntomo, [wz_zcenters] * ntomo,
        p,gaussian_process)

    cov_index = []
    for i in range(len(wz) + 1):
        cov_index.append(len(wz_zcenters) * i)
    if gaussian_process:
        wz_cov_corrected=None
    else:
        cov_full2 = _make_cov_array(wz_jackknives)
        wz_cov_corrected= convert_cov(cov_full2,
                      from_mode='full', to_mode=cov_mode, cov_index=cov_index)



    '''
    #just for testing purposes:
    wz_jackknives_corrected=wz_jackknives
    wz_corrected=wz
    zp_t_TOT_corrected=zp_t_TOT
    wz_cov_corrected=wz_cov
    '''
    fig = sampler.plot_flatchain()
    fig.savefig('{0}countour{1}.pdf'.format(compare_path,compare_label))

    '''
    #save pz corrected
    pd.DataFrame(zp_t_TOT).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'full_pz_uncorrected')
    pd.DataFrame(zp_t_TOT_corrected).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'full_pz_corrected')
    pd.DataFrame(pz_binned).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'pz_uncorrected')
    pd.DataFrame(pz_corrected).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'pz_corrected')
    pd.DataFrame(wz_zcenters).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'z')
    pd.DataFrame(wz_zbins).to_hdf('{0}{1}_pzcorrected.h5'.format(compare_path,label),'z_edges')
    # save wz corrected
    '''

    #plot
    if wz_cov_corrected:
        plot(norm,wz_zcenters,wz_zbins,zp_t_TOT,zp_t_TOT_corrected,np.array(wz)[0],np.array(wz_cov)[0],np.array(wz_corrected)[0],np.array(wz_cov_corrected)[0],'{0}'.format(compare_label),compare_path,cov_mode,(np.array(wz_jackknives_corrected).T)[:,:])
    else:
        plot(norm,wz_zcenters,wz_zbins,zp_t_TOT,zp_t_TOT_corrected,np.array(wz)[0],np.array(wz_cov)[0],np.array(wz)[0],np.array(wz_cov)[0],'{0}'.format(compare_label),compare_path,cov_mode,np.zeros((10,10)))
    # save wz
    # save_dndz(wz_corrected)
def save_wz(Nz,cov,z,z_bin,label_save):
    pd.DataFrame(Nz[:,0]).to_hdf(label_save, 'results')
    pd.DataFrame(Nz[:,1:]).to_hdf(label_save, 'jackknife')
    pd.DataFrame(cov).to_hdf(label_save, 'cov')
    pd.DataFrame(z).to_hdf(label_save, 'z')
    pd.DataFrame(z_bin).to_hdf(label_save, 'z_edges')



    return locals()

def load_wz(label, cov_mode='diag',path_w='./output_dndz/best_Nz/'):
    hpath = '{0}{1}.h5'.format(path_w,label)
    # Ntomo, Nz

    wz = pd.read_hdf(hpath, 'results').values.T[0][None, :]
    # (Njack, Ntomo, Nz)
    gaussian_process=False
    try:
        wz_samples = pd.read_hdf(hpath, 'jackknife').values.T[:, None, :]
    except:
        # I have not implemented the correction for gaussian processes;
        # the issue here is that I don t have the jackknives corrected!
        gaussian_process=True
        wz_samples_err = pd.read_hdf(hpath, 'err').values





    #reference_bins_interval=load_obj('./pairscount/reference_bins_interval')

    wz_zcenters = pd.read_hdf(hpath, 'z').values.T[0][None, :][0]
    wz_zbins = pd.read_hdf(hpath, 'z_edges').values.T[0][None, :][0]

    #print (wz_zcenters,wz_zbins)
    # normalize wz and wz_samples


    #wz = wz / np.sum(wz)
    norm_wz = normalize(wz[0], wz_zbins)
    wz = wz / norm_wz
    #norm_wz_samples = normalize(wz_samples[:,0,:], wz_zbins)
    if not gaussian_process:
        for i in range(wz_samples.shape[0]):
            norm_wz_samples = normalize(wz_samples[i,0,:], wz_zbins)
            wz_samples[i,0,:] = wz_samples[i,0,:] /norm_wz_samples
    else:
        wz_samples_err=wz_samples_err/norm_wz
        wz_samples=None


    # create the cov
    cov_index = []
    for i in range(len(wz) + 1):
        cov_index.append(len(wz_zcenters) * i)
    if not gaussian_process:
        cov_full = _make_cov_array(wz_samples)
        wz_cov = convert_cov(cov_full,
                      from_mode='full', to_mode=cov_mode, cov_index=cov_index)
    else:
        cov_full=np.zeros((wz_samples_err.shape[0],wz_samples_err.shape[0]))
        for ii in range(wz_samples_err.shape[0]):
            cov_full[ii,ii]=wz_samples_err[ii]**2.
        wz_cov = convert_cov(cov_full,
                      from_mode='full', to_mode=cov_mode, cov_index=cov_index)

    return wz, wz_zbins, wz_zcenters, wz_cov, wz_samples,gaussian_process

def load_pz(z_bins,shift_pz_1):
    hdd=pd.read_hdf('./pairscount/dataset.h5', 'unk')

    z = pd.read_hdf('./pairscount/dataset.h5', 'unk')['Z_T'].values
    z=z+shift_pz_1
    mask_z=z>=0.
    z=z[mask_z]
    tomo_bins = pd.read_hdf('./pairscount/dataset.h5', 'unk')['bins'].values

    # cuts the tomographic bin
    unknown_bins_interval=load_obj('./pairscount/unknown_bins_interval')
    unkn_z=unknown_bins_interval['z']
    tomo_bins=tomo_bins[mask_z]
    mask=(0.5<tomo_bins) & (tomo_bins<unkn_z.shape[0]+0.5)
    z=z[mask]
    norm_tomo=z.shape[0]
    pz_zbins = np.linspace(-0.5, 2.5, 15001)

    pz_zcenters = 0.5 * (pz_zbins[1:] + pz_zbins[:-1])

    pz, centers_out, bins_out, norm_out = catalog_to_histogram(z, pz_zbins, cut_to_range=False)
    pz_binned,_,_,_= catalog_to_histogram(z, z_bins, cut_to_range=False)

    # blow pz size
    pz = pz[None, :]

    return pz, pz_zbins, pz_zcenters, pz_binned,norm_tomo,z

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

def convert_cov(covs, from_mode, to_mode, cov_index=None):
    """Code to go from one way of presenting covs to another

    Parameters
    ----------
    covs :          Input covariance matrix or matrices
    from/to_mode :  string, must be in ['full', 'tomo', 'diag']
    cov_index :     List of ints; says when each tomo bin stars in the full cov
                    matrix. Obviously not needed it not converting from full

    Returns
    -------
    covs_to :   Converted covariance matrix or matrices in appropriate format

    Notes
    -----
    Here are the three covariance modes:

    diag :  A list of arrays of the same format as the dndz. Each entry
            corresponds to the dndz's sigma^2 error.
    tomo :  A list of 2d arrays. These are the covariance matrices within each
            tomographic bin.
    full :  A single 2d array. This is the full covariance

    """
    if from_mode == to_mode:
        return covs
    modes = ['full', 'tomo', 'diag']
    if from_mode not in modes or to_mode not in modes:
        raise Exception('from or to mode not available! from: {0}, to: {1}'.format(from_mode, to_mode))

    if to_mode == 'full':
        _cov_index = np.cumsum([0] + [len(covi) for covi in covs])
        ndim = sum([len(covi) for covi in covs])
        covs_to = np.zeros((ndim, ndim))
        for ith in range(len(_cov_index) - 1):
            i_start = _cov_index[ith]
            i_end = _cov_index[ith + 1]
            if from_mode == 'tomo':
                covi = covs[ith]
            elif from_mode == 'diag':
                covi = np.eye(len(covs[ith])) * covs[ith]
            covs_to[i_start:i_end] = covi
    elif to_mode == 'tomo':
        if from_mode == 'full':
            covs_to = []
            for ith in range(len(cov_index) - 1):
                i_start = cov_index[ith]
                i_end = cov_index[ith + 1]
                covs_to.append(covs[i_start:i_end, i_start:i_end])
        elif from_mode == 'diag':
            covs_to = [np.eye(len(covi)) * covi for covi in covs]
    elif to_mode == 'diag':
        if from_mode == 'full':
            covs_to = []
            for ith in range(len(cov_index) - 1):
                i_start = cov_index[ith]
                i_end = cov_index[ith + 1]
                covs_to.append(np.diag(covs[i_start:i_end, i_start:i_end]))
        elif from_mode == 'tomo':
            covs_to = [np.diag(covi) for covi in covs]

    return covs_to


def _make_cov_array(samples):
    # samples: [Njackknife, Ntomo, Nredshift]
    N_jackknife, N_tomo, N_redshift = samples.shape
    # create df from samples
    samples_flattened = np.array([sample.flatten() for sample in samples])

    jackknife, iz_and_tomo = np.mgrid[0:N_jackknife, 0:N_tomo * N_redshift]
    df = pd.DataFrame({'w': samples_flattened.flatten(),
                       'jackknife': jackknife.flatten(),
                       'iz_and_tomo': iz_and_tomo.flatten()})

    # too clever by half.
    # note that Cov_JK -> cov_ij = E((Xi - E(Xi))(Xj - E(Xj))) len(Nij - 1)
    # also note that this ends up being considerably faster than a nested
    # for-loop
    cov = df.pivot(index='jackknife', columns='iz_and_tomo', values='w').cov().values * (N_jackknife - 1)

    return cov



def save_sampler(sampler, label, sample_path):
    h5_path = '{0}/compare.h5'.format(sample_path)
    priors_pickle_path = '{0}/compare_priors_{1}.pkl'.format(sample_path, label)
    # save the chain
    chain = sampler.sampler._chain.copy()
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
    sample_median = sampler.median_sample()
    sample_error = sampler.error_sample()
    sample_asymmetric_error = sampler.error_asymmetric_sample()
    sample_argmax = sampler.best_sample()
    sample_samples = sampler.sample(N_samples)
    lnprior = sampler.lnprior(sample_median)  # for each param
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
    chi2 = -2 * sampler.lnlike(sample_median)  # for each tomo
    lnprob = sampler.lnprob(sample_median)
    lnprior_tot = np.sum(lnprior)
    results['chi2'] = {}
    results['chi2']['lnprior'] = results_str.format(lnprior_tot)
    results['chi2']['lnprob'] = results_str.format(lnprob)
    for chi2_i, chi2_val in enumerate(chi2):
        results['chi2']['tomo_{0}'.format(chi2_i)] = results_str.format(chi2_val)
        results['chi2']['dof_{0}'.format(chi2_i)] = '{0}'.format(len(sampler.wz[chi2_i]))

    with open(text_path, 'w') as f:
        f.write(yaml.dump(results, default_flow_style=False))

    results_output={'shift':results[sampler.labels[0]]['median'],'error+':results[sampler.labels[0]]['error+'],'error-':results[sampler.labels[0]]['error-']}
    save_obj('{0}/{1}_shifts'.format(sample_path, label),results_output)
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

def _chi2(estimate, truth, cov):
    from scipy import linalg
    cv_chol = linalg.cholesky(cov, lower=True)
    cv_sol = linalg.solve(cv_chol, estimate - truth, lower=True)
    chi2_val = 0.5 * np.sum(cv_sol ** 2)
    return chi2_val

def _tomo_chi2(wz, pz, cov, cov_mode):
    # returns chi2s but unsummed
    if cov_mode == 'full':
        chi2 = [_chi2(np.array(wz).flatten(), np.array(pz).flatten(), cov)]
    elif cov_mode == 'tomo':
        chi2 = [_chi2(wzi, pzi, covi) for wzi, pzi, covi in zip(wz, pz, cov)]
    elif cov_mode == 'diag':
        chi2 = [np.sum((wzi - pzi) ** 2 / covi) for wzi, pzi, covi in zip(wz, pz, cov)]
    return np.array(chi2)

def _compare_indices(self_z, other_z, other_index_in=None):
    """Check that the z match up, and return the subset that are close.
    """
    if not other_index_in:
        other_index_in = np.array([True] * len(other_z))
    # adjust indices based on redshifts.
    self_index = np.array([np.any(np.abs(zi - other_z[other_index_in]) < 0.0001)
                             for zi in self_z])
    other_index = np.array([np.any(np.abs(zi - self_z[self_index]) < 0.0001)
                             for zi in other_z])
    return self_index, other_index

class RedshiftBiasSampler(object):
    """Model a set of pz and wz as redshift bias
    """

    def __init__(self, wz=None, wz_zbins=None, wz_zcenters=None, wz_cov=None,
                 pz=None, pz_zbins=None, pz_zcenters=None,
                 cov_mode='full',
                 nwalkers=64, nburnin=100, nrun=500, live_dangerously=False,
                 p0=[],
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
        self.wz_cov = wz_cov
        self.pz = pz

        self.ndim = len(self.wz)
        # deal with the redshift bins
        if len(wz_zbins) != self.ndim:
            # blow up wz_zbins
            self.wz_zbins = [wz_zbins] * self.ndim
        else:
            self.wz_zbins = wz_zbins
        if len(wz_zcenters) != self.ndim:
            # blow up wz_zcenters
            self.wz_zcenters = [wz_zcenters] * self.ndim
        else:
            self.wz_zcenters = wz_zcenters
        if len(pz_zbins) != self.ndim:
            # blow up pz_zbins
            self.pz_zbins = [pz_zbins] * self.ndim
        else:
            self.pz_zbins = pz_zbins
        if len(pz_zcenters) != self.ndim:
            # blow up pz_zcenters
            self.pz_zcenters = [pz_zcenters] * self.ndim
        else:
            self.pz_zcenters = pz_zcenters

        # determine cov mode
        if len(self.wz_cov) == len(self.wz):
            if self.wz_cov[0].shape == self.wz[0].shape:
                self.wz_cov_mode = 'diag'
            else:
                self.wz_cov_mode = 'tomo'
        else:
            self.wz_cov_mode = 'full'

        # set up model kwargs
        self.model_kwargs = self.setup_model_kwargs(**model_kwargs)

        # set up labels
        self.labels = self.setup_labels()

        # set up priors
        self.priors = priors
        self.prior_function, self.prior_weight = self.setup_prior(priors)
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
        pz = self.correct_pz_full(self.pz, self.wz_zbins, self.pz_zcenters,
                                  self.wz_zbins, self.wz_zcenters, p)
        chi2 = self.evaluate(self.wz, pz, self.wz_cov)

        return -0.5 * chi2

    def lnprior(self, p):
        logprior = [prior.logpdf(pi) * wi for pi, prior, wi in zip(p, self.prior_function, self.prior_weight)]
        return logprior

    def lnprob(self, p):
        self.nsteps += 1

        if self.time0:
            progress = self.nsteps / self.nwalkers / (self.nburnin + self.nrun)
            update_progress(progress, timeit.default_timer(), self.time0)

        lp = np.sum(self.lnprior(p))
        if not np.isfinite(lp):
            return -np.inf

        ll = np.sum(self.lnlike(p))
        return lp + ll

    def evaluate(self, wz, pz, cov):
        # how well do wz and pz compare?
        return self._tomo_chi2(wz, pz, cov)

    # chi2
    @classmethod
    def _chi2(cls, estimate, truth, cov):
        return _chi2(estimate, truth, cov)

    def _tomo_chi2(self, estimate, truth, cov):
        return _tomo_chi2(estimate, truth, cov, self.wz_cov_mode)

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
            pdf[i] = spline.integral(zmin, zmax) / (zmax - zmin)
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
        pos = stuff[0]
        # run
        self.sampler.reset()
        stuff = self.sampler.run_mcmc(pos, nrun)

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

        return corner.corner(flatchain, labels=self.labels, **kwargs)

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
        # if kind == 'uniform':
        #     return stats.uniform(**kwargs)
        # elif kind == 'truncnorm':
        #     return stats.truncnorm(**kwargs)
        # elif kind == 'norm':
        #     return stats.norm(**kwargs)
        # else:
        #     raise Exception('Stats object not allowed!')
        return getattr(stats, kind)(**kwargs)

    # during fitting it is easier to lop all corrections on pz
    def correct_pz_full(self, pz, pz_zbins, pz_zcenters, wz_zbins, wz_zcenters,
                        p, **kwargs):
        # do both correct_pz and correct_wz on ONLY the pz
        pz_corrected = self.correct_pz(pz, pz_zbins, pz_zcenters, p, **kwargs)
        # by construction, pz correction and wz correction are just opposite
        pz_corrected_wz = self.correct_wz(pz_corrected, wz_zbins, wz_zcenters,
                                          -p, **kwargs)
        return pz_corrected_wz

    def correct_wz_full(self, wz, pz_zbins, pz_zcenters, wz_zbins, wz_zcenters,
                        p, **kwargs):
        # MARCO: I saw this is not used. I
        # TODO: Do I need to also add extra correction for cov?
        # do both correct_wz and correct_pz on ONLY the wz
        wz_corrected = self.correct_wz(wz, wz_zbins, wz_zcenters, p, **kwargs)
        # by construction, wz correction and pz correction are just opposite
        # we use wz_zbins to put wz_corrected_pz in same shape as output wz
        wz_corrected_pz = self.correct_pz(wz_corrected, pz_zbins, pz_zcenters,
                                          -p, **kwargs)
        return wz_corrected_pz

    ###########################################################################
    # TODO: In the future, the below functions are what we define for each
    #       sampler class!
    ###########################################################################

    # p to params
    @classmethod
    def params(cls, p):
        # go from p to meaningful params
        bias = p[:-1:2]
        amplitude = np.exp(p[1:-1:2])
        gamma = p[-1]
        return {'bias': bias, 'amplitude': amplitude, 'gamma': gamma}

    def setup_labels(self, ndim=-1):
        if ndim < 1:
            ndim = self.ndim
        labels = np.array([['$\Delta z_{{{0}}}$'.format(i + 1),
                            '$k_{{{0}}}$'.format(i + 1)]
                           for i in range(ndim)]).flatten().tolist()
        labels +=  ['$\gamma$']

        return labels

    def setup_model_kwargs(self, z0):
        return {'z0': z0}

    def setup_prior(self, priors):
        # convert prior from dictionary to appropriate list
        prior_function = []
        prior_weight = []
        for i in range(self.ndim):
            for key in ['deltaz', 'amplitude']:
                prior_function.append(self.interpret_prior(priors[key]['kind'], priors[key]['kwargs']))
                prior_weight.append(priors[key]['weight'])
        prior_function.append(self.interpret_prior(priors['gamma']['kind'], priors['gamma']['kwargs']))
        prior_weight.append(priors['gamma']['weight'])

        return prior_function, prior_weight

    # photoz
    def correct_pz_catalog(self, zcats, p, **kwargs):
        params = self.params(p)
        biases = params['bias']
        zcats_prime = [zcat - bias for zcat, bias in zip(zcats, biases)]
        return zcats_prime

    def correct_pz(self, pz, zbins, zcenters, p, catalog=False, **kwargs):
        """
        NOTE: zbins -> final binning scheme
        NOTE: zcenters -> initial redshifts of pz (shape should match pz)
        """
        if catalog:
            # if zcats are just catalogs, then apply catalog correction
            # TODO: weights are not included in here currently...
            pdfs = [catalog_to_histogram(zcat, zbin,
                    z_min=min(zbin), z_max=max(zbin),
                    cut_to_range=False)[0]  # TODO: cut_to_range=False?
                    for zcat, zbin
                    in zip(self.correct_pz_catalog(pz, p), zbins)]
        else:
            # if zcats are already binned pdfs, apply bias and rebin
            params = self.params(p)
            biases = params['bias']
            pdfs = []
            for pzi, zbin, zcenter, bias in zip(pz, zbins, zcenters, biases):
                pdf = self.rebin(zcenter - bias, pzi, zbin)
                pdfs.append(pdf)

        return pdfs

    # clustering
    def correct_wz_jck(self, wz, wz_jackknives,zbins, zcenters, p,gaussian_process, **kwargs):
        """
        NOTE: zbins -> not used
        NOTE: zcenters -> should match with input wz shape
        """
        params = self.params(p)
        amplitudes = params['amplitude']
        gamma = params['gamma']
        z0 = self.model_kwargs['z0']
        wzs_prime = [((1 + z) / (1 + z0)) ** gamma * amplitude * wzi
                     for z, wzi, amplitude in zip(zcenters, wz, amplitudes)]

        #wzs_prime = [wzi
        #             for z, wzi, amplitude in zip(zcenters, wz, amplitudes)]


        if gaussian_process:
            wz_jackknives_prime=None
        else:
            wz_jackknives_prime=copy.deepcopy(wz_jackknives)
            for i in range(wz_jackknives.shape[0]):
                wz_jackknives_prime[i,:] = [((1 + z) / (1 + z0)) ** gamma * amplitude * wzi  for z, wzi, amplitude in zip(zcenters, wz_jackknives[i,:], amplitudes)]
        #for i in range(wz_jackknives.shape[0]):
        #    wz_jackknives_prime[i,:] = [wzi  for z, wzi, amplitude in zip(zcenters, wz_jackknives[i,:], amplitudes)]
            wz_jackknives_prime=np.array(wz_jackknives_prime)[:,0,:]


        return wzs_prime,wz_jackknives_prime


    def correct_wz(self, wz, zbins, zcenters, p, **kwargs):
        """
        NOTE: zbins -> not used
        NOTE: zcenters -> should match with input wz shape
        """
        params = self.params(p)
        amplitudes = params['amplitude']
        gamma = params['gamma']
        z0 = self.model_kwargs['z0']
        wzs_prime = [((1 + z) / (1 + z0)) ** gamma * amplitude * wzi
                     for z, wzi, amplitude in zip(zcenters, wz, amplitudes)]


        '''
        if wz_jackknives:
            wz_jackknives_prime=wzs_prime = [((1 + z) / (1 + z0)) ** gamma * amplitude * wzi
                                 for z, wzi, amplitude in zip(zcenters, wz_jackknives, amplitudes)]
        '''

        return wzs_prime

    def correct_wz_cov(self, covs, zbins, zcenters, p, **kwargs):
        """
        NOTE: zbins -> not used
        NOTE: zcenters -> should match with input wz shape
        """
        params = self.params(p)
        amplitudes = params['amplitude']
        gamma = params['gamma']
        z0 = self.model_kwargs['z0']

        if self.wz_cov_mode == 'full':
            # need to keep track of length of different pieces:
            _cov_index = np.cumsum([0] + [len(wzi) for wzi in self.wz])
            covs_prime = covs.copy()
            for i, z_i, amplitude_i in zip(range(len(zcenters)), zcenters, amplitudes):
                mult_i = ((1 + z_i) / (1 + z0)) ** gamma * amplitude_i
                i_start = _cov_index[i]
                i_end = _cov_index[i + 1]
                for j, z_j, amplitude_j in zip(range(len(zcenters)), zcenters, amplitudes):
                    mult_j = ((1 + z_j) / (1 + z0)) ** gamma * amplitude_j
                    j_start = _cov_index[j]
                    j_end = _cov_index[j + 1]

                    covs_prime[i_start:i_end, j_start:j_end] = covs[i_start:i_end, j_start:j_end] * mult_i[None] * mult_j[:, None]

        elif self.wz_cov_mode == 'tomo':
            # gotta deal with the shape of cov
            covs_prime = []
            for z, cov, amplitude in zip(zcenters, covs, amplitudes):
                mult = ((1 + z) / (1 + z0)) ** gamma * amplitude
                covs_prime.append(cov * mult[None] * mult[:, None])
        elif self.wz_cov_mode == 'diag':
            covs_prime = [((1 + z) / (1 + z0)) ** gamma * amplitude * cov
                         for z, cov, amplitude in zip(zcenters, covs, amplitudes)]
        return covs_prime





def plot(norm,z,z_bin,zp_t_TOT,zp_t_TOT_corrected,wz,cov,wz_corrected,cov_corrected,label_save,output,cov_mode,wz_jack):

    #renormalize
    mute1=norm/np.sum(wz)
    wz=wz*mute1
    mute2=norm/np.sum(wz_corrected)
    wz_corrected=wz_corrected*mute1
    cov=cov*mute1*mute1
    cov_corrected=cov_corrected*mute1*mute1




    if cov_mode=='diag':
        err=np.sqrt(cov)
        err_corrected=np.sqrt(cov_corrected)
        cov=copy.deepcopy(np.diag(err*err))
        cov_corrected=copy.deepcopy(np.diag(err_corrected*err_corrected))
    else:
        err=np.zeros(cov.shape[0])
        err_corrected=np.zeros(cov.shape[0])
        for i in range(cov.shape[0]):
            err[i]=np.sqrt(cov[i,i])
            err_corrected[i]=np.sqrt(cov_corrected[i,i])

    fig= plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(zp_t_TOT,bins=z_bin,color='blue',alpha=0.4,label='photo_z',histtype='step',edgecolor='r')
    plt.hist(zp_t_TOT_corrected,bins=z_bin,color='blue',alpha=0.4,label='photo_z_corrected',histtype='step',edgecolor='b')




    plt.errorbar(z,wz,err,fmt='o',color='red',label='clustz')
    plt.errorbar(z,wz_corrected,err_corrected,fmt='o',color='blue',label='clustz corrected')


    plt.xlim(min(z-0.1),max(z+0.4))
    plt.xlabel('$z$')
    plt.ylabel('$N(z)$')

    N,_=np.histogram(zp_t_TOT,bins=z_bin)
    N1,_=np.histogram(zp_t_TOT_corrected,bins=z_bin)

    dict_stat1=compute_statistics(z_bin,z,N,wz,cov,wz_jack)
    dict_stat2=compute_statistics(z_bin,z,N1,wz_corrected,cov_corrected,wz_jack)

    #put text where I want
    mute_phi=max(wz)
    mute_z=max(z)


    label_diag=''
    if cov_mode=='diag':
        label_diag='_diag'



    ax.text(0.8, 0.95,'<z>_pdf_bin='+str(("%.3f" % dict_stat1['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.9,'<z>_pdf_bin_corr='+str(("%.3f" % dict_stat2['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.85,'median_pdf_bin='+str(("%.3f" % dict_stat1['median_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.8,'median_pdf_bin_corr='+str(("%.3f" % dict_stat2['median_true'])),fontsize=11, ha='center', transform=ax.transAxes)


    ax.text(0.8, 0.75,'<z>_clustz='+str(("%.3f" % dict_stat1['mean_rec']))+'+-'+str(("%.3f" % dict_stat1['mean_rec_err'+label_diag])),fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.7,'<z>_clustz_corr='+str(("%.3f" % dict_stat2['mean_rec']))+'+-'+str(("%.3f" % dict_stat2['mean_rec_err'+label_diag])),fontsize=12, ha='center', transform=ax.transAxes)

    '''
    ax.text(0.8, 0.8,'std_pdf='+str(("%.3f" % dict_stat['std_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.75,'std_clustz='+str(("%.3f" % dict_stat['std_rec']))+'+-'+str(("%.3f" % dict_stat['std_rec_err'+label_diag])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.7,'$\chi^2/dof=$'+str(("%.3f" % dict_stat['chi_reduced'])),fontsize=12 , ha='center', transform=ax.transAxes)
    '''
    plt.legend(loc=2,prop={'size':10},fancybox=True)


    plt.savefig((output+'/{0}.pdf').format(label_save), format='pdf', dpi=100)
    plt.close()


def covariance_jck_2(TOTAL_PHI,jk_r):

  #  Covariance estimation

  average=np.zeros(TOTAL_PHI.shape[1])
  cov_jck=np.zeros((TOTAL_PHI.shape[1],TOTAL_PHI.shape[1]))
  err_jck=np.zeros(TOTAL_PHI.shape[1])


  for kk in range(jk_r):
    average+=TOTAL_PHI[kk,:]
  average=average/(jk_r)

 # print average
  for ii in range(TOTAL_PHI.shape[1]):
     for jj in range(ii+1):
          for kk in range(jk_r):
            cov_jck[jj,ii]+=TOTAL_PHI[kk,ii,]*TOTAL_PHI[kk,jj]

          cov_jck[ii,jj]=(-average[jj]*average[ii]*jk_r+cov_jck[ii,jj])*(jk_r-1)/(jk_r)
          cov_jck[jj,ii]=cov_jck[ii,jj]

  for ii in range(TOTAL_PHI.shape[1]):
   err_jck[ii]=np.sqrt(cov_jck[ii,ii])
 # print err_jck

  #compute correlation
  corr=np.zeros((TOTAL_PHI.shape[1],TOTAL_PHI.shape[1]))
  for i in range(TOTAL_PHI.shape[1]):
      for j in range(TOTAL_PHI.shape[1]):
        corr[i,j]=cov_jck[i,j]/(np.sqrt(cov_jck[i,i]*cov_jck[j,j]))

  average=average*(jk_r)/(jk_r-1)
  return {'cov' : cov_jck,
          'err' : err_jck,
          'corr':corr,
          'mean':average}
