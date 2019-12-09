import numpy as np
import copy
def setup_covariance(conf):
    default_estimator = np.zeros((len(conf["zrmg"] ),conf["n_jck"]))

    for jk in range(conf["n_jck"]):
        alp_bot = np.random.normal(0.,conf["e_alpha"],len(conf["zrmg"] ))/(conf["n_jck"]-1.)
        default_estimator[:,jk] = (conf["Nz"][:,jk+1]-(conf["alpha"]*(1+alp_bot)-2.)*conf["mag_hyperrank2"][0]-conf["bias_rmg"][:,jk+1]*conf["mag_hyperrank2"][0])/conf["bias_tot"][:,jk+1]
        default_estimator[:,jk] = default_estimator[:,jk]/np.trapz(default_estimator[:,jk],conf["zrmg"] )
        cov = covariance_jck(default_estimator,save_all['n_jck'],"jackknife")
    return np.linalg.inv(cov['cov'])



from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
import emcee


def weighted_percentile(data, percents, weights=None):

    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y

def constraints(param, w):
    low = weighted_percentile(param, 16, weights=w)
    high = weighted_percentile(param, 84, weights=w)
    
    low2 = weighted_percentile(param, 29, weights=w)
    high2 = weighted_percentile(param, 71, weights=w)

    return np.average(param, weights=w),low,high

def setup_priors(priors):
    initial_values = []
    list_var = []
    prior_function =dict()
    
    for key in priors.keys():
        if priors[key][0]=="uniform":
            prior_function.update({key: {'kind': 'uniform', 'weight': 1,
                                   'kwargs': {'loc': priors[key][1][0], 'scale': priors[key][1][1]-priors[key][1][0]}}}) 
        elif priors[key][0]=="gaussian":
            prior_function.update({key: {'kind': 'truncnorm', 'weight': 1,
                                    'kwargs': {'a': -8.0, 'b': 8.0,
                                               'loc': priors[key][1][0], 'scale': priors[key][1][1]}}})
            
    from scipy import stats
    for key in prior_function.keys():
        
        if "multi" in key:
            prior_function[key] = prior_function[key]
        else:
            prior_function[key]= getattr(stats, prior_function[key]['kind'])(**prior_function[key]['kwargs'])
    return prior_function

def estimate(conf,agents,nwalkers = 20,nsteps = 1000):
    def comput_nz(p0):
        hyp = int(p0[2]*conf["n_realisations"])
        try:
            y = (conf["Nz"][:,0]-2.*p0[0]*(np.array(p[2:]-2.)*conf["mag_hyperrank2"][ hyp ])-2*(p0[1]-2.)*conf["bias_rmg"][:,0]*conf["mag_hyperrank1"][ hyp ])/conf["bias_tot"][:,0]
            
        except:
            y = (conf["Nz"][:,0]-2.*p0[0]*(conf["alpha"]-2.)*conf["mag_hyperrank2"][ hyp ]-2*(p0[1]-2.)*conf["bias_rmg"][:,0]*conf["mag_hyperrank1"][ hyp ])/conf["bias_tot"][:,0]
            
        y = y/np.trapz(y,conf["zrmg"])
        return y,np.trapz(y*conf["zrmg"],conf["zrmg"])/np.trapz(y,conf["zrmg"])
        
    def compute_nz_som(p0):
        hyp = int(p0[2]*conf["n_realisations"])
        th = conf["nz_hyperrank"][hyp]
        tt = th/np.trapz(th,conf["zrmg"])
        return tt,np.trapz(tt*conf["zrmg"],conf["zrmg"])/np.trapz(tt,conf["zrmg"])
    def lnlike(p0):
      
        
        hyp = int(p0[2]*conf["n_realisations"])
        

        y,ym = comput_nz(p0)
        tt,ttm = compute_nz_som(p0)
        w = tt-y
        wm = ttm-ym
        if conf["method"] ==  'mean':
            
            
            chi2 = wm**2/conf['sys_m']**2
        if conf["method"] ==  'shape':
            chi2=np.matmul(w,np.matmul(conf["inv_cov"],w))
    
        if conf["method"] ==  'both':
            chi2=np.matmul(w,np.matmul(conf["inv_cov"],w))+ wm**2/conf['sys_m']**2
    

        
        
        return -0.5 * chi2
    
    def log_prob(p0):

        lp = np.sum(lnprior(p0))
        
        if not np.isfinite(lp):
            return -np.inf
 
        ll = lnlike(p0)
        
        if not np.isfinite(ll):
            return -np.inf
        return ll
    
    def lnprior(p0):


        logprior = []
        logprior.append(conf["prior_function"]['alpha_wl'].logpdf(p0[1]))
        logprior.append(conf["prior_function"]['b_wl'].logpdf(p0[0]))
        logprior.append(conf["prior_function"]['h'].logpdf(p0[2]))

        return np.array(logprior)
    
    ndim = 3
    conf["prior_function"]= setup_priors(conf["priors"])  
    i1=[]
    i2=[]
    i3=[]
    for j in range(nwalkers):
        i1.append(conf["prior_function"]['b_wl'].rvs())
        i2.append(conf["prior_function"]['alpha_wl'].rvs())
        i3.append(conf["prior_function"]['h'].rvs())
    initial = np.array(np.vstack([i1,i2,i3])).T
    #initial= np.random.uniform(0,1,nwalkers*ndim).reshape(nwalkers, ndim)
    #with closing(Pool(processes=agents)) as pool:
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim, log_prob)
            
    sampler.run_mcmc(initial, nsteps, progress=True)
    
    params = dict()
    
    b0,bl,bh  = constraints(sampler.chain[:,:,0].flatten(),np.ones(len(sampler.chain[:,:,0].flatten())))
    params["bwl"]=b0,b0-bl,bh-b0
    print "b_wl {0:2f} + {1:2f} - {2:2f}".format(b0,b0-bl,bh-b0)
    
    b1,bl,bh  = constraints(sampler.chain[:,:,1].flatten(),np.ones(len(sampler.chain[:,:,1].flatten())))
    print "alpha_wl {0:2f} + {1:2f} - {2:2f}".format(b1,b1-bl,bh-b1)
    params["awl"]=b1,b1-bl,bh-b1
    
    b2,bl,bh  = constraints(sampler.chain[:,:,0].flatten(),np.ones(len(sampler.chain[:,:,0].flatten())))
    
    print (b2  )                 
    print "h {0} + {1} - {2}".format(int(b2*conf["n_realisations"]),int(conf["n_realisations"]*(b2-bl)),int(conf["n_realisations"]*(bh-b2)))
    params["h"]=int(b2),int(b2-bl),int(bh-b2)
    return params, comput_nz([b0,b1,b2]),compute_nz_som([b0,b1,b2]),sampler.chain



def auto_fill_conf(data,maskz,binx,method ="mean"):
        
        conf = dict()
        conf["method"] = method
        conf["alpha"] = 0.#fmag(save_all_eboss['info']['z_ref'][maskz])
        conf["e_alpha"] = 0.2
        conf["zrmg"] = data['info']['z_ref'][maskz]
        data_vect = copy.copy(data['runs'][0])
        conf["Nz"]  = data_vect["Nz"][binx ,maskz,:]
        conf["bias_tot"] = np.sqrt(data_vect["bz"][binx,maskz,:]/data['b_wl'][binx,maskz,])
        conf["bias_rmg"] = compute_bias(data_vect["bz"],data['auto_theory'],binx)[maskz,:]
        conf["bias_wl"] = compute_bias(data['b_wl'],data['auto_theory'],binx)[maskz,:]
        conf["bias_wl_ave"] = np.trapz((conf["bias_wl"]*conf["Nz"])[:,0],conf["zrmg"])/np.trapz((conf["Nz"])[:,0],conf["zrmg"])

        conf["mag_hyperrank1"] = [data['mag_pos'][binx,maskz]]
        conf["mag_hyperrank2"] = [data['mag_pos1'][binx,maskz]]
        conf["n_jck"] = data['n_jck']
        return conf
    
    
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]
