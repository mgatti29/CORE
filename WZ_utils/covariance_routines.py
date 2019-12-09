
cosmo = {'omega_M_0':0.286, 
         'omega_lambda_0':1-0.286,
         'omega_k_0':0.0, 
         'omega_b_0' : 0.048,
         'h':0.69,
         'sigma_8' : 0.82,
         'n': 0.96}
import cosmolopy.distance as cd
import numpy as np
from astropy.cosmology import Planck15
cosmol = Planck15

def covariance_scalar_jck(TOTAL_PHI,jk_r, type_c = 'jackknife'):

  #  Covariance estimation
  if type_c == 'jackknife':
      fact=(jk_r-1.)/(jk_r)

  elif type_c=='bootstrap':
      fact=1./(jk_r)
        
  average=0.
  cov_jck=0.
  err_jck=0.


  for kk in range(jk_r):
    average+=TOTAL_PHI[kk]
  average=average/(jk_r)

  for kk in range(jk_r):
    #cov_jck+=TOTAL_PHI[kk]#*TOTAL_PHI[kk]

    cov_jck+=(-average+TOTAL_PHI[kk])*(-average+TOTAL_PHI[kk])


  err_jck=np.sqrt(cov_jck*fact)


  #average=average*(jk_r)/(jk_r-1)
  return {'cov' : cov_jck*fact,
          'err' : err_jck,
          'mean': average}


def compute_mean(z0,nz0):
    norm0 = 0.
    mean0 = 0.
    for kk in range(len(z0)):
        norm0+=nz0[kk]
        mean0+=nz0[kk]*z0[kk]
    return mean0/norm0
 
    
def compute_std(z0,nz0):
    norm0 = 0.
    mean0 = 0.
    std = 0.
    for kk in range(len(z0)):
        norm0+=nz0[kk]
        mean0+=nz0[kk]*z0[kk]
    mean0 = mean0/norm0
    norm0 = 0.
    for kk in range(len(z0)):
        norm0+=nz0[kk]
        std+=nz0[kk]*(z0[kk]-mean0)**2
    if std>0.and norm0>0.:
        return  np.sqrt(std/norm0)
    else:
    
        return 0.
    
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

  if type_cov=='bootstrap':
    average=average
  else:
    average=average*fact
  return {'cov' : cov_jck,
          'err' : err_jck,
          'corr':corr,
          'mean':average}


def covariance_jck_w(vec,jk_r,type_cov,w):

  #  Covariance estimation
    len_w = len(vec[:,0])
    jck = len(vec[0,:])
    
    err = np.zeros(len_w)
    mean = np.zeros(len_w)
    cov = np.zeros((len_w,len_w))
    corr = np.zeros((len_w,len_w))
    
    mean = np.zeros(len_w)
    for i in range(jk_r):
        mean += vec[:,i]*w[i]
    mean = mean/np.sum(w)
    
    if type_cov=='jackknife':
        fact=(np.sum(w))/np.sum(w)

    elif type_cov=='bootstrap':
        fact=1./(np.sum(w))
    
    for i in range(len_w):
        for j in range(len_w):
            for k in range(jck):
                cov[i,j] += (vec[i,k]- mean[i])*(vec[j,k]- mean[j])*fact*w[k]

    for ii in range(len_w):
        err[ii]=np.sqrt(cov[ii,ii])

  #compute correlation
    for i in range(len_w):
        for j in range(len_w):
            corr[i,j]=cov[i,j]/(np.sqrt(cov[i,i]*cov[j,j]))


    return {'cov' : cov,
          'err' : err,
          'corr':corr,
          'mean':mean}




def covariance_scalar_jck_w(TOTAL_PHI,jk_r, type_c = 'jackknife',w=0.):
  
  #  Covariance estimation 
  if type_c=='jackknife':
        fact=(np.sum(w))/np.sum(w)

  elif type_c=='bootstrap':
        fact=1./(np.sum(w))
        
  average=0.
  cov_jck=0.
  err_jck=0.


  for kk in range(jk_r):
    average+=TOTAL_PHI[kk]*w[kk]
  average=average/np.sum(w)

  for kk in range(jk_r):
    #cov_jck+=TOTAL_PHI[kk]#*TOTAL_PHI[kk]

    cov_jck+=(-average+TOTAL_PHI[kk])*(-average+TOTAL_PHI[kk])*fact*w[kk]


  err_jck=np.sqrt(cov_jck)


  #average=average*(jk_r)/(jk_r-1)
  return {'cov' : cov_jck,
          'err' : err_jck,
          'mean': average}