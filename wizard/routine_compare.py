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

#*******************************************************************

def update_progress(progress,elapsed_time=0,starting_time=0):
    import time
    import timeit
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
    #    status = "Done...\r\n"
    block = int(round(barLength*progress))



    if progress*100>1. and elapsed_time>0 :
        remaining=((elapsed_time-starting_time)/progress)*(1.-progress)
        text = "\rPercent: [{0}] {1:.2f}% {2}  - elapsed time: {3} - estimated remaining time: {4}".format( "#"*block + "-"*(barLength-block), progress*100, status,time.strftime('%H:%M:%S',time.gmtime(elapsed_time-starting_time)),time.strftime('%H:%M:%S',time.gmtime(remaining)))
    else:
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

#*******************************************************************

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):

    with open(name+ '.pkl', 'rb') as f:
        return pickle.load(f)


def plot(amplitude,norm,z,z_bin,photozs,photozs_corr,photozs_comparison,errp,errm,wz,cov,wz_corrected,cov_corrected,label_save,output,cov_mode,wz_jack):


 for i in range(wz.shape[0]):
    #renormalize
    hist_pz,_=np.histogram(photozs[str(i)],bins=z_bin[i,:])

    mute1=(norm[i])
    wz[i,:]=wz[i,:]*mute1
    cov[:,:,i]=cov[:,:,i]*mute1*mute1

    import copy

    wz_corrected=copy.deepcopy(wz)
    cov_corrected=copy.deepcopy(cov)

    mute1=(amplitude[i])
    wz_corrected[i,:]=wz[i,:]*mute1
    cov_corrected[:,:,i]=cov[:,:,i]*mute1*mute1

    err=np.zeros(cov.shape[0])
    err_corrected=np.zeros(cov.shape[0])
    for h in range(cov.shape[0]):
        err[h]=np.sqrt(cov[h,h,i])
        err_corrected[h]=np.sqrt(cov_corrected[h,h,i])

    fig= plt.figure()
    ax = fig.add_subplot(111)


    #computing residual *********************
    mask_corr=photozs_corr[str(i)]>0.
    z_corr=photozs_corr[str(i)][mask_corr]
    mask_comp=photozs_comparison[str(i)]>0.
    z_comp=photozs_comparison[str(i)][mask_comp]
    residual=-np.mean(z_corr)+np.mean(z_comp)
    #computing residual *********************



    plt.hist(photozs[str(i)],bins=z_bin[i,:],color='blue',alpha=0.4,label='photo_z',histtype='step',edgecolor='r')
    plt.hist(photozs_corr[str(i)],bins=z_bin[i,:],color='blue',alpha=0.4,label='photo_z_corrected',histtype='step',edgecolor='b')




    plt.errorbar(z[i,:],wz[i,:],err,fmt='o',color='red',label='clustz')
    plt.errorbar(z[i,:],wz_corrected[i,:],err_corrected,fmt='o',color='blue',label='clustz corrected')


    plt.xlim(min(z[i,:]-0.1),max(z[i,:]+0.4))
    plt.xlabel('$z$')
    plt.ylabel('$N(z)$')

    N,_=np.histogram(photozs[str(i)],bins=z_bin[i,:])
    N1,_=np.histogram(photozs_corr[str(i)],bins=z_bin[i,:])



    dict_stat1=compute_statistics(z_bin[i,:],z[i,:],N,wz[i,:],cov[:,:,i],wz_jack[:,i,:].T)
    dict_stat2=compute_statistics(z_bin[i,:],z[i,:],N1,wz_corrected[i,:],cov_corrected[:,:,i],amplitude[i]*wz_jack[:,i,:].T)
    dict_stat3=compute_statistics(z_bin[i,:],z[i,:],N1,wz[i,:],cov[:,:,i],wz_jack[:,i,:].T)

    #put text where I want
    mute_phi=max(wz[i,:])
    mute_z=max(z[i,:])


    label_diag=''
    if cov_mode=='diag':
        label_diag='_diag'



    ax.text(0.8, 0.95,'<z>_pdf_bin='+str(("%.3f" % dict_stat1['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.9,'<z>_pdf_bin_corr='+str(("%.3f" % dict_stat2['mean_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.85,'median_pdf_bin='+str(("%.3f" % dict_stat1['median_true'])),fontsize=11 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.8,'median_pdf_bin_corr='+str(("%.3f" % dict_stat2['median_true'])),fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.75,'std_pdf='+str(("%.3f" % dict_stat1['std_true'])),fontsize=12 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.7,'std_pdf_corr='+str(("%.3f" % dict_stat2['std_true'])),fontsize=12 , ha='center', transform=ax.transAxes)

    ax.text(0.8, 0.65,'$\chi^2=$'+str(("%.3f" % dict_stat1['chi_diag']))+' ('+str(("%.3f" % dict_stat1['chi']))+') [DOF: '+str(len(Nz[:,0]))+']',fontsize=11 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.6,'$\chi^2=$'+str(("%.3f" % dict_stat2['chi_diag']))+' ('+str(("%.3f" % dict_stat2['chi']))+') [DOF: '+str(len(Nz[:,0]))+']',fontsize=11 , ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.55,'$\chi^3=$'+str(("%.3f" % dict_stat['chi_diag']))+' ('+str(("%.3f" % dict_stat3['chi']))+') [DOF: '+str(len(Nz[:,0]))+']',fontsize=11 , ha='center', transform=ax.transAxes)

    ax.text(0.8, 0.5,'<z>_clustz='+str(("%.3f" % dict_stat1['mean_rec']))+'+-'+str(("%.3f" % dict_stat1['mean_rec_err'+label_diag]))+' ('+str(("%.3f" % dict_stat1['mean_rec_err']))+')',fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.45,'median_clustz='+str(("%.3f" % dict_stat1['median_rec']))+'+-'+str(("%.3f" % dict_stat1['median_rec_err'])),fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.8, 0.4,'std_clustz='+str(("%.3f" % dict_stat1['std_rec']))+'+-'+str(("%.3f" % dict_stat1['std_rec_err'+label_diag]))+' ('+str(("%.3f" % dict_stat1['std_rec_err']))+')',fontsize=12 , ha='center', transform=ax.transAxes)

    ax.text(0.8, 0.3,'res shift='+str(("%.3f" % residual))+'+'+str(("%.3f" % errp))+'-'+str(("%.3f" % errm)),fontsize=12 , ha='center', transform=ax.transAxes)




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

def compute_mean(z,N,z_edges,index,cov):
    mean_bin=0.
    norm_mean_bin=0.
    for jk in range(len(z)):
        mean_bin+=N[jk]*z[jk]
        norm_mean_bin+=N[jk]


    mean_cov=0.
    mean_cov_diag=0.
    for k in range(len(N)):
        for i in range(len(N)):
            if i==k:
                mean_cov_diag+=((cov[k,i,index])*(norm_mean_bin*z[k]-mean_bin*norm_mean_bin)*(norm_mean_bin*z[i]-mean_bin*norm_mean_bin))
            mean_cov+=((cov[k,i,index])*(norm_mean_bin*z[k]-mean_bin*norm_mean_bin)*(norm_mean_bin*z[i]-mean_bin*norm_mean_bin))
    mean_cov=np.sqrt(mean_cov)/(norm_mean_bin*norm_mean_bin)
    mean_cov_diag=np.sqrt(mean_cov_diag)/(norm_mean_bin*norm_mean_bin)

    return mean_bin/norm_mean_bin,mean_cov_diag

def compute_mean1(N,z):
    meanu=0.
    normu=0.
    stdu=0.
    for h in range(len(N)):
        meanu+=N[h]*z[h]
        normu+=N[h]
    meanu=meanu/normu
    for jk in range(len(N)):
        stdu+=N[jk]*(z[jk]-meanu)*(z[jk]-meanu)
    stdu=np.sqrt(stdu/normu)

    return meanu,stdu

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
