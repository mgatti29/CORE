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
import os
from functions_nz import *
cosmol=Planck15


#*************************************************************************
#                         REGULARIZATION
#*************************************************************************
# we should only regularize the files we are interested into
def regularization_routine(best_params,reference_bins_interval,tomo_bins,N,jk_r,only_diagonal,set_negative_to_zero,fit,prior_gaussian_process,resampling,resp):


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

        for file1 in best_params.keys():
            file='{0}'.format(file1)

            if ('gaussian' not in file):


                hdf = pd.HDFStore('./output_dndz/TOMO_{0}/best_Nz/{1}_{2}_{3}_{4}.h5'.format(i+1,file,resampling,resp,jk_r))

                Nz = np.array(hdf['results'][0])
                Nz_jack=np.array(hdf['jackknife'])

                cov = np.array(hdf['cov'])
                err=np.zeros(cov.shape[0])
                z=np.array(hdf['z'][0])
                z_edges=np.array(hdf['z_edges'][0])

                #print (Nz.shape,Nz_jack.shape,z.shape,z_edges.shape,N.shape)
                hdf.close()
                for jj in range(cov.shape[0]):
                    err[jj]=np.sqrt(cov[jj,jj])

            # negative points corrections.

                if set_negative_to_zero == 'fixed':
                    Nz[Nz<0.]=0.
                    err[Nz<0.]=0.

                elif set_negative_to_zero == 'mcmc':
                    Nz_corrected,sigma_dwn,sigma_up,mean_z,sigma_mean_dwn,sigma_mean_up,std_z,std_dwn,std_up=negative_emcee(z,cov,Nz)

                if fit=='gaussian_process' :
                    label_save='{0}'.format(file)
                    output='./output_dndz/best_Nz/'
                    plot2(z,z_edges,zp_t_TOT,Nz,Nz_jack,N,label_save,output,jk_r,only_diagonal,cov,err,True,prior_gaussian_process,resampling)

                elif fit=='gaussian_match_pdf':
                    label_save='{0}'.format(file)
                    output='./output_dndz/TOMO_{0}/best_Nz/'.format(i+1)
                    plot3(z,z_edges,N[str(i)]['zp_t'],Nz,Nz_jack,N[str(i)]['N'],label_save,output,jk_r,only_diagonal,cov,err,True,prior_gaussian_process,resampling)

                else:
                    label_save='0fixed_{0}'.format(file.replace('.h5',''))
                    NN=np.zeros((Nz.shape[0],Nz_jack.shape[1]+1))
                    NN[:,0]=Nz
                    NN[:,1:]=Nz_jack
                    output='./output_dndz/TOMO_{0}/best_Nz/'.format(i+1)
                    #plot(resampling,reference_bins_interval,N,Nz[method],label_save,jk_r,)
                    #plot(resampling,reference_bins_interval,NN,Nz_tomo,label_save,jk_r,gaussian_process,set_to_zero,mcmc_negative,only_diagonal,verbose,save_fig=1):
                    plot2(z,z_edges,N[str(i)]['zp_t'],Nz,Nz_jack,N[str(i)]['N'],label_save,output,jk_r,only_diagonal,cov,err,False,False,resampling)
                    #plot2(z,z_bin,zp_t_TOT,Nz,Nz_jack,N,label_save,output,jk_r,only_diagonal,cov,err,prior_gaussian_process)

                    #TODO: standardiza plots output!
#*************************************************************************
#                 plotting & saving
#*************************************************************************
def plot2(z,z_bin,zp_t_TOT,Nz,Nz_jack,N,label_save,output,jk_r,only_diagonal,cov,err,gaussian,prior_gaussian_process,resampling):


    label_save=label_save.replace(".h5", "")
    if gaussian:
        with Silence(stdout='./output_dndz/gaussian_log{0}.txt'.format(label_save), mode='w'):

            dict_stat_gp,rec,theta,rec1,theta1,cov_gp=gaussian_process_module2(z_bin,z,Nz,err,cov,N,prior_gaussian_process)



    #compute statistics.

    dict_stat=compute_statistics(z_bin,z,N,Nz,cov,resampling,Nz_jack)


    fig= plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(zp_t_TOT,bins=z_bin,color='blue',alpha=0.4,label='True distribution',histtype='stepfilled',edgecolor='None')

    plt.errorbar(z,Nz,err,fmt='o',color='black',label='clustz')

    if gaussian:
        plt.plot(rec[:,0], rec[:,1], 'k', color='#CC4F1B',label='gaussian process')
        plt.fill_between(rec[:,0], rec[:,1]-rec[:,2], rec[:,1]+rec[:,2],
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


    plt.xlim(min(z-0.1),max(z+0.4))
    plt.xlabel('$z$')
    plt.ylabel('$N(z)$')


    #put text where I want
    mute_phi=max(Nz)
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
    ax.text(0.8, 0.6,'$\chi^2/dof=$'+str(("%.3f" % dict_stat['chi'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)

    if gaussian:
        ax.text(0.8, 0.55,'<z>_clustz_GP='+str(("%.3f" % dict_stat_gp['mean_rec']))+'+-'+str(("%.3f" % dict_stat_gp['mean_rec_err'+label_diag])),fontsize=11, ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.5,'std_clustz_GP='+str(("%.3f" % dict_stat_gp['std_rec']))+'+-'+str(("%.3f" % dict_stat_gp['std_rec_err'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)
        ax.text(0.8, 0.45,'median_clustz_GP='+str(("%.3f" % dict_stat_gp['median_rec'])),fontsize=11, ha='center', transform=ax.transAxes)



    plt.legend(loc=2,prop={'size':10},fancybox=True)

    if gaussian:
        plt.savefig((output+'/gaussian_{0}.pdf').format(label_save), format='pdf', dpi=100)
    else:
        plt.savefig((output+'/{0}.pdf').format(label_save), format='pdf', dpi=100)
    plt.close()





    if gaussian:

        save_obj((output+'/statistics_gauss_{0}').format(label_save),dict_stat_gp)

        pd.DataFrame(rec1[:,1]).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'results')
        pd.DataFrame(rec1[:,2]).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'err')
        pd.DataFrame(cov_gp).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'cov')
        pd.DataFrame(z).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'z')
        pd.DataFrame(z_bin).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'z_edges')

        pd.DataFrame(rec[:,0]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'results')
        pd.DataFrame(rec[:,1:]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'err')






    #return dict_stat,dict_stat_gp

def plot3(z,z_bin,zp_t_TOT,Nz,Nz_jack,N,label_save,output,jk_r,only_diagonal,cov,err,gaussian,prior_gaussian_process,resampling):



    label_save=label_save.replace(".h5", "")

    #* correct pz*******
    N_true,_=np.histogram(zp_t_TOT,bins=z_bin)

    mute_rel=(N_true-Nz)#/N_true


    cov1=np.zeros((cov.shape[0],cov.shape[1]))
    err1=np.zeros(err.shape[0])
    mute_rel=np.zeros(err.shape[0])

    for  ind_i in range(cov.shape[0]):

        #if (N_true[ind_i]!=0.):
            err1[ind_i]=err[ind_i]#/N_true[ind_i]
            mute_rel[ind_i]=(N_true[ind_i]-Nz[ind_i])#/N_true[ind_i]
        #else:
        #    mute_rel[ind_i]=0.

            for  ind_j in range(cov.shape[0]):
            #if (N_true[ind_i]!=0) & (N_true[ind_j]!=0.):
                cov1[ind_i,ind_j]=cov[ind_i,ind_j]#/(N_true[ind_i]*N_true[ind_j])
            #else:
            #    cov1[ind_i,ind_j]=0.

    if gaussian:
        #print (N)
        with Silence(stdout='./output_dndz/gaussian_log{0}.txt'.format(label_save), mode='w'):

            rec,theta,rec1,theta1,cov_gp=gaussian_process_module2(z_bin,z,mute_rel,err1,cov1,N_true,prior_gaussian_process)



    #compute statistics.

    dict_stat=compute_statistics(z_bin,z,N,Nz,cov,resampling,Nz_jack)


    fig= plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(zp_t_TOT,bins=z_bin,color='blue',alpha=0.4,label='True distribution',histtype='stepfilled',edgecolor='None')

    plt.errorbar(z,Nz,err,fmt='o',color='black',label='clustz')

    #print (N_true.shape,rec.shape)
    if gaussian:
            N_true,_=np.histogram(zp_t_TOT,bins=z_bin)
            mute_rel=(N_true-Nz)/N_true

            plt.plot(rec1[:,0], (N_true-rec1[:,1]), 'k', color='#CC4F1B',label='gaussian process' )
            plt.fill_between(rec1[:,0],  N_true- (rec1[:,1]-rec1[:,2]),(N_true- (rec1[:,1]+rec1[:,2])), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


    plt.xlim(min(z-0.1),max(z+0.4))
    plt.xlabel('$z$')
    plt.ylabel('$N(z)$')


    #put text where I want
    mute_phi=max(Nz)
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
    ax.text(0.8, 0.6,'$\chi^2/dof=$'+str(("%.3f" % dict_stat['chi'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)

    #if gaussian:
        #ax.text(0.8, 0.55,'<z>_clustz_GP='+str(("%.3f" % dict_stat_gp['mean_rec']))+'+-'+str(("%.3f" % dict_stat_gp['mean_rec_err'+label_diag])),fontsize=11, ha='center', transform=ax.transAxes)
        #ax.text(0.8, 0.5,'std_clustz_GP='+str(("%.3f" % dict_stat_gp['std_rec']))+'+-'+str(("%.3f" % dict_stat_gp['std_rec_err'+label_diag])),fontsize=11 , ha='center', transform=ax.transAxes)
        #ax.text(0.8, 0.45,'median_clustz_GP='+str(("%.3f" % dict_stat_gp['median_rec'])),fontsize=11, ha='center', transform=ax.transAxes)



    plt.legend(loc=2,prop={'size':10},fancybox=True)

    if gaussian:
        plt.savefig((output+'/gaussian_correction_{0}.pdf').format(label_save), format='pdf', dpi=100)
    else:
        plt.savefig((output+'/{0}.pdf').format(label_save), format='pdf', dpi=100)
    plt.close()





    if gaussian:

        #save_obj((output+'/statistics_gauss_{0}').format(label_save),dict_stat_gp)

        pd.DataFrame(rec1[:,1]).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'results')
        pd.DataFrame(rec1[:,2]).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'err')
        pd.DataFrame(cov_gp).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'cov')
        pd.DataFrame(z).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'z')
        pd.DataFrame(z_bin).to_hdf((output+'/gaussian_{0}.h5').format(label_save), 'z_edges')

        pd.DataFrame(rec[:,0]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'results')
        pd.DataFrame(rec[:,1:]).to_hdf((output+'/gaussian_full_{0}.h5').format(label_save), 'err')






    #return dict_stat,dict_stat_gp

def gaussian_process_module2(z_bin,z,Nz,err,cov,N,prior_gaussian_process):


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
    if prior_gaussian_process !='None':
        initheta=[np.float(prior_gaussian_process[0]),np.float(prior_gaussian_process[1])]


    #prior on theta[1]



    g=dgp.DGaussianProcess(z,Nz1,err1,cXstar=(xmin,xmax,nstar),prior=prior_theta,priorargs=(max(Nz)),grad='False')#,verbose=False) #)Xstar=z)
    g1=dgp.DGaussianProcess(z,Nz1,err1,Xstar=z,prior=prior_theta,priorargs=(max(Nz)),grad='False')#,verbose=False) #this is for the statistics.

    if prior_gaussian_process !='None':
        (rec,theta)=g.gp(theta=initheta,thetatrain='False')
        (rec1,theta1)=g1.gp(theta=initheta,thetatrain='False')
    else:
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


    #dict_stat_gp=compute_statistics(z_bin,z,N,Nz_gp,cov_gp,np.zeros((100,100)))
    return  rec,theta,rec1,theta1,cov_gp


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
