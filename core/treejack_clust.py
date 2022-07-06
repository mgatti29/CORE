from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
import pandas as pd

import copy
import timeit
import treecorr # Module for correlation functions, pip install TreeCorr.
import kmeans_radec # Module for finding centers and assign jackknife index. https://github.com/esheldon/kmeans_radec/

from scipy import spatial
from scipy.spatial import distance
import math
import astropy # astropy. https://github.com/astropy/astropy
from astropy.cosmology import Planck15 as Planck15
from treejack_cython import dist_cent,loop
from .dataset import save_obj,load_obj
import random
cosmol=Planck15
from .dataset import update_progress


class Jack(object):

    def __init__(self, jackknife_speedup,label_u,label_ur,label_r,label_rr, w_estimator,conf,pairs_sel,rau,decu,raru,decru,rar,decr,rarr,decrr,jku,jkr,jkru,jkrr,weight_u,weight_r,weight_ur,weight_rr,FACT_DIST,z,mode,corr = 'NN', zu=None,zr=None,zur=None,zrr=None,centers=None, njk=None,verbose=False):


        '''
        Input:

        mode=density,cross,auto,auto_rp
        '''
        self.jackknife_speedup=jackknife_speedup
        self.label_u=label_u
        self.label_ur=label_ur
        self.label_r=label_r
        self.label_rr=label_rr
        self.w_estimator=w_estimator
        self.conf = conf
        self.corr = corr
        self.rau=rau
        self.decu=decu
        self.raru=raru
        self.decru=decru
        self.rar=rar
        self.decr=decr
        self.rarr=rarr
        self.decrr=decrr
        self.zu=zu
        self.zr=zr
        self.zur=zur
        self.zrr=zrr
        self.z=z
        self.jku=jku
        self.jkr=jkr
        self.jkru=jkru
        self.jkrr=jkrr
        self.weight_u=weight_u
        self.weight_r=weight_r
        self.weight_ur =weight_ur
        self.weight_rr=weight_rr
        self.FACT_DIST=FACT_DIST
        self.centers = centers
        self.njk = njk
        self.mode=mode
        self.verbose=verbose
        self.pairs_select=pairs_sel
        if not self.njk:
            self.njk = 50

    def NNCorrelation(self):
        t0 = time.time()

        if self.verbose:
            print
            print time.ctime()
            print
        self.prolog()
        pairs= self.epilog()
        if self.verbose:
            print
            print "Elapsed time", time.strftime('%H:%M:%S',time.gmtime(time.time()-t0))
            print
        return pairs



    def convert_units(self):
        """Change to 'degrees' or to Mpc the units if necessary.
        """

        if 'sep_units' in self.conf.keys():
            un = self.conf['sep_units']
            if un == 'arcmin':
                todeg = 1./60.
            elif un == 'arcsec':
                todeg = 1./60.**2.
            elif un == 'kpc':
                todeg= 1./1000.
            else:
                todeg = 1.
        else:
            todeg = 1.
        return todeg


    def max_distance_region(self):
        cnt = self.centers
        '''
        in a given jackknife region, it finds the furthest object from the region center.
        I have to speed it up. Jakknife regions are always the same, one just has to compute the maximum distance for the unnown once.
        '''
        '''
        cnt = self.centers

        ra_m_arr=[self.rau,self.rar,self.raru,self.rarr]
        dec_m_arr=[self.decu,self.decr,self.decru,self.decrr]
        jk_m_arr=[self.jku,self.jkr,self.jkru,self.jkrr]
        max_dist_region1=np.zeros((len(cnt),4))
        max_dist_region=np.zeros(len(cnt))
        labels=[self.label_u,self.label_ur,self.label_r,self.label_rr]


        range_shuffled_j=random.sample(range(4),4)

        for j in range_shuffled_j:

         range_shuffled=random.sample(range(len(cnt)),len(cnt))
         if  os.path.exists('./pairscount/pairs_dist/'+labels[j]+'.pkl'):
                max_dist_region1[:,j]=load_obj('./pairscount/pairs_dist/'+labels[j])

         else:
            print ('building tree {0}'.format(labels[j]))

            ra_m=ra_m_arr[j]
            dec_m=dec_m_arr[j]
            jk_m=jk_m_arr[j]


            # convert radec to xyz
            cosdec = np.cos(dec_m)
            aJx_u = cosdec * np.cos(ra_m)
            aJy_u = cosdec * np.sin(ra_m)
            aJz_u = np.sin(dec_m)




            for i in range_shuffled:

                 if len(ra_m[jk_m==i]) ==0 or len(dec_m[jk_m==i])==0:
                    max_dist_region1[i,j]=0.
                 else:


                    ra_c,dec_c=cnt[i]

                    cosdec = np.cos(dec_c)
                    aJx_r = cosdec * np.cos(ra_c)
                    aJy_r = cosdec * np.sin(ra_c)
                    aJz_r = np.sin(dec_c)


                    tree_m=spatial.cKDTree(np.c_[aJx_u[jk_m==i], aJy_u[jk_m==i], aJz_u[jk_m==i]])

                    start=timeit.default_timer()

                    #min_dist_m=(tree_m.query([aJx_r,aJy_r,aJz_r],k=1))
                    max_dist_m,index_dist=tree_m.query([aJx_r,aJy_r,aJz_r],k=len(ra_m[jk_m==i]))


                    ra_new=ra_m[jk_m==i]
                    dec_new=dec_m[jk_m==i]


                    if (len(ra_m[jk_m==i])==1):
                        max_dist_region1[i,j]=self.dist_cent_2(ra_c,dec_c,ra_new[index_dist],dec_new[index_dist])
                    else:
                        max_dist_region1[i,j]=self.dist_cent_2(ra_c,dec_c,ra_new[index_dist[-1]],dec_new[index_dist[-1]])

                # save_distance_objects and speedup

            save_obj('./pairscount/pairs_dist/'+labels[j],max_dist_region1[:,j])
        for i in range(len(cnt)):
            max_dist_region[i]=max(max_dist_region1[i,:])
        '''
        max_dist_region=load_obj('./pairscount/pairs_dist/'+str(len(cnt)))
        self.max_dist_region=max_dist_region



    def distance(self):
        """Finds the minimum distance to a center for each center.
           Fixes double of this distance as the criteria for not considering correlations,
           which is a conservative choice. This distance has to be at least 4 times the
           maximum angular separation considered. Centers beyond this distance will not be
           considered in the correlations.
        """

        # Find the minimum distance to a center for each center.

        cnt = self.centers
        dist = np.array([np.sort([self.dist_cent(cnt[i],cnt[j]) for i in range(len(cnt))])[1] for j in range(len(cnt))])
        dist = (dist)*self.FACT_DIST



        todeg = self.convert_units()

        if 'max_sep' in self.conf.keys():
            max_sep = self.conf['max_sep'] * todeg

            if self.mode=='auto_rp' or self.mode=='density':
                #in this two cases, the maximum separation is in Mpc and needs to be converted into degrees.
                max_sep=max_sep/((1+self.z)*cosmol.angular_diameter_distance(self.z).value*(2*math.pi)/360)

        else:
            raise NotImplementedError("Make use of 'max_sep' in configuration.")



        # Check that the distance is at least 4 times the maximum angular separation.
        #print (max_sep,dist)
        self.center_min_dis = np.array( [ 4.*max_sep if x < 4.*max_sep else x for x in dist] )

    def estimator(self,DD,DR,RD,RR):

        if self.w_estimator == 'LS':
            results = (DD-DR-RD+RR)/(RR)
        elif self.w_estimator == 'Natural':
            results = (DD - RR) / (RR)
        elif self.w_estimator == 'Hamilton':
            results = (DD * RR - RD * DR) / (RD * DR)
        elif self.w_estimator == 'Natural_noRu':
            results = (DD - DR) / DR
        elif self.w_estimator == 'Natural_noRr':
            results = (DD - RD) / RD
        return results




    def dist_cent(self, a, b):
        """Angular distance between two centers (units: degrees). Makes use of spherical law of cosines.
        """
        todeg = np.pi/180.
        a = a*todeg
        b = b*todeg
        cos = np.sin(a[1])*np.sin(b[1]) + np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0])
        return np.arccos(cos)/todeg


    def cond(self, i, j):

        """Return the maximum conditional distance for a pair of centers to
           determine if the correlation should be computed.
        """

        con = self.center_min_dis
        return max(con[i], con[j])





    def collect(self, pairs):
        N=self.cat_lengths

        shape = (self.njk, self.conf['nbins'])
        DD_a,DR_a,RD_a,RR_a,DD_c,DR_c,RD_c,RR_c=np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)
        DD,DR,RD,RR=np.zeros( self.conf['nbins']),np.zeros( self.conf['nbins']),np.zeros( self.conf['nbins']),np.zeros( self.conf['nbins'])
        if self.jackknife_speedup:
            for n in range(self.conf['nbins']):
                DD[n] = np.sum(pairs[:,0,0,n]) + 0.5 * np.sum(pairs[:,1,0,n])
            #print (pairs[:][0][0],pairs[:][1][0])
                DR[n] = (np.sum(pairs[:,0,1,n]) + 0.5 * np.sum(pairs[:,1,1,n]))#*N[0]/N[1]
                RD[n] = (np.sum(pairs[:,0,2,n]) + 0.5 * np.sum(pairs[:,1,2,n]))#*N[0]/N[2]
                RR[n] = (np.sum(pairs[:,0,3,n]) + 0.5 * np.sum(pairs[:,1,3,n]))#*N[0]/N[3]

            # only the jk
                DD_a[:,n] = (pairs[:,0,0,n])
                DR_a[:,n] = (pairs[:,0,1,n])
                RD_a[:,n] = (pairs[:,0,2,n])
                RR_a[:,n] = (pairs[:,0,3,n])

                DD_c[:,n]  =  0.5 * (pairs[:,1,0,n])
                DR_c[:,n]  =  0.5 * (pairs[:,1,1,n])
                RD_c[:,n]  =  0.5 * (pairs[:,1,2,n])
                RR_c[:,n]  =  0.5 * (pairs[:,1,3,n])
            #print (DD,DR ,RD,RR, DD_a,DR_a ,RD_a,RR_a, DD_c,DR_c ,RD_c,RR_c)
        else:

            N=self.cat_lengths
            shp = pairs.shape
            pairs1 = pairs.reshape(shp[0]*shp[1], shp[2], shp[3])
            stack = np.sum(pairs1, 0)
            DD, DR, RD, RR = np.array([stack[i] for i in range(len(N))])

            N = self.N_1
            #print (pairs.shape)
            for jk in range(self.njk):
                DD_a[jk,:] = (pairs[jk,jk,0,:])
                DR_a[jk,:] = (pairs[jk,jk,1,:])
                RD_a[jk,:] = (pairs[jk,jk,2,:])
                RR_a[jk,:] = (pairs[jk,jk,3,:])
                for n in range(self.conf['nbins']):
                    DD_c[jk,n]  =  0.5 * (np.sum(pairs[jk,:,0,n])+np.sum(pairs[:,jk,0,n])-2*pairs[jk,jk,0,n])
                    DR_c[jk,n]  =  0.5 * (np.sum(pairs[jk,:,1,n])+np.sum(pairs[:,jk,1,n])-2*pairs[jk,jk,1,n])
                    RD_c[jk,n]  =  0.5 * (np.sum(pairs[jk,:,2,n])+np.sum(pairs[:,jk,2,n])-2*pairs[jk,jk,2,n])
                    RR_c[jk,n]  =  0.5 * (np.sum(pairs[jk,:,3,n])+np.sum(pairs[:,jk,3,n])-2*pairs[jk,jk,3,n])

        corr = self.estimator(DD,DR,RD,RR)
        if self.mode=='auto_rp':
                corr=corr*self.conf['max_rpar']*2.

        return corr, DD,DR ,RD,RR, DD_a,DR_a ,RD_a,RR_a, DD_c,DR_c ,RD_c,RR_c,self.jck_N



#***********************************************************************
    #pairs.shape :  jck  x  jck  x  4  x  number_bins

    # jackknife_realizations






    # bootstrap_realizations




    # full DD DR RD RR

#***********************************************************************

    def parallel(self, i, j):

        def dic(d):
            DD = d['DD']
            DR = d['DR']
            RD = d['RD']
            RR = d['RR']
            return [DD, DR, RD, RR]

        [[ra_a, dec_a, z_a, jk_a], [ra_b, dec_b, z_b, jk_b], [ra_ra, dec_ra, z_ra, jk_ra], [ra_rb, dec_rb, z_rb, jk_rb]] = self.info



        # Create the Catalog object. One for each jackknife region.
        try:

            mask=np.in1d(jk_a,i)
            #print ('dio',np.unique(jk_a[mask]),i)
            cat_a = treecorr.Catalog(ra=ra_a[mask], dec=dec_a[mask], ra_units='deg', dec_units='deg',w=self.weight_u[mask])
        except:
            cat_a = None
        try:
            mask=np.in1d(jk_b,j)
            #print ('diob',np.unique(jk_b[mask]),j)

            cat_b = treecorr.Catalog(ra=ra_b[mask], dec=dec_b[mask], ra_units='deg', dec_units='deg',w=self.weight_r[mask])

        except:
            cat_b = None


        try:
            mask=np.in1d(jk_rb,j)
            cat_rb = treecorr.Catalog(ra=ra_rb[mask], dec=dec_rb[mask], ra_units='deg', dec_units='deg',w=self.weight_rr[mask])
        except:
            cat_rb=None
        try:
            mask=np.in1d(jk_ra,i)
            cat_ra = treecorr.Catalog(ra=ra_ra[mask], dec=dec_ra[mask], ra_units='deg', dec_units='deg',w=self.weight_ur[mask])
        except:
            cat_ra=None


        # Create a NNCorrelation (2pt counts-counts) object
        # with the given configuration.

        dd = treecorr.NNCorrelation(self.conf)
        dr = treecorr.NNCorrelation(self.conf)
        rd = treecorr.NNCorrelation(self.conf)
        rr = treecorr.NNCorrelation(self.conf)

        if not ((cat_a == None) or (cat_b == None)):
            if self.mode=='auto_rp':
                if 'DD' in self.pairs_select:
                    dd.process(cat_a, cat_b,metric='Rperp')
            else:
                if 'DD' in self.pairs_select:
                    dd.process(cat_a, cat_b)

        if not ((cat_a == None) or (cat_rb == None)):
            if self.mode=='auto_rp':
                if 'DR' in self.pairs_select:
                    dr.process(cat_a, cat_rb,metric='Rperp')
            else:
                if 'DR' in self.pairs_select:
                    dr.process(cat_a, cat_rb)

        if not ((cat_ra == None) or (cat_b == None)):
            if self.mode=='auto_rp':
                if 'RD' in self.pairs_select:
                    rd.process(cat_ra, cat_b,metric='Rperp')
            else:
                if 'RD' in self.pairs_select:
                    rd.process(cat_ra, cat_b)

        if not ((cat_ra == None) or (cat_rb == None)):
            if self.mode=='auto_rp':
                if 'RR' in self.pairs_select:
                    rr.process(cat_ra, cat_rb,metric='Rperp')
            else:
                if 'RR' in self.pairs_select:
                    rr.process(cat_ra, cat_rb)
        try:
            pairs = {'theta': np.exp(dd.logr), 'DD': dd.weight, 'DR': dr.weight, 'RD': rd.weight, 'RR': rr.weight}
        except:
            pairs =  {'theta': np.zeros(self.conf['nbins']), 'DD': np.zeros(self.conf['nbins']), 'DR': np.zeros(self.conf['nbins']), 'RD': np.zeros(self.conf['nbins']), 'RR': np.zeros(self.conf['nbins'])}


        return dic(pairs)



    def dist_cent_2(self,ra1,dec1,ra2,dec2):

            todeg = np.pi/180.
            ra1 = ra1*todeg
            ra2 = ra2*todeg
            dec1 = dec1*todeg
            dec2 = dec2*todeg

            cos = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
            return np.arccos(cos)/todeg


    def parallel_density(self, i, j):
        def compute_spatial_obj(aJra_u,aJdec_u,aJra_r,aJdec_r,aJw_u,aJw_r,atheta_sep):

            cosmol_dist_z=((1+self.z)*cosmol.angular_diameter_distance(self.z).value*(2*math.pi)/360)



           # print end-start


            start3=timeit.default_timer()
            add1=np.zeros(len(atheta_sep)-1)
            if len(aJdec_r) ==0 or len(aJdec_u)==0:
                pass
            else:

                # convert radec to xyz
                cosdec = np.cos(aJdec_u)
                aJx_u = cosdec * np.cos(aJra_u)
                aJy_u = cosdec * np.sin(aJra_u)
                aJz_u = np.sin(aJdec_u)
                cosdec = np.cos(aJdec_r)
                aJx_r = cosdec * np.cos(aJra_r)
                aJy_r = cosdec * np.sin(aJra_r)
                aJz_r = np.sin(aJdec_r)

                # Tb = spatial.cKDTree(np.c_[aJra_u,aJdec_u])
                # Ta = spatial.cKDTree(np.c_[aJra_r,aJdec_r])
                Tb = spatial.cKDTree(np.c_[aJx_u, aJy_u, aJz_u])
                Ta = spatial.cKDTree(np.c_[aJx_r, aJy_r, aJz_r])


                ball_result= Ta.query_ball_tree(Tb,atheta_sep[-1])

                for ii in range(len(ball_result)):
                    #it could be speeded up putting also this loop in the cython module somehow.
                    BALL=np.array(ball_result[ii],dtype='<i')


                    add1=np.array(loop(cosmol_dist_z,aJra_r[ii],aJdec_r[ii],aJw_r[ii],aJra_u.astype('<d'),aJdec_u.astype('<d'),aJw_u.astype('<d'),BALL,radius.astype('<d'),add1.astype('<d'),len(atheta_sep)))

            return add1


        def dic(d):
            DD = d['DD']
            DR = d['DR']
            RD = d['RD']
            RR = d['RR']
            return [DD, DR, RD, RR]

        [[ra_a, dec_a, _,jk_a], [ra_b, dec_b, _,jk_b], [ra_ra, dec_ra, _,jk_ra], [ra_rb, dec_rb, _,jk_rb]] = self.info
        # Create the Catalog object. One for each jackknife region.




        mask=np.in1d(jk_a,i)
        Jra_u=ra_a[mask]
        Jdec_u=dec_a[mask]
        Jw_u=self.weight_u[mask]

        mask=np.in1d(jk_ra,i)
        Jra_ru=ra_ra[mask]
        Jdec_ru=dec_ra[mask]
        Jw_ru=self.weight_ur[mask]

        mask=np.in1d(jk_b,j)
        Jra_r=ra_b[mask]
        Jdec_r=dec_b[mask]
        Jw_r=self.weight_r[mask]

        mask=np.in1d(jk_rb,j)
        Jra_rr=ra_rb[mask]
        Jdec_rr=dec_rb[mask]
        Jw_rr=self.weight_rr[mask]



        # Create a NNCorrelation (2pt counts-counts) object
        # with the given configuration.

        #************  MODIFY->ANULUS   ************
        min_sep, max_sep, nedges = self.conf['min_sep'], self.conf['max_sep'], self.conf['nbins']+1
        radius = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
        radius =10**np.array([radius[i] for i in range(len(radius))])
        tompc=self.convert_units()
        radius*=tompc
        theta_sep=np.array([radius[i]/((1+self.z)*cosmol.angular_diameter_distance(self.z).value*(2*math.pi)/360) for i in range(len(radius)) ])


        #I convert the radial information into angular binning.

        dd=np.zeros(len(radius)-1)
        dr=np.zeros(len(radius)-1)
        rd=np.zeros(len(radius)-1)
        rr=np.zeros(len(radius)-1)

        if 'DD' in self.pairs_select:
            dd=compute_spatial_obj(Jra_u,Jdec_u,Jra_r,Jdec_r,Jw_u,Jw_r,theta_sep)
        if 'DR' in self.pairs_select:
            dr=compute_spatial_obj(Jra_u,Jdec_u,Jra_rr,Jdec_rr,Jw_u,Jw_rr,theta_sep)
        if 'RD' in self.pairs_select:
            rd=compute_spatial_obj(Jra_ru,Jdec_ru,Jra_r,Jdec_r,Jw_ru,Jw_r,theta_sep)
        if 'RR' in self.pairs_select:
            rr=compute_spatial_obj(Jra_ru,Jdec_ru,Jra_rr,Jdec_rr,Jw_ru,Jw_rr,theta_sep)

        ww=np.zeros(len(rr))
        pairs = {'radius': radius, 'theta': theta_sep, 'DD': dd, 'DR': dr, 'RD': rd, 'RR': rr}

        return dic(pairs)


    def prolog(self):

        #insert njk


        njk = self.njk

        cnt = self.centers

        #
        if self.FACT_DIST != 'ALL':
            self.distance()
            self.max_distance_region()

        ra_a, dec_a, z_a , jk_a, NA = self.rau, self.decu,self.zu,self.jku,np.sum(self.weight_u)
        ra_b, dec_b, z_b , jk_b, NB = self.rar, self.decr,self.zr,self.jkr,np.sum(self.weight_r)
        ra_ra, dec_ra, z_ra, jk_ra, NRA = self.raru, self.decru,self.zur,self.jkru,np.sum(self.weight_ur)
        ra_rb, dec_rb, z_rb, jk_rb, NRB = self.rarr, self.decrr,self.zrr,self.jkrr,np.sum(self.weight_rr)


        self.info = [[ra_a, dec_a, z_a, jk_a],[ra_b, dec_b, z_b, jk_b], [ra_ra, dec_ra, z_ra, jk_ra], [ra_rb, dec_rb, z_rb, jk_rb]]



        if self.mode=='cross' or self.mode=='density':
            self.cat_lengths = [NA*NB, NA*NRB, NRA*NB, NRA*NRB]
        else:
            self.cat_lengths = [NA*NB, NA*NRB, NRA*NB, NRA*NRB]

        def NN(ind,weight):
            lengths=np.zeros(njk)
            for n in range(njk):
                lengths[n]=np.sum(weight[ind==n])
            return lengths

        Na, Nb, Nra, Nrb = NN(jk_a,self.weight_u), NN(jk_b,self.weight_r), NN(jk_ra,self.weight_ur), NN(jk_rb,self.weight_rr)

        # len jacknives **********************************
        jck_N=np.zeros((njk,4))
        jck_N[:,0]= NN(jk_a,self.weight_u)
        jck_N[:,1]= NN(jk_b,self.weight_r)
        jck_N[:,2]= NN(jk_ra,self.weight_ur)
        jck_N[:,3]= NN(jk_rb,self.weight_rr)
        # ************************************************
        self.jck_N=jck_N

        N_A, N_B, N_RA, N_RB = NA-np.array(Na), NB-np.array(Nb), NRA-np.array(Nra), NRB-np.array(Nrb)

        self.N_1 = [N_A*N_B, N_A*N_RB, N_RA*N_B, N_RA*N_RB]

        if 'nbins' in self.conf:
                shape = (4, self.conf['nbins'])

        else:
            raise Exception('Not implemented yet. Please use nbins in config.')
            #shape = (4, self.conf['bin_size'])

        cnt = self.centers



        t1=time.time()

        pairs = [[np.zeros(shape) for i in range(njk)] for j in range(njk)]
        pairs_ring = [[np.zeros(shape) for i in range(2)] for j in range(njk)]
        a=np.concatenate((np.array([[(i,j) for i in range(njk)] for j in range(njk)])))
        #treecorr.config.set_omp_threads(self.conf['nodes'])
        #print (self.FACT_DIST)
        if self.FACT_DIST != 'ALL':
            todeg = self.convert_units()
            max_sep = self.conf['max_sep'] * todeg
            if self.mode=='auto_rp' or self.mode=='density':
                max_sep=max_sep/((1+self.z)*cosmol.angular_diameter_distance(self.z).value*(2*math.pi)/360)



            sel = np.array([ max([0.,(self.dist_cent(cnt[i],cnt[j]) - (self.max_dist_region[i]+self.max_dist_region[j]))]) < (3. * max_sep )for (i,j) in a])
            b = a[sel]
           # print (b.shape,a.shape)
           # print (b)
        else:

            b=a
        def fun(x):
            i,j = x
            try:
                if  self.mode=='density':
                    pairs[i][j] = self.parallel_density([i],[j])
                else:
                    pairs[i][j] = self.parallel([i],[j])
            except RuntimeError, e:
                print e
                pairs[i][j] = np.zeros(shape)

        def fun_speedup(othersample,otehrsample1,jackk):
            try:
                if  self.mode=='density':

                    pairsCC1=self.parallel_density(othersample,[jackk])
                    pairsCC2=self.parallel_density([jackk],othersample1)
                    pairs_auto=self.parallel_density([jackk],[jackk])
                    for prs in range(4):
                        pairsCC1[prs]+=pairsCC2[prs]
                    pairs_ring[jackk][1] = pairsCC1
                    pairs_ring[jackk][0] = pairs_auto
                else:
                    #print (othersample,[jackk])
                    #print (othersample1,[jackk])
                    pairsCC1=self.parallel(othersample,[jackk])
                    pairsCC2=self.parallel([jackk],othersample1)
                    pairs_auto=self.parallel([jackk],[jackk])

                    for prs in range(4):
                        pairsCC1[prs]+=pairsCC2[prs]
                    pairs_ring[jackk][1] = pairsCC1
                    pairs_ring[jackk][0] = pairs_auto
            except RuntimeError, e:
                print e
                pairs_ring[jackk][0] = np.zeros(shape)
                pairs_ring[jackk][1] = np.zeros(shape)

        start=timeit.default_timer()


        # if jackknife regions are too many, we are loosing efficiency in building the trees. for each jackknife regions, we can group the neighbours.
        #self.jackknife_speedup=True
        #print (len(b))
        mute=0
        
        # attempts

        
        if self.jackknife_speedup:

                for counter,jackk in enumerate(np.unique(b[:,1])):
                    mask=(b[:,1]==jackk) & (b[:,0]!=jackk)
                    othersample=b[mask,0]
                    mask=(b[:,0]==jackk) & (b[:,1]!=jackk)
                    othersample1=b[mask,1]
                    fun_speedup(othersample,othersample1,jackk)
                    mute+=len(othersample)+1
                    #update_progress(np.float(counter+1)/np.float(len(np.unique(b[:,1]))),timeit.default_timer(),start)
                self.pairs = np.array(pairs_ring)
        else:

            for counter,x in enumerate(b):
                #update_progress(np.float(counter)/np.float(len(b)),timeit.default_timer(),start)
                fun(x)
            self.pairs = np.array(pairs)
        #print (mute)



    def epilog(self):

        pairs = self.pairs

        corr, DD,DR ,RD,RR, DD_a,DR_a ,RD_a,RR_a, DD_c,DR_c ,RD_c,RR_c,self.jck_N = self.collect(pairs)
        convert=self.convert_units()
        min_sep, max_sep, nedges = self.conf['min_sep']*convert, self.conf['max_sep']*convert, self.conf['nbins']+1
        th = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
        theta = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])



        #print (corr.shape,DD.shape,DR_a.shape)#, DD,DR ,RD,RR, DD_a,DR_a ,RD_a,RR_a, DD_c,DR_c ,RD_c,RR_c)

        '''

        [[ra_a, dec_a, z_a, jk_a], [ra_b, dec_b, z_b, jk_b], [ra_ra, dec_ra, z_ra, jk_ra], [ra_rb, dec_rb, z_rb, jk_rb]]= self.info

        cat_l1 = treecorr.Catalog(ra=ra_a, dec=dec_a, ra_units='deg', dec_units='deg',w=self.weight_u)
        cat_l1r = treecorr.Catalog(ra=ra_ra, dec=dec_ra, ra_units='deg', dec_units='deg',w=self.weight_ur)
        cat_l2 = treecorr.Catalog(ra=ra_b, dec=dec_b, ra_units='deg', dec_units='deg',w=self.weight_r)
        cat_l2r = treecorr.Catalog(ra=ra_rb, dec=dec_rb, ra_units='deg', dec_units='deg',w=self.weight_rr)


        dd = treecorr.NNCorrelation(self.conf)
        dr = treecorr.NNCorrelation(self.conf)

        dd.process(cat_l1, cat_l2)
        dr.process(cat_l1, cat_l2r)

        dd_w=np.sum(self.weight_u)*np.sum(self.weight_r)
        dr_w=np.sum(self.weight_u)*np.sum(self.weight_rr)

        #pairs= (dd.weight-dr.weight*dd_w/dr_w-rd.weight*dd_w/rd_w+rr.weight*dd_w/rr_w)/(rr.weight*dd_w/rr_w)

        #print (dd.weight,DD[0],np.sum(DD_a[:,0])+np.sum(DD_c[:,0]))
        print (dd.weight[0],DD[0],np.sum(DD_a[:,0]),np.sum(DD_c[:,0]))
        '''

        pairs=dict()
        pairs.update({'theta':theta})
        pairs.update({'w_estimator':self.w_estimator})
        pairs.update({'w':corr})
        pairs.update({'DD':DD})
        pairs.update({'DR':DR})
        pairs.update({'RD':RD})
        pairs.update({'RR':RR})

        pairs.update({'DD_a':DD_a})
        pairs.update({'DR_a':DR_a})
        pairs.update({'RD_a':RD_a})
        pairs.update({'RR_a':RR_a})

        pairs.update({'DD_c':DD_c})
        pairs.update({'DR_c':DR_c})
        pairs.update({'RD_c':RD_c})
        pairs.update({'RR_c':RR_c})

        pairs.update({'jck_N':self.jck_N})

        return pairs

    
    
