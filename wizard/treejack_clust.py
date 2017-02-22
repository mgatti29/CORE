from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
import pandas as pd
import ipdb
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

cosmol=Planck15


class Jack(object):

    def __init__(self, w_estimator,conf,pairs_sel,rau,decu,raru,decru,rar,decr,rarr,decrr,jku,jkr,jkru,jkrr,weight_u,weight_r,weight_ur,weight_rr,FACT_DIST,z,mode,corr = 'NN', zu=None,zr=None,zur=None,zrr=None,centers=None, njk=None,verbose=False):
        '''
        Input:

        mode=density,cross,auto,auto_rp
        '''
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


    def jackknife_centers(self, cat_r=None):
        if self.centers is None:

            if cat_r is None: raise Exception('No random catalogue provided for the Jackknife centers')
            if self.verbose:
                print "Computing centers. Will be saved under 'centers.txt'"
            rand_file_a = cat_r
            A=pd.read_table(rand_file_a, sep=' ', header=None).T.as_matrix()
            ra_ra, dec_ra = A[0], A[1]
            if len(ra_ra) > 2*10**5:
                sel = np.random.choice(len(ra_ra), 2*10**5, replace=False)
                ra_ra , dec_ra = ra_ra[sel], dec_ra[sel]
            if self.verbose:
                print 'Length a randoms sample: %d'%len(ra_ra)
            radec_ra = np.zeros((len(ra_ra),2))
            radec_ra[:,0] = ra_ra
            radec_ra[:,1] = dec_ra
            km = kmeans_radec.kmeans_sample(radec_ra,self.njk,maxiter=500,tol=1e-05)
            self.centers = km.centers
            np.savetxt('centers.txt', self.centers)

        cnt = self.centers

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
        '''
        in a give jackknife region, it finds the furthest object from the region center.
        jackknives are defined on reference_randoms, so it uses them!
        '''
        cnt = self.centers
        
        ra_m_arr=[self.rau,self.rar,self.raru,self.rarr]
        dec_m_arr=[self.decu,self.decr,self.decru,self.decrr]
        jk_m_arr=[self.jku,self.jkr,self.jkru,self.jkrr]
        max_dist_region1=np.zeros((len(cnt),4))
        max_dist_region=np.zeros(len(cnt))
        for j in range(4):
        
            ra_m=ra_m_arr[j]
            dec_m=dec_m_arr[j]
            jk_m=jk_m_arr[j]


            # convert radec to xyz
            cosdec = np.cos(dec_m)
            aJx_u = cosdec * np.cos(ra_m)
            aJy_u = cosdec * np.sin(ra_m)
            aJz_u = np.sin(dec_m)




            for i in range(len(cnt)):
                if len(ra_m[jk_m==i]) ==0 or len(dec_m[jk_m==i])==0:
                   # print (ra_m[jk_m==i])
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
        for i in range(len(cnt)):
            #print (i,max_dist_region1[i,:])
            max_dist_region[i]=max(max_dist_region1[i,:])
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

    def jackknife_index(self, cat):

        f = cat
        A=pd.read_table(f, sep=' ', header=None).T.as_matrix()
        ra, dec = A[0], A[1]
        length = len(ra)
        if self.verbose:
            print 'Length '+cat+' sample: %d'%length
        try:
            jk = pd.read_table(cat[:-4]+'_index.txt', sep=' ', header=None).T.as_matrix()
            #if len(jk.shape) > 1: raise Exception
        except:
            print "Computing index. Will be saved under 'cat_index.txt'"
            if len(ra) > 3*10**5: numb = int(len(ra)/(3.*10**5)) + 1
            else: numb = 2

            l = [int(x) for x in np.linspace(0, len(ra), numb)]
            jks = []
            for i,x in enumerate(l[:-1]):
                t1 = time.time()
                ra2 = ra[l[i]:l[i+1]]
                dec2 = dec[l[i]:l[i+1]]
                radec = np.zeros((len(ra2),2))
                radec[:,0] = ra2
                radec[:,1] = dec2
                jks.append( kmeans_radec.find_nearest(radec, self.centers) )

            jk = np.concatenate((jks))
            np.savetxt(cat[:-4]+'_index.txt', jk)

        return ra, dec, jk, length

    def collect(self, pairs):
        N = self.N_1
        def rem_regA(jk):
            C = np.delete(pairs, jk, 0)
            C = np.delete(C, jk, 1)
            shp = C.shape
            C = C.reshape(shp[0]*shp[1], shp[2], shp[3])
            stack = np.sum(C, 0)
            DD, DR, RD, RR = np.array([stack[i]*N[0][jk]/N[i][jk] for i in range(len(N))])

            #if self.w_estimator=='lS':


            corr = self.estimator(DD,DR,RD,RR)
            if self.mode=='auto_rp':
                corr=corr*self.conf['max_rpar']*2.
            return corr,DD,DR,RD,RR


        def rem_reg2(jk):

            ind = np.arange(self.njk).tolist()
            ind = np.arange(njk).tolist()
            ind.remove(jk)
            inds = np.ix_(ind,ind)
            C = pairs[inds]
            shp = C.shape
            C = C.reshape(shp[0]*shp[1], shp[2], shp[3])
            stack = np.sum(C, 0)
            DD, DR, RD, RR = stack*N[0,jk]/N[:,jk]
            if self.mode=='auto_rp':
                corr=corr*self.conf['max_rpar']*2.

            corr = self.estimator(DD,DR,RD,RR)

            return corr

        MUTE1=np.array([rem_regA(jk) for jk in range(self.njk)])
        jackknifes=MUTE1[:,0]
        DD_j=MUTE1[:,1]
        DR_j=MUTE1[:,2]
        RD_j=MUTE1[:,3]
        RR_j=MUTE1[:,4]

        def estimate(x):
            l = len(x)-1
            mean = np.mean(x)
            std = np.sqrt(l*np.mean(abs(x - mean)**2))
            return mean, std

        _, ang = jackknifes.shape
        C = np.array([estimate(jackknifes[:,x]) for x in range(ang)])
        mean, std = C[:,0], C[:,1]

        N=self.cat_lengths
        shp = pairs.shape
        pairs = pairs.reshape(shp[0]*shp[1], shp[2], shp[3])
        stack = np.sum(pairs, 0)
        DD, DR, RD, RR = np.array([stack[i]*N[0]/N[i] for i in range(len(N))])
        corr = self.estimator(DD,DR,RD,RR)
        if self.mode=='auto_rp':
                corr=corr*self.conf['max_rpar']*2.
        return jackknifes, DD_j,DR_j ,RD_j,RR_j, mean, std, corr,DD,DR ,RD,RR





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
            if self.mode=='auto_rp':
                cat_a = treecorr.Catalog(ra=ra_a[jk_a==i], dec=dec_a[jk_a==i],r=z_a[jk_a==i], ra_units='deg', dec_units='deg',w=self.weight_u[jk_a==i])
            else:
                cat_a = treecorr.Catalog(ra=ra_a[jk_a==i], dec=dec_a[jk_a==i], ra_units='deg', dec_units='deg',w=self.weight_u[jk_a==i])
        except RuntimeError:
            cat_a = None
        try:

            if self.mode=='auto_rp':
                cat_b = treecorr.Catalog(ra=ra_b[jk_b==j], dec=dec_b[jk_b==j],r=z_b[jk_b==j], ra_units='deg', dec_units='deg',w=self.weight_r[jk_b==j])
            else:
                cat_b = treecorr.Catalog(ra=ra_b[jk_b==j], dec=dec_b[jk_b==j], ra_units='deg', dec_units='deg',w=self.weight_r[jk_b==j])

        except RuntimeError:
            cat_b = None


        try:
            if self.mode=='auto_rp':
                cat_rb = treecorr.Catalog(ra=ra_rb[jk_rb==j], dec=dec_rb[jk_rb==j],r=z_rb[jk_rb==j], ra_units='deg', dec_units='deg',w=self.weight_rr[jk_rb==j])
            else:
                cat_rb = treecorr.Catalog(ra=ra_rb[jk_rb==j], dec=dec_rb[jk_rb==j], ra_units='deg', dec_units='deg',w=self.weight_rr[jk_rb==j])
        except:
            cat_rb=None
        try:
            if self.mode=='auto_rp':
                cat_ra = treecorr.Catalog(ra=ra_ra[jk_ra==i], dec=dec_ra[jk_ra==i],r=z_ra[jk_ra==i], ra_units='deg', dec_units='deg',w=self.weight_ur[jk_ra==i])
            else:
                cat_ra = treecorr.Catalog(ra=ra_ra[jk_ra==i], dec=dec_ra[jk_ra==i], ra_units='deg', dec_units='deg',w=self.weight_ur[jk_ra==i])
        except:
            cat_ra=None


        # Create a NNCorrelation (2pt counts-counts) object
        # with the given configuration.

        dd = treecorr.NNCorrelation(self.conf)
        dr = treecorr.NNCorrelation(self.conf)
        rd = treecorr.NNCorrelation(self.conf)
        rr = treecorr.NNCorrelation(self.conf)





        if cat_a is not None and cat_b is not None:
            if self.mode=='auto_rp':
                if 'DD' in self.pairs_select:
                    dd.process(cat_a, cat_b,metric='Rperp')
            else:
                if 'DD' in self.pairs_select:
                    dd.process(cat_a, cat_b)

        if cat_a is not None and cat_rb is not None:
            if self.mode=='auto_rp':
                if 'DR' in self.pairs_select:
                    dr.process(cat_a, cat_rb,metric='Rperp')
            else:
                if 'DR' in self.pairs_select:
                    dr.process(cat_a, cat_rb)

        if cat_ra is not None and cat_b is not None:
            if self.mode=='auto_rp':
                if 'RD' in self.pairs_select:
                    dr.process(cat_ra, cat_b,metric='Rperp')
            else:
                if 'RD' in self.pairs_select:
                    dr.process(cat_ra, cat_b)

        if cat_ra is not None and cat_rb is not None:
            if self.mode=='auto_rp':
                if 'RR' in self.pairs_select:
                    dr.process(cat_ra, cat_rb,metric='Rperp')
            else:
                if 'RR' in self.pairs_select:
                    dr.process(cat_ra, cat_rb)
        pairs = {'theta': np.exp(dd.logr), 'DD': dd.weight, 'DR': dr.weight, 'RD': rd.weight, 'RR': rr.weight}



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


        Jra_u=ra_a[jk_a==i]
        Jdec_u=dec_a[jk_a==i]
        Jw_u=self.weight_u[jk_a==i]
        Jra_ru=ra_ra[jk_ra==i]
        Jdec_ru=dec_ra[jk_ra==i]
        Jw_ru=self.weight_ur[jk_ra==i]


        Jra_r=ra_b[jk_b==j]
        Jdec_r=dec_b[jk_b==j]
        Jw_r=self.weight_r[jk_b==j]
        Jra_rr=ra_rb[jk_rb==j]
        Jdec_rr=dec_rb[jk_rb==j]
        Jw_rr=self.weight_rr[jk_rb==j]



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
       # self.jackknife_centers(self.cats['cat_ra'])
        cnt = self.centers

        #
        if self.FACT_DIST != 'ALL':
            self.distance()
            self.max_distance_region()
        #this part: we just have to read jcknn column!

        '''
        ra_a, dec_a, jk_a, NA = self.jackknife_index(self.cats['cat_a'])
        ra_b, dec_b, jk_b, NB = self.jackknife_index(self.cats['cat_b'])
        ra_ra, dec_ra, jk_ra, NRA = self.jackknife_index(self.cats['cat_ra'])
        ra_rb, dec_rb, jk_rb, NRB = self.jackknife_index(self.cats['cat_rb'])
        '''


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
        mult = np.multiply.outer
        self.N = [mult(Na, Nb), mult(Na, Nrb), mult(Nra, Nb), mult(Nra, Nrb)]


        #ipdb.set_trace()
        N_A, N_B, N_RA, N_RB = NA-np.array(Na), NB-np.array(Nb), NRA-np.array(Nra), NRB-np.array(Nrb)


        if self.mode=='cross' or self.mode=='density':
            self.N_1 = [N_A*N_B, N_A*N_RB, N_RA*N_B, N_RA*N_RB]
        else:
            self.N_1 = [N_A*N_B, N_A*N_RB, N_RA*N_B, N_RA*N_RB]

        if 'nbins' in self.conf:
                shape = (4, self.conf['nbins'])
        else:
            raise Exception('Not implemented yet. Please use nbins in config.')
            #shape = (4, self.conf['bin_size'])

        cnt = self.centers



        t1=time.time()

        pairs = [[np.zeros(shape) for i in range(njk)] for j in range(njk)]
        a=np.concatenate((np.array([[(i,j) for i in range(njk)] for j in range(njk)])))
        #treecorr.config.set_omp_threads(self.conf['nodes'])
        #print (self.FACT_DIST)
        if self.FACT_DIST != 'ALL':
            todeg = self.convert_units()
            max_sep = self.conf['max_sep'] * todeg
            if self.mode=='auto_rp' or self.mode=='density':
                max_sep=max_sep/((1+self.z)*cosmol.angular_diameter_distance(self.z).value*(2*math.pi)/360)

            
            
            sel = np.array([ max([0.,(self.dist_cent(cnt[i],cnt[j]) - (self.max_dist_region[i]+self.max_dist_region[j]))]) < (5. * max_sep )for (i,j) in a])
            b = a[sel]
          #  print (b.shape,a.shape)
            #print (b)
        else:

            b=a


        def fun(x):
            i,j = x
            try:
                if  self.mode=='density':
                    pairs[i][j] = self.parallel_density(i,j)
                else:
                    pairs[i][j] = self.parallel(i,j)
            except RuntimeError, e:
                print e
                pairs[i][j] = np.zeros(shape)

        [fun(x) for x in b]

        self.pairs = np.array(pairs)




    def epilog(self):

        pairs = self.pairs

        jackknifes, DD_j,DR_j ,RD_j,RR_j, mean, std, corr,DD,DR ,RD,RR = self.collect(pairs)
        convert=self.convert_units()
        min_sep, max_sep, nedges = self.conf['min_sep']*convert, self.conf['max_sep']*convert, self.conf['nbins']+1
        th = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
        theta = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])


        pairs=dict()

        '''
        #THIS IS TO CROSS-CHECK FACT_DIST IS ENOUGH!
        cat_a = treecorr.Catalog(ra=self.rau, dec=self.decu, ra_units='deg', dec_units='deg',w=self.weight_u)
        cat_b = treecorr.Catalog(ra=self.rar, dec=self.decr, ra_units='deg', dec_units='deg',w=self.weight_r)
        cat_ra = treecorr.Catalog(ra=self.raru, dec=self.decru, ra_units='deg', dec_units='deg',w=self.weight_ur)
        cat_rb = treecorr.Catalog(ra=self.rarr, dec=self.decrr, ra_units='deg', dec_units='deg',w=self.weight_rr)


        dd = treecorr.NNCorrelation(self.conf)
        dr = treecorr.NNCorrelation(self.conf)
        rd = treecorr.NNCorrelation(self.conf)
        rr = treecorr.NNCorrelation(self.conf)
        dd.process(cat_a, cat_b)
        dr.process(cat_a, cat_rb)
        rd.process(cat_ra, cat_b)
        rr.process(cat_ra, cat_rb)
        dd_w=np.sum(self.weight_u)*np.sum(self.weight_r)
        #print(dd.tot,dd_w)
        #dd_w=np.sum(self.weight_u)*np.sum(self.weight_r)
        #print
        dd_w=np.sum(self.weight_u)*np.sum(self.weight_r)
        dr_w=np.sum(self.weight_u)*np.sum(self.weight_rr)
        rd_w=np.sum(self.weight_ur)*np.sum(self.weight_r)
        rr_w=np.sum(self.weight_ur)*np.sum(self.weight_rr)


        print (DD,DD/dd.weight,DR,DR/(dr.weight*dd_w/dr_w),RD,RD/(rd.weight*dd_w/rd_w),RR,RR/(rr.weight*dd_w/rr_w))


        total= (dd.weight-dr.weight*dd_w/dr_w-rd.weight*dd_w/rd_w+rr.weight*dd_w/rr_w)/(rr.weight*dd_w/rr_w)
        #print (total,corr)
        pairs.update({'w_total':total})

        '''


        for output in [[corr,jackknifes,'w'],[DD,DD_j,'DD'],[DR,DR_j,'DR'],[RD,RD_j,'RD'],[RR,RR_j,'RR']]:
        #for output in [[total,jackknifes,'w'],[dd.weight,DD_j,'DD'],[dr.weight*dd_w/dr_w,DR_j,'DR'],[rd.weight*dd_w/rd_w,RD_j,'RD'],[rr.weight*dd_w/rr_w,RR_j,'RR']]:
            new_array=np.zeros((len(mean),DD_j.shape[0]+1))
            new_array[:,0]=output[0]
            new_array[:,1:]=output[1].T
            pairs.update({output[2]:new_array})

        pairs.update({'theta':theta})
        pairs.update({'NDu':np.sum(self.weight_u)})
        pairs.update({'NRu':np.sum(self.weight_ur)})
        pairs.update({'NDr':np.sum(self.weight_r)})
        pairs.update({'NRr':np.sum(self.weight_rr)})
        pairs.update({'w_estimator':self.w_estimator})



        return pairs
