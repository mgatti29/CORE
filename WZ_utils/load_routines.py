

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


import numpy as np
import cPickle as pickle
import os
import shutil
import pandas as pd
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute


def read_results_fast(folder,n_jck,gamma,angular_bins):
    reference_bins_interval=load_obj(folder+'pairscount/reference_bins_interval')
    unknown_bins_interval=load_obj(folder+'pairscount/unknown_bins_interval')

    tomo_bins =  len(unknown_bins_interval['z'])
    reference_bins = len(reference_bins_interval['z'])
   
    
    z_ref = reference_bins_interval['z']

    # standard qz ****************************************
    
    qz = np.zeros((tomo_bins,reference_bins,n_jck+1))
    try:
        for i in range(tomo_bins):
            for j in range(reference_bins):
                z_ref_value = z_ref[j]
                mm = 1.
                try:
                    mute = load_obj(folder+"pairscount/pairs/CC_P_shear_{2}_{0}_{1}".format(i+1,j+1,angular_bins))
                except:
                    mute = load_obj(folder+"pairscount/pairs/CC_P_gl_{2}_{0}_{1}".format(i+1,j+1,angular_bins))
                xi = mute[0]-mute[2]
                xij = mute[1]-mute[3]
                dic = covariance_jck(mute[1].T,100,'jackknife')
                #plt.errorbar(mute[8],xi,dic['err'])

                qz[i,j,0] = np.trapz(xi/mm,mute[8])
                for jk in range(n_jck):
                    qz[i,j,jk+1] = np.trapz(xij[jk,:]/mm,mute[8])
    except:
        pass
    
    # Nz & qz corrected ***************************
    Nz = np.zeros((tomo_bins,reference_bins,n_jck+1))
    Nz_arr = np.zeros((tomo_bins,reference_bins,angular_bins,n_jck+1))
    qz_boost = np.zeros((tomo_bins,reference_bins,n_jck+1))
    qz_boost_arr = np.zeros((tomo_bins,angular_bins,reference_bins,n_jck+1))
    
    for i in range(tomo_bins):
            for j in range(reference_bins):
                z_ref_value = z_ref[j]

                path = folder+"pairscount/pairs/CC_P__{2}_{0}_{1}".format(i+1,j+1,angular_bins)

                m = make_wz_errors(path,'jackknife',n_jck,True)

                mask_nz = m[2][:,0]!=0.
                masks = m[2][:,0]==0.
                if len(masks[masks])>0. :
                    print (path)
                Nz[i,j,0] = np.trapz((m[1][mask_nz,0]-m[2][mask_nz,0])/m[2][mask_nz,0]/(m[0]**gamma)[mask_nz],m[0][mask_nz])
                
                Nz_arr[i,j,mask_nz,0] = (m[1][mask_nz,0]-m[2][mask_nz,0])/m[2][mask_nz,0]
                
                for jk in range(n_jck):
                    mask_nz = m[2][:,jk+1]!=0.
                    Nz[i,j,jk+1] = np.trapz((m[1][mask_nz,jk+1]-m[2][mask_nz,jk+1])/m[2][mask_nz,jk+1]/(m[0]**gamma)[mask_nz],m[0][mask_nz])
                    
                    Nz_arr[i,j,mask_nz,jk+1] =(m[1][mask_nz,jk+1]-m[2][mask_nz,jk+1])/m[2][mask_nz,jk+1]

                boost  = np.ones(len(m[1][:,0]))
                boostj  = np.ones((len(m[1][:,0]),len(m[2][0,1:])))

                mask =m[1][:,0]!=0.
                boost[mask] = m[1][mask,0]/m[2][mask,0]
                for jk in range(n_jck):

                    mask =m[2][:,jk]!=0.
                    boostj[mask,jk] = m[1][mask,1+jk]/m[2][mask,jk]

                try:

                    mm = 1.#/((1.+z_ref_value)*cosmol.angular_diameter_distance(z_ref_value).value)
                    try:
                        mute = load_obj(folder+"pairscount/pairs/CC_P_shear_{2}_{0}_{1}".format(i+1,j+1,angular_bins))
                    except:
                        mute = load_obj(folder+"pairscount/pairs/CC_P_gl_{2}_{0}_{1}".format(i+1,j+1,angular_bins))

                    xi = mute[0]
                    xij = mute[1]


                    dic = covariance_jck(mute[1].T,100,'jackknife')
                    #plt.errorbar(mute[8],xi,dic['err'])

                    qz_boost_arr[i,:,j,0] = (boost*xi/mm)
                    qz_boost[i,j,0] = np.trapz(boost*xi/mm,mute[8])

                    for jk in range(n_jck):
                        qz_boost_arr[i,:,j,jk+1] = (boostj[:,jk]*xij[jk,:]/mm)
                        qz_boost[i,j,jk+1] = np.trapz(boostj[:,jk]*xij[jk,:]/mm,mute[8])
                except:
                    pass
    
    # bias factor  ***************************

    bz = np.zeros((tomo_bins,reference_bins,n_jck+1))
    bz_arr = np.zeros((tomo_bins,angular_bins,reference_bins,n_jck+1))
    theta_arr = np.zeros((tomo_bins,angular_bins,reference_bins))
    for i in range(tomo_bins):
        
        for j in range(reference_bins):
            z_ref_value = z_ref[j]
            
            path = folder+"/pairscount/pairs/AC_R_P__{2}_{0}_{1}".format(i+1,j+1,angular_bins)
            m = make_wz_errors(path,'jackknife',n_jck,True)
    
            theta_arr[i,:,j] =m[0]
            bz_arr[i,:,j,0] =  (m[1][:,0]-m[2][:,0])/m[2][:,0]
            bz[i,j,0] = np.trapz((m[1][:,0]-m[2][:,0])/m[2][:,0]/(m[0]**gamma),m[0])
            for jk in range(n_jck):
                bz_arr[i,:,j,jk+1] =  (m[1][:,jk+1]-m[2][:,jk+1])/m[2][:,jk+1]
                bz[i,j,jk+1] = np.trapz((m[1][:,jk+1]-m[2][:,jk+1])/m[2][:,jk+1]/(m[0]**gamma),m[0])


            
    return {'Nz':Nz,'qz':qz,'qz_boost':qz_boost,'bz':bz,'bz_arr' :bz_arr, 'theta_arr':theta_arr,'qz_boost_arr':qz_boost_arr,"Nz_arr":Nz_arr}




def load_bias_wl(path_auto,n_jck,gamma,bins,ang_bins,nn=4):

    tomo_bins =  nn

    b_wl  = np.zeros((tomo_bins,bins,n_jck+1))
    b_wl_arr  = np.zeros((tomo_bins,bins,ang_bins,n_jck+1))
    for i in range(tomo_bins):
        for j in range(bins):
            path = path_auto[i]+"/pairscount/pairs/AC_R_P__{2}_{0}_{1}".format(1,j+1,ang_bins)
            m = make_wz_errors(path,'jackknife',n_jck,True)
            b_wl[i,j,0] = np.trapz((m[1][:,0]-m[2][:,0])/m[2][:,0]/(m[0]**gamma),m[0])
            if np.isnan(b_wl[i,j,0]):
                b_wl[i,j,0]=b_wl[i,j-1,0]
            if np.isinf(b_wl[i,j,0]):
                b_wl[i,j,0]=b_wl[i,j-1,0]
                
            for jk in range(n_jck+1):
                mask_nz = m[2][:,jk]!=0.
                b_wl_arr[i,j,mask_nz,jk] = (m[1][mask_nz,jk]-m[2][mask_nz,jk])/m[2][mask_nz,jk]
                    
                
            for jk in range(n_jck):
                    b_wl[i,j,jk+1] = np.trapz((m[1][:,jk+1]-m[2][:,jk+1])/m[2][:,jk+1]/(m[0]**gamma),m[0])
                    
                    

                        
                    if np.isnan(b_wl[i,j,jk+1]):
                        try:
                            (b_wl[i,j,jk+1]) = (b_wl[i,j-1,jk+1])
                        except:
                            pass

                    if np.isinf(b_wl[i,j,jk+1]):
                        try:
                            (b_wl[i,j,jk+1]) = (b_wl[i,j-1,jk+1])
                        except:
                            pass
    print 'done'
    return b_wl,b_wl_arr