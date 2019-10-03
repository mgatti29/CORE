
from libc.math cimport cos,sin,acos,M_PI
import numpy as np
cimport numpy as np

def dist_cent(double ra1, double dec1,double ra2,double dec2):
    cdef double todeg
    cdef double mute
    cdef double mute1
    
    todeg = M_PI/180.
    ra1 = ra1*todeg
    ra2 = ra2*todeg
    dec1 = dec1*todeg
    dec2 = dec2*todeg
    mute = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1-ra2)
    mute1=acos(mute)/todeg
    return mute1

#def loop(int len_ball,double[:] add1, double[:] cosmol_dist_z, double[:,:] len_ball_result, double[:] aJra_r, double[:]

#def loop(double cosmol_dist_z, double aJra_r,double aJdec_r, double aJw_r, double[:] aJra_u ,double[:] aJdec_u, double[:] #aJw_u, int[:] ball_result_ii, double[:] radius, double[:] add1,int len_theta):

def loop(double cosmol_dist_z, double aJra_r,double aJdec_r, double aJw_r,  np.ndarray[double, ndim=1, mode="c"] aJra_u , np.ndarray[double, ndim=1, mode="c"] aJdec_u, np.ndarray[double, ndim=1, mode="c"] aJw_u, np.ndarray[int, ndim=1, mode="c"] ball_result_ii, np.ndarray[double, ndim=1, mode="c"] radius, np.ndarray[double, ndim=1, mode="c"] add1,int len_theta):


    cdef int index_add,index_u,jj,k,len_ball=len(ball_result_ii)

    cdef double weight
    

    for k in range(len_ball):
        index_u=ball_result_ii[k]
        weight=cosmol_dist_z*dist_cent(aJra_r,aJdec_r,aJra_u[index_u],aJdec_u[index_u])
        for jj in range(len_theta-1):
            if radius[jj]<weight<radius[jj+1] :
                add1[jj]=add1[jj]+aJw_r*aJw_u[index_u]/weight
  #  print add1
    return add1

def loop_old(double cosmol_dist_z, double aJra_r,double aJdec_r, double aJw_r,  np.ndarray[double, ndim=1, mode="c"] aJra_u , np.ndarray[double, ndim=1, mode="c"] aJdec_u, np.ndarray[double, ndim=1, mode="c"] aJw_u, np.ndarray[int, ndim=1, mode="c"] ball_result_ii, np.ndarray[double, ndim=1, mode="c"] radius, np.ndarray[double, ndim=1, mode="c"] add1,int len_theta):


    cdef int index_add,index_u,jj,k,qq,len_ball=len(ball_result_ii)

    cdef double weight
    

    for k in range(len_ball):
        index_u=ball_result_ii[k]
        weight=cosmol_dist_z*dist_cent(aJra_r,aJdec_r,aJra_u[index_u],aJdec_u[index_u])
        for jj in range(1,len_theta):
            for qq in range(jj):
                if radius[qq]<weight<radius[jj] :
                    index_add=qq+jj*len_theta
                    add1[index_add]=add1[index_add]+aJw_r*aJw_u[index_u]/weight
  #  print add1
    return add1
'''
def loop(int len_ball,double[:] add1, double[:] cosmol_dist_z, double[:,:] len_ball_result, double[:] aJra_r, double[:] aJdec_r, double[:] aJra_u, double[:] aJdec_u)
    cdef int ii,jj,qq
    cef double weight
    
    for ii in range(len(ball_result)):
        for k in range(len(ball_result[ii])):
            weight=cosmol_dist_z*dist_cent(aJra_r[ii],aJdec_r[ii],aJra_u[ball_result[ii][k]],aJdec_u[ball_result[ii][k]])
                    
            for jj in range(1,len(atheta_sep)):
                for qq in range(jj):
                    if radius[qq]<weight<radius[jj] :
                        add1[qq+jj*len(atheta_sep)]+=aJw_r[ii]*aJw_u[ball_result[ii][k]]/weight
    return add1


def K_weight(double[:,:] magarr_sn,double[:,:] magarr_BAO, double[:] weight):
    cdef int N=magarr_sn.shape[0]
    cdef int M=magarr_BAO.shape[0]
    cdef float a1
    cdef float a2
    cdef float a3
    cdef float a4
    cdef float b1
    cdef float b2
    cdef float b3
    cdef float b4
    cdef float dist1
    cdef float dist2
    cdef double[:,:] distances=np.zeros((N,15))
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(magarr_sn)
    distances, indices = nbrs.kneighbors((magarr_sn))
   # print M
   # print np.array(distances)
   # print distances.shape
   # print np.array(distances).shape
   # print indices.shape
    # distances & centre
    # vars()['MAGARR_SN_ARR'+str(j)][i],distances[j,-1],volume=distances[j,-1]**4.
    #applying this to the photometric sample:
    print 'weighting phase 2'
    for k in range(N):
        #print distances[k,14]
        #print np.array(distances)[k,14]
        for h in range(M):
        
        
            a1=magarr_BAO[h,0]
            a2=magarr_BAO[h,1]
            a3=magarr_BAO[h,2]
            a4=magarr_BAO[h,3]
            b1=magarr_sn[k,0]
            b2=magarr_sn[k,1]
            b3=magarr_sn[k,2]
            b4=magarr_sn[k,3]
            dist1=(a1-b1)*(a1-b1)
            dist1+=(a2-b2)*(a2-b2)
            dist1+=(a3-b3)*(a3-b3)
            dist1+=(a4-b4)*(a4-b4)
            dist2=distances[k,14]*distances[k,14]
            if dist1<dist2:
                weight[k]+=1
        weight[k]=(1./M)*weight[k]/15.
    
    return np.array(weight)




def check_duplicates(double[:] ra, double[:] dec, double[:,:]check_sample, i_check,double tol,self):
    cdef int N=ra.shape[0]
    cdef int M=check_sample.shape[0]
    cdef int i,j
    
    for j in range(N):
        for i in range(M):
            if self:
                if j!=i:
                
                    if (check_sample[i,0]-ra[j])*(check_sample[i,0]-ra[j])+(check_sample[i,1]-dec[j])*(check_sample[i,1]-dec[j])<tol:
                        #print i, j
                        #print '('+str(ra[j])+','+str(dec[j])+','+str(check_sample[j,2])+')     ('+str(check_sample[i,0])+','+str(check_sample[i,1])+','+str(check_sample[i,2])+')'
                        i_check[j]=1
            else:
                 if (check_sample[i,0]-ra[j])*(check_sample[i,0]-ra[j])+(check_sample[i,1]-dec[j])*(check_sample[i,1]-dec[j])<tol:
                       # print '('+str(ra[j])+','+str(dec[j])+')     ('+str(check_sample[i,0])+','+str(check_sample[i,1])+')'
                        i_check[j]=1

    return np.array(i_check)


def weighting(double[:] random_weights,double[:] random_weights_2, double[:] new_ra, double[:] new_dec, double[:] ra_weight, double[:] dec_weight, double[:] new_ra_rndm, double[:] new_dec_rndm):
    cdef int N=random_weights.shape[0]
    cdef int M=new_ra.shape[0]
    cdef int Q=ra_weight.shape[0]
    cdef int i, j

    for j in range(N):

        #q=4
        for i in range(M):
            if (((new_ra_rndm[j]-new_ra[i])**2.+(new_dec_rndm[j]-new_dec[i])**2.) < 0.05*0.05):
                random_weights[j]+=1
        #tot
        for i in range(Q):
            if (((new_ra_rndm[j]-ra_weight[i])**2.+(new_dec_rndm[j]-dec_weight[i])**2.) < 0.05*0.05):
                random_weights_2[j]+=1

        if random_weights_2[j]==0.:
            random_weights_2[j]=1.
        random_weights[j]=random_weights[j]/random_weights_2[j]

    return np.array(random_weights)
'''
