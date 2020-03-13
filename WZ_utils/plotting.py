import numpy as np
import copy
import numpy as np
from covariance_routines import *

def compute_mean(z,Nz):
    norm = 0.
    meana=0.
    for i in range(len(z)):
        norm+=Nz[i]
        meana+=z[i]*(Nz[i])
    return 1.*meana/norm
    
def plota(ax,data,lab='',cl='b' , i=0,tht = None, normed = False,bias = False, th_correction = False,boot=False,zmax_norm=[3.,3.,3.,3.,3.,3.],zmin_norm=[0.,0.,0.,0.,0.,0.],z_ref = np.linspace(0.15,0.9,20),sigma=5,z_reft=np.linspace(0.15,0.9,20),linestye='dashed'):
        
        ## normalize over the same range *******************************
        try:
            vector = copy.copy(data[i,:,:])

        except:
            vector = copy.copy(data[i,:])

        try:
            mute = bias[0]
            vector =vector/bias
        except:
            pass
        try:

            mute = th_correction[0]
            vector =vector/th_correction
        except:
            pass
            
       
        try:
            mask = (z_ref>zmin_norm[i]) & (z_ref<zmax_norm[i]) & (vector[:,0] ==vector[:,0])
            norm = np.trapz(vector[mask,0],z_ref[mask])
            
            cd = copy.copy(vector[:,1:])
            for g in range(vector[:,1:].shape[1]):
                cd[mask,g]=cd[mask,g]/np.trapz(cd[mask,g],z_ref[mask])
            if boot:
                derr = covariance_jck(cd,100,'bootstrap')
            else:
                derr = covariance_jck(cd,100,'jackknife')
        
            normth = np.trapz(tht[i][mask],z_ref[mask])
            if not normed:
                norm = 1.
                normth = 1.
            if boot:
                dzz=0.05
            else:
                dzz=0.
                
            #print lab, derr['err'][mask]/norm
            ax[0,i].errorbar(z_ref[mask]+dzz,vector[mask,0]/norm,derr['err'][mask],linestyle = linestye, label =lab,color = cl)
        
            #yy = ((data[i,mask,0]/norm)-(tht[i][mask]/normth))/(tht[i][mask]/normth)
            #ax[1,i].errorbar(z_ref[mask],yy ,derr['err'][mask]/norm/(tht[i][mask]/normth),linestyle = 'dashed', label =lab,color = cl)
            mute_dict1 = dict()

            means = compute_mean(z_ref[mask],vector[mask,0])
            stdd = compute_std(z_ref[mask],vector[mask,0])
            
            means = compute_mean(z_reft[mask],tht[i][mask])
            stdd = compute_std(z_ref[mask],tht[i][mask])
            
            
            
            mask = mask & (z_ref>(means-sigma*stdd)) & (z_ref<(means+sigma*stdd))# & (vector !=0.)
            
            norm = np.trapz(vector[mask,0],z_ref[mask])
            mute_dict1["final"] = vector[mask,0]/norm
            mute_dict1["dz"] = z_ref[mask]-means
            mute_dict1["z_ref"] = z_ref
            mute_dict1["mask"] = mask
            mute_dict1["truth"] = tht[i,mask]/np.trapz(tht[i][mask],z_ref[mask])
            
            means = compute_mean(z_ref[mask],vector[mask,0])-compute_mean(z_reft[mask],tht[i][mask])
            
            
            means_v = np.zeros(100)
            stds = compute_std(z_ref[mask],vector[mask,0])/compute_std(z_ref[mask],tht[i][mask])
            print " *********"
            print lab
            print "std: {0:2.4f} std true {1:2.4f}".format(compute_std(z_ref[mask],vector[mask,0]),compute_std(z_ref[mask],tht[i][mask]))
            print "interval: {0:2.4f},{1:2.4f}".format(compute_mean(z_ref[mask],vector[mask,0])-sigma*stdd,compute_mean(z_ref[mask],vector[mask,0])+sigma*stdd)
            std_v = np.zeros(100)
            for d in range(100):
                means_v[d] = compute_mean(z_ref[mask],vector[mask,d+1])-compute_mean(z_ref[mask],tht[i][mask])
                std_v[d] = compute_std(z_ref[mask],vector[mask,d+1])/compute_std(z_ref[mask],tht[i][mask])
            if boot:
                errd = covariance_scalar_jck(means_v,100,"bootstrap")
            else:
                errd = covariance_scalar_jck(means_v,100,"jackknife")
            #print means
            ax[1,i].axvspan(means-errd['err'],means+errd['err'] , alpha = 0.4,label =lab,color = cl)

            mute_dict1["mean"] = means
            mute_dict1["mean_err"] = errd['err']
            
            if boot:
                errd = covariance_scalar_jck(std_v,100,"bootstrap")
            else:
                errd = covariance_scalar_jck(std_v,100,"jackknife")
  

            mute_dict1["std"] =stds
            print "err {0:2.4f}".format(errd['err']*compute_std(z_ref[mask],tht[i][mask]))
            mute_dict1["std_err"] = errd['err']
            
            
            
            
            
            normth = np.trapz(vector[mask,0],z_ref[mask])/np.trapz(tht[i][mask],z_ref[mask])
         
            cd = copy.copy(vector[:,1:])
            for g in range(vector[:,1:].shape[1]):
                cd[mask,g]=cd[mask,g]
            if boot:
                derr = covariance_jck(cd,100,'bootstrap')
            else:
                derr = covariance_jck(cd,100,'jackknife')
                
            ax[3,i].errorbar(z_ref[mask],vector[mask,0]/tht[i,mask]/normth ,derr['err'][mask]/tht[i,mask]/normth ,linestyle = linestye, label =lab,color = cl)
            minz = means-sigma*stdd
            maxz = means+sigma*stdd
            
            return minz,maxz,mute_dict1
        
        
        
        except:
            
            mask = (z_ref>zmin_norm[i]) & (z_ref<zmax_norm[i]) & (vector !=0.)
            norm = np.trapz(vector[mask],z_ref[mask])
  
            normth = np.trapz(tht[i][mask],z_ref[mask])
            if not normed:
                norm = 1.
                normth = 1.
            ax[0,i].plot(z_ref[mask],vector[mask]/norm,linestyle = linestye, label =lab,color = cl)
        

            means = compute_mean(z_ref[mask],vector[mask])
            
            stdd = compute_std(z_ref[mask],vector[mask])
            
            mask = mask & (z_ref>(means-sigma*stdd)) & (z_ref<(means+sigma*stdd))# & (vector !=0.)
            means = compute_mean(z_ref[mask],vector[mask]-compute_mean(z_ref[mask],tht[i][mask]))
            
            
            print "interval: {0:2.2f} {1:2.2f}".format(means-sigma*stdd,means+sigma*stdd)
            ax[1,i].plot([0,0],[-1,1] ,linestyle = linestye, label =lab,color = cl)
        
        
            #ax[2,i].plot([1,1],[-1,1] ,linestyle = 'dashed', label =lab,color = cl)
        
            minz = means-sigma*stdd
            maxz = means+sigma*stdd
            return minz,maxz,{"mean":means}
        
        
        
        
def plota1(ax,data,lab='',cl='b' , i=0,tht = None, normed = False,bias = False, th_correction = False,boot=False,zmax_norm=[3.,3.,3.,3.,3.,3.],zmin_norm=[0.,0.,0.,0.,0.,0.],z_ref = np.linspace(0.15,0.9,20),sigma=5,z_reft=np.linspace(0.15,0.9,20),linestye='dashed'):
        
        ## normalize over the same range *******************************
        try:
            vector = copy.copy(data[i,:,:])

        except:
            vector = copy.copy(data[i,:])

        try:
            mute = bias[0]
            vector =vector/bias
        except:
            pass
        try:

            mute = th_correction[0]
            vector =vector/th_correction
        except:
            pass
            
       
        try:
            mask = (z_ref>zmin_norm[i]) & (z_ref<zmax_norm[i]) & (vector[:,0] ==vector[:,0])
            norm = np.trapz(vector[mask,0],z_ref[mask])
            
            cd = copy.copy(vector[:,1:])
            for g in range(vector[:,1:].shape[1]):
                cd[mask,g]=cd[mask,g]/np.trapz(cd[mask,g],z_ref[mask])
            if boot:
                derr = covariance_jck(cd,100,'bootstrap')
            else:
                derr = covariance_jck(cd,100,'jackknife')
        
            normth = np.trapz(tht[i][mask],z_ref[mask])
            if not normed:
                norm = 1.
                normth = 1.
            if boot:
                dzz=0.05
            else:
                dzz=0.
                
            #print lab, derr['err'][mask]/norm
            ax[0,i].errorbar(z_ref[mask]+dzz,vector[mask,0]/norm,derr['err'][mask],linestyle = linestye, label =lab,color = cl)
        
            #yy = ((data[i,mask,0]/norm)-(tht[i][mask]/normth))/(tht[i][mask]/normth)
            #ax[1,i].errorbar(z_ref[mask],yy ,derr['err'][mask]/norm/(tht[i][mask]/normth),linestyle = 'dashed', label =lab,color = cl)
            mute_dict1 = dict()

            means = compute_mean(z_ref[mask],vector[mask,0])
            stdd = compute_std(z_ref[mask],vector[mask,0])
            
            means = compute_mean(z_reft[mask],tht[i][mask])
            stdd = compute_std(z_ref[mask],tht[i][mask])
            
            
            
            mask = mask & (z_ref>(means-sigma*stdd)) & (z_ref<(means+sigma*stdd))# & (vector !=0.)
            
            norm = np.trapz(vector[mask,0],z_ref[mask])
            mute_dict1["final"] = vector[mask,0]/norm
            mute_dict1["final_jck"] = vector[mask,1:]/norm
            mute_dict1["final_full"] = vector[:,0]/norm
            mute_dict1["final_full_jck"] = vector[:,1:]/norm
        
            mute_dict1["dz"] = z_ref[mask]-means
            mute_dict1["dz_n"] = z_ref-means
            mute_dict1["z_ref"] = z_ref
            mute_dict1["mask"] = mask
            mute_dict1["truth"] = tht[i,mask]/np.trapz(tht[i][mask],z_ref[mask])
            mute_dict1["truth_full"] = tht[i,:]/np.trapz(tht[i][mask],z_ref[mask])
            
            means = compute_mean(z_ref[mask],vector[mask,0])-compute_mean(z_reft[mask],tht[i][mask])
            
            
            means_v = np.zeros(100)
            stds = compute_std(z_ref[mask],vector[mask,0])/compute_std(z_ref[mask],tht[i][mask])
            #print " *********"
            #print lab
            #print "std: {0:2.4f} std true {1:2.4f}".format(compute_std(z_ref[mask],vector[mask,0]),compute_std(z_ref[mask],tht[i][mask]))
            #print "interval: {0:2.4f},{1:2.4f}".format(compute_mean(z_ref[mask],vector[mask,0])-sigma*stdd,compute_mean(z_ref[mask],vector[mask,0])+sigma*stdd)
            std_v = np.zeros(100)
            for d in range(100):
                means_v[d] = compute_mean(z_ref[mask],vector[mask,d+1])-compute_mean(z_ref[mask],tht[i][mask])
                std_v[d] = compute_std(z_ref[mask],vector[mask,d+1])/compute_std(z_ref[mask],tht[i][mask])
            if boot:
                errd = covariance_scalar_jck(means_v,100,"bootstrap")
            else:
                errd = covariance_scalar_jck(means_v,100,"jackknife")
            #print means
            ax[2,i].axvspan(means-errd['err'],means+errd['err'] , alpha = 0.4,label =lab,color = cl)

            mute_dict1["mean"] = means
            mute_dict1["mean_err"] = errd['err']
            
            if boot:
                errd = covariance_scalar_jck(std_v,100,"bootstrap")
            else:
                errd = covariance_scalar_jck(std_v,100,"jackknife")
  

            mute_dict1["std"] =stds
            #print "err {0:2.4f}".format(errd['err']*compute_std(z_ref[mask],tht[i][mask]))
            mute_dict1["std_err"] = errd['err']
            
            
            
            
            
            normth = np.trapz(vector[mask,0],z_ref[mask])/np.trapz(tht[i][mask],z_ref[mask])
         
            cd = copy.copy(vector[:,1:])
            for g in range(vector[:,1:].shape[1]):
                cd[mask,g]=cd[mask,g]
            if boot:
                derr = covariance_jck(cd,100,'bootstrap')
            else:
                derr = covariance_jck(cd,100,'jackknife')
                
            ax[1,i].errorbar(z_ref[mask],vector[mask,0]/tht[i,mask]/normth ,derr['err'][mask]/tht[i,mask]/normth ,linestyle = linestye, label =lab,color = cl)
            minz = means-sigma*stdd
            maxz = means+sigma*stdd
            
            return minz,maxz,mute_dict1
        
        
        
        except:
            
            mask = (z_ref>zmin_norm[i]) & (z_ref<zmax_norm[i]) & (vector !=0.)
            norm = np.trapz(vector[mask],z_ref[mask])
  
            normth = np.trapz(tht[i][mask],z_ref[mask])
            if not normed:
                norm = 1.
                normth = 1.
            ax[0,i].plot(z_ref[mask],vector[mask]/norm,linestyle = linestye, label =lab,color = cl)
        

            means = compute_mean(z_ref[mask],vector[mask])
            
            stdd = compute_std(z_ref[mask],vector[mask])
            
            mask = mask & (z_ref>(means-sigma*stdd)) & (z_ref<(means+sigma*stdd))# & (vector !=0.)
            means = compute_mean(z_ref[mask],vector[mask]-compute_mean(z_ref[mask],tht[i][mask]))
            
            
            #print "interval: {0:2.2f} {1:2.2f}".format(means-sigma*stdd,means+sigma*stdd)
            ax[1,i].plot([0,0],[-1,1] ,linestyle = linestye, label =lab,color = cl)
        
        
            #ax[2,i].plot([1,1],[-1,1] ,linestyle = 'dashed', label =lab,color = cl)
        
            minz = means-sigma*stdd
            maxz = means+sigma*stdd
            return minz,maxz,{"mean":means}

def compute_bias(data,theory,i):
        return np.sqrt(np.array([data[i,kkk,:]/theory[i, kkk] for kkk in range(len(theory[0,:]))]))
    
def compute_th_correction(true_nz,theory,i):
        return np.array(theory[i, :]/true_nz[i,:])
    
    
class SmoothFunction:
    def __init__(self, mean=1., rms=[0.1,0.1,0.1,0.1,0.1],
                range=(-1.,1.), buffer=True):
        '''The smooth function will be a sum of Legendre polynomials.
        The function will have the specified (deterministic) mean, and
        then a sum of Legendre polynomials with coefficients each drawn
        from a Gaussian distribution.  The rms array specifies the RMS 
        variation in the function across the range that will be contributed
        by each order of Legendre.  The size of the rms array gives the
        max order of Legendre polynomial. 
          If buffer=True, the range and RMS will be adjusted internally to
        keep the excess-RMS regions near the Legendre endpoints out of your
        desired range'''
        self.mean = mean
        self.order = len(rms)
        # The sigmas that will be needed to give each order 
        # of Lagrange the desired RMS:
        self.sigma = np.sqrt((2*np.arange(self.order)+1)/2.)*np.array(rms)
        # Values to use to rescale input range to (-1,1)
        self.xmid = 0.5*(range[0]+range[1])
        self.xscale = 2. / (range[1]-range[0])
        if buffer:
            self.xscale *= 0.85
            self.sigma /= 0.76
        self.coefficients = np.zeros(self.order,dtype=float)
        # Next item is the log of Gaussian prefactor for probabilities
        self.logProbNorm = -0.5*self.order*np.log(2*np.pi) - np.sum(np.log(self.sigma))
        return
    def sample(self):
        # Sample a new set of coefficients 
        self.coefficients = np.random.normal(size=self.order)*self.sigma
        return
    def setCoefficients(self,coefficients):
        # Set coefficients to the given vector.
        # Returns the log(probability) of these coefficients
        cc = np.array(coefficients)
        if cc.size != self.sigma.size:
            raise ValueError("coefficients size does not match variance dimensions")
        self.coefficients = cc
        return self.logProbNorm - 0.5*np.sum((self.coefficients/self.sigma)**2)
    def __call__(self,x):
        # Return the value of the function at the array of values x
        # First remap the inputs to (-1,1)
        xx = (np.array(x)-self.xmid)*self.xscale
        # Create output array
        out = np.ones_like(xx) * self.mean
        # And two arrays to use in Legendre recurrence
        pm1 = np.ones_like(xx)  # P_0
        pm = np.array(xx)       # P_1
        if self.order>0:
            out += self.coefficients[0]*pm1
        if self.order>1:
            out += self.coefficients[1]*pm
        for m in range(2,self.order):
            pm, pm1 = ( ((2*m+1.)/(m+1))*xx*pm - (m/(m+1.))*pm1), pm
            out += self.coefficients[m]*pm
        return out
