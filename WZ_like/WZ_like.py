from cosmosis.datablock import option_section, names
from scipy.stats import norm,multivariate_normal
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
import pdb
import pickle
import shutil
from scipy.interpolate import UnivariateSpline

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

def rebin(z_old, pdf_old, zbins):
        # spline
        kwargs_spline = {'s': 0,  # force spline to go through data points
                         'ext': 'zeros',  # ext=0 means extrapolate, =1 means return 0
                         'k': 3,
                        }
        spline = UnivariateSpline(z_old, pdf_old, **kwargs_spline)
        pdf = np.zeros(len(zbins) - 1)
        for i in range(len(zbins) - 1):
            zmin = zbins[i]
            zmax = zbins[i + 1]
            pdf[i] = spline.integral(zmin, zmax) #/ (zmax - zmin)
        return pdf
    
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
    
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f,encoding='latin1')
        f.close()
    return mute


from scipy import arange, array, exp

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)
    def ufunclike(xs):
        return list(map(pointwise, array(xs)))
    return ufunclike


def setup(options):
    # load clustering-z quantities

    file_to_load = options.get_string(option_section, "measured_wz", default = "None")
    wz_data  = load_obj(file_to_load)

    bb = options.get_double_array_1d(option_section, "bins_to_be_included")
    wz_data['bins_to_be_included']  = bb.astype(int)
    
    # read label of wl and rmg samples from cosmosis
    
    sample_wl = options.get_string(option_section, "sample_wl", "")
    if sample_wl == "":
        pz = names.wl_number_density
    else:
        pz = sample_wl
        
    sample_lens = options.get_string(option_section, "sample_lens", "")
    if sample_lens == "":
        rmg = names.wl_number_density
    else:
        rmg = sample_lens
        
        
    wz_data['mean_rm'] = options.get_bool(option_section, "mean_rm", True)
    wz_data['mean_eboss'] = options.get_bool(option_section, "mean_eboss", False)
    wz_data['std_rm'] = options.get_bool(option_section, "std_rm", False)
    wz_data['magnif'] = options.get_bool(option_section, "magnif", False)
    wz_data['full_shape'] = options.get_bool(option_section, "full_shape", False)

    print ('*****WZ CONFIG ******')
    print ('mean_rm ',wz_data['mean_rm'] )
    print ('mean_eboss ',wz_data['mean_eboss'])
    print ('std_rm ',wz_data['std_rm'] )
    print ('magnif ',wz_data['magnif'] )
    
    wz_data['rmg'] = rmg
    wz_data['pz'] = pz
    config_data = wz_data
    return config_data





def execute(block, config):
    magnif = config['magnif']
    mean_rm =config['mean_rm']
    mean_eboss =config['mean_eboss']
    std_rm =config['std_rm']
    
    rmg = config['rmg']
    pz = config['pz']  
    # compute mean redshift lens sample ************
    z_rmg = block[rmg, "z"]
    nbin_rmg = block[rmg, "nbin"]
    
    z_rmg_main_analysis = []
    for i in range(1, nbin_rmg + 1):
        bin_name = "bin_%d" % i
        nz_rmg = block[rmg, bin_name]
        z_rmg_main_analysis.append(compute_mean(z_rmg,nz_rmg))
    config['z_rmg_main_analysis'] = np.array(z_rmg_main_analysis)
    config['rmg_label'] = rmg
    # load rmg nz and compute mean z
    nbin = block[pz, "nbin"]
    z = block[pz, "z"]
    
    like_tot = []
    for i in range(1, nbin + 1):
        bin_name = "bin_%d" % i
        nz = block[pz, bin_name]

    config['wl_nz'] = nz
    config['wl_label'] = pz

    
    
    mag_pos = config['mag_pos'] 
    mag_pos1 = config['mag_pos1'] 
    Nz_rm = config['Nz_rm'] 
    Nz_eboss = config['Nz_eboss']
    syst_mean_rm = config['syst_mean_rm'] 
    syst_std_rm = config['syst_std_rm']
    alpha_mag_rm = config['alpha_mag_rm']
    bincenters_rm = config['bincenters_rm']
    bincenters_eboss = config['bincenters_eboss']
    
    #you need to read these somehow
    alpha1 = block["mag_alpha_lens", "alpha_1"]
    alpha2 = block["mag_alpha_lens", "alpha_2"]
    alpha3 = block["mag_alpha_lens", "alpha_3"]
    alpha4 = block["mag_alpha_lens", "alpha_4"]
    alpha5 = block["mag_alpha_lens", "alpha_5"]


    
    alph = np.array([alpha1,alpha2,alpha3,alpha4,alpha5])
     
    
    #interpolate over some redshift array (mean of the redmagic bins)

    f0 = interp1d(config['z_rmg_main_analysis'],alph)
    f = extrap1d(f0)
    
    #calcolarli per il redshift del reference sample

    alpha_mag_rm = np.array(f( bincenters_rm))
    
    print ('magnifi',alpha_mag_rm)
    
    bias_rm = config['bias_rm'] 
    bias_eboss = config['bias_eboss'] 
    bincenters_rm = config['bincenters_rm']
    bincenters_eboss = config['bincenters_eboss']
    
    th_correction =  config['th_correction']
    th_correction_eboss =  config['th_correction_eboss']
    nbin = block[config['wl_label'], "nbin"]
    z = block[config['wl_label'], "z"]
    
    bin_edges_rm = bincenters_rm[0]-(bincenters_rm[1]-bincenters_rm[0])*0.5
    bin_edges_rm = np.append(bin_edges_rm, bincenters_rm+(bincenters_rm[1]-bincenters_rm[0])*0.5)
    
    bin_edges_eboss = bincenters_eboss[0]-(bincenters_eboss[1]-bincenters_eboss[0])*0.5
    bin_edges_eboss = np.append(bin_edges_eboss, bincenters_eboss+(bincenters_eboss[1]-bincenters_eboss[0])*0.5)
    
    like_tot = 0.
    for i in range(1, nbin + 1):
        if i in config['bins_to_be_included']:
            bin_name = "bin_%d" % i
            nz = block[pz, bin_name]
            
            # rebin *********************
            nz_rebin = rebin(z, nz, bin_edges_rm)
            try:
                nz_rebin_eboss = rebin(z, nz, bin_edges_eboss)
            except:
                pass
            
            # 2 sigma mask ******************
            mask_sigma = (bincenters_rm > (compute_mean(bincenters_rm,nz_rebin) -2.*compute_std(bincenters_rm,nz_rebin))) & (bincenters_rm < (compute_mean(bincenters_rm,nz_rebin) + 2.*compute_std(bincenters_rm,nz_rebin)))
            try:
                mask_sigma_eboss = (bincenters_eboss > (compute_mean(bincenters_eboss,nz_rebin_eboss) -2.*compute_std(bincenters_eboss,nz_rebin_eboss))) & (bincenters_eboss < (compute_mean(bincenters_eboss,nz_rebin_eboss) + 2.*compute_std(bincenters_eboss,nz_rebin_eboss)))
            except:
                pass
            
            
            # compute NZ ************************
            
            bbb = block['bias_wl', 'bin_%d' % i]
            aaa = block['mag_alpha_wl', 'alpha_%d' % i]

            
            if(magnif == True): 
                theory_Nz = np.array([np.array(Nz_rm[i-1,:,jk]-bbb*(alpha_mag_rm-2.)*mag_pos1[i-1,:]-(aaa-2.)*bias_rm[i-1,:,jk]*mag_pos[i-1,:])/(bias_rm[i-1,:,jk]*th_correction) for jk in range((Nz_rm.shape[2]))]).T
                
            else:
                theory_Nz = np.array([np.array(Nz_rm[i-1,:,jk])/(bias_rm[i-1,:,jk]*th_correction) for jk in range((Nz_rm.shape[2]))]).T
                
            try:
                theory_Nz_eboss = np.array([Nz_eboss[i-1,:,jk]/(bias_eboss[i-1,:,jk]*th_correction_eboss) for jk in range((Nz_eboss.shape[2]))]).T
                
            except:
                pass
            # *******************
            
            if (( mean_rm == True) and (mean_eboss == True)):
                mean_true_z = compute_mean(bincenters_rm[mask_sigma], nz_rebin[mask_sigma] )
                mean_clustering_z_rm = compute_mean(bincenters_rm[mask_sigma], theory_Nz[mask_sigma,0])**2
                mean_true_z_eboss = compute_mean(bincenters_eboss[mask_sigma_eboss], nz_rebin_eboss[mask_sigma_eboss])
                mean_clustering_z_eboss = compute_mean(bincenters_eboss[mask_sigma_eboss], theory_Nz_eboss[mask_sigma_eboss,0])

                y = np.array([mean_true_z- mean_clustering_z_rm,mean_true_z_eboss-mean_clustering_z_eboss])
                ccov = config['cross_cov'][i-1]
                like_tot+=-0.5*np.sum(np.matmul(y,np.matmul(np.linalg.inv(ccov),y)))
                
                
            else:
                
                if( mean_rm == True):

                    mean_true_z = compute_mean(bincenters_rm[mask_sigma], nz_rebin[mask_sigma] )

                    mean_clustering_z_rm = compute_mean(bincenters_rm[mask_sigma], theory_Nz[mask_sigma,0])**2

                    print ('mean_true_z ', mean_true_z)
                    print ('mean_clustering_z_rm', mean_clustering_z_rm)

                    like_tot+=like_mean_rm

                if(mean_eboss == True):

                    mean_true_z_eboss = compute_mean(bincenters_eboss[mask_sigma_eboss], nz_rebin_eboss[mask_sigma_eboss])
                    #std_true_z_eboss = compute_std(bincenters_eboss[mask_sigma_eboss], nz_rebin_eboss[mask_sigma_eboss])

                    mean_clustering_z_eboss = compute_mean(bincenters_eboss[mask_sigma_eboss], theory_Nz_eboss[mask_sigma_eboss,0])


                    #std_clustering_z_eboss = compute_std(bincenters_eboss, theory_Nz_eboss)
                    like_mean_eboss  = -0.5*((mean_true_z_eboss-mean_clustering_z_eboss)/config['syst_mean_eboss'][i-1])**2
                    print ('mean_true_z_eboss ', mean_true_z_eboss)
                    print ('mean_clustering_z_eboss', mean_clustering_z_eboss)

                    like_tot+=like_mean_eboss
                
            if (std_rm == True):
                
                std_true_z = compute_std(bincenters_rm, nz_rebin )
                std_clustering_z_rm = compute_std(bincenters_rm, theory_Nz[:,0])
                like_std_rm =  -0.5*((std_true_z-std_clustering_z_rm)/config['syst_std_rm'][i-1])**2
                
                print ('std_true_z_rm ', std_true_z)
                print ('std_clustering_z_rm', std_clustering_z_rm)
            
                like_tot+= like_std_rm
                
                
            # Gary's likelihood.
            if config['full_shape']:
                
                # compute covariance ***********
                try:
                    mute = covariance_jck(theory_Nz_eboss[:,1:],theory_Nz_eboss[:,1:].shape[1],'jackknife')
                    err_eboss = mute['err']
                except:
                    pass
                mute = covariance_jck(theory_Nz[:,1:],theory_Nz[:,1:].shape[1],'jackknife')
                err_rm = mute['err']   
                
                
                # multiply for the prior1
                
                params1=np.array([1.,block["wz_nuisance_parameters", "c{0}_2".format(i)],block["wz_nuisance_parameters", "c{0}_2".format(i)]])
                poly1 = np.polynomial.Polynomial(params1)
                sfx = poly1(config['shape_mode']['err'][i-1]['xn'])
                
                # normalisation in the 2sigma interval
                
                
                
                normy = np.sum((nz_rebin*sfx)[mask_sigma])
                normwz = np.sum(theory_Nz[mask_sigma,0])
                y =(nz_rebin*sfx)/normy-theory_Nz[:,0]/normwz
                

                like_tot+=-0.5*np.sum(((y )/(err_rm/normwz) )**2)
                
                #-poly1(xn)[mean_i]+1
            #ax[2,i].plot(x,yy3, linestyle = 'dashed',color='grey',alpha=0.3)
            
        
    block[names.likelihoods, 'wz_like'] = like_tot


    return 0

def cleanup(config):
    pass


