import numpy as np
import matplotlib.pyplot as plt

import matplotlib.transforms as mtransforms
import matplotlib as mpl
import os
from os import path
import time
from numpy import linalg
import matplotlib.mlab as mlab
import math
import sys
import shutil
import copy
import timeit

from .dataset import save_obj, load_obj, update_progress



def scale_optimization(best_params,Nbins,interval_width,step_width,only_diagonal,optimization,N,resampling,resampling_pairs,jk_r):
  if resampling_pairs:
      resampling+='_pairs'
  else:
      resampling+='_'
  for i,tomo in enumerate(N.keys()):
    print ('\nTOMO_BIN {0}'.format(int(tomo)+1))


    best=dict()
    for method in best_params.keys():
        counter=0
        moment=dict()
        for nnn in range(len(Nbins)):
            if not optimization:
                interval_bins=Nbins[nnn]
                step=1
            else:
                interval_bins=int(max([math.ceil(Nbins[nnn]/2.),interval_width]))
                step=int(math.ceil(Nbins[nnn]/step_width))

            for thetmax in range(interval_bins,Nbins[nnn]+1,step):
                for thetmin in range(0,thetmax-interval_bins+1,step):
                    label_save='{0}_{1}_{2}_{3}_{4}_{5}'.format(method,thetmin,thetmax,Nbins[nnn],resampling,jk_r)

                    try:
                        statistics=load_obj(('./output_dndz/TOMO_{0}/Nz/statistics_{1}').format(int(tomo)+1,label_save))
                    except:
                        statistics=None


                    ustatistics={'stats':copy.deepcopy(statistics),
                            'thetmin':thetmin,
                            'thetmax':thetmax,
                            'Nbins':Nbins[nnn]}

                    moment.update({'{0}'.format(counter):ustatistics})
                    counter+=1
        best.update({'{0}'.format(method):moment})

    label_diag=''
    if only_diagonal:
        label_diag='_diag'


    for method in best.keys():

        #print (method)
        output_text=open('./output_dndz/TOMO_'+str(int(tomo)+1)+'/best_Nz/bestof_'+method,'w')
        a,b=best_methods(method,best[method],'stats',output_text,label_diag,tomo,resampling,jk_r)

        output_text.close()


def best_methods(method,best_params,stats,output_text,label_diag,tomo,resampling,jk_r):
    SN=[]
    methods_label=[]
    #print label_diag,stat

    for stat in best_params.keys():
        #print(method,best_params[stat][stats])
        if best_params[stat][stats] != None:
            if not np.isnan(best_params[stat][stats]['mean_rec']) and not np.isnan(best_params[stat][stats]['mean_rec_err{0}'.format(label_diag)]):
                SN.append(best_params[stat][stats]['mean_rec_err{0}'.format(label_diag)])
            else:
                SN.append(np.inf)
        else: SN.append(np.inf)
        methods_label.append(stat)


    if stats=='stats': opt=''
    #elif stats=='bb_stats': opt='[bias corrected]'
    else: opt='[bias and gaussian corrected]'

    SN=np.array(SN)
    indexu=np.argmin(SN)

    if not SN[indexu]==np.inf:

        print  ('best of {0} {1}: {2:.3f} +- {3:.3f} [{4},{5},{6}]').format(method,opt,best_params[methods_label[indexu]][stats]['mean_rec'],SN[indexu],best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins'])
        output_text.write (('best of {0} {1}: {2:.3f} +- {3:.3f} [{4},{5},{6}]').format(method,opt,best_params[methods_label[indexu]][stats]['mean_rec'],SN[indexu],best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']))
        output_text.write('\n')

        #print (stats)
        if stats=='stats':
            try:
                shutil.copy('./output_dndz/TOMO_'+str(int(tomo)+1)+'/Nz/{0}_{1}_{2}_{3}_{4}_{5}.pdf'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins'],resampling,jk_r), './output_dndz/TOMO_'+str(int(tomo)+1)+'/best_Nz/{0}_{1}_{2}.pdf'.format(method,resampling,jk_r))
            except:
                pass
            shutil.copy('./output_dndz/TOMO_'+str(int(tomo)+1)+'/Nz/{0}_{1}_{2}_{3}_{4}_{5}.h5'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins'],resampling,jk_r), './output_dndz/TOMO_'+str(int(tomo)+1)+'/best_Nz/{0}_{1}_{2}.h5'.format(method,resampling,jk_r))
            shutil.copy('./output_dndz/TOMO_'+str(int(tomo)+1)+'/Nz/statistics_{0}_{1}_{2}_{3}_{4}_{5}.pkl'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins'],resampling,jk_r), './output_dndz/TOMO_'+str(int(tomo)+1)+'/best_Nz/statistics_{0}_{1}_{2}.pkl'.format(method,resampling,jk_r))


        return best_params[methods_label[indexu]][stats]['mean_rec'],best_params[methods_label[indexu]][stats]['mean_rec_err{0}'.format(label_diag)]
    else:
        return np.inf,np.inf
