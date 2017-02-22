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



def scale_optimization(best_params,Nbins,interval_width,step_width,only_diagonal,optimization):
    counter=0
    for method in best_params.keys():
        #print (method)
        for nnn in range(len(Nbins)):
            if not optimization:
                interval_bins=Nbins[nnn]
                step=1
            else:
                interval_bins=int(max([math.ceil(Nbins[nnn]/2.),interval_width]))
                step=int(math.ceil(Nbins[nnn]/step_width))

            for thetmax in range(interval_bins,Nbins[nnn]+1,step):
                for thetmin in range(0,thetmax-interval_bins+1,step):
                    label_save='Nz_{0}_{1}_{2}_{3}'.format(method,thetmin,thetmax,Nbins[nnn])
                    label_save1='BNz_{0}_{1}_{2}_{3}'.format(method,thetmin,thetmax,Nbins[nnn])

                    try: statistics=load_obj(('./output_dndz/Nz/statistics_{0}').format(label_save))
                    except: statistics=None


                    try: BBstatistics=load_obj(('./output_dndz/Nz/statistics_{0}').format(label_save1))
                    except: BBstatistics=None


                    ustatistics={'stats':copy.deepcopy(statistics),
                            'bb_stats':copy.deepcopy(BBstatistics),
                            'thetmin':thetmin,
                            'thetmax':thetmax,
                            'Nbins':Nbins[nnn]}

                    best_params[method].update({'{0}'.format(counter):ustatistics})
                    counter+=1

    best_mean=[]
    best_mean_err=[]
    method_list=[]

    bbbest_mean=[]
    bbbest_mean_err=[]
    bbmethod_list=[]

    y_axis_final=[]
    bby_axis_final=[]

    counter_y=0
    counter_y+=1
    label_diag=''
    if only_diagonal:
        label_diag='_diag'


    for method in best_params.keys():

        SN=[]
        BBSN=[]
        methods_label=[]

    # N_Z BEST

        output_text=open('./output_dndz/best_Nz/bestof_'+method,'w')
        a,b=best_methods(method,best_params[method],'stats',output_text,label_diag)
        if a != np.inf:
            best_mean.append(a)
            best_mean_err.append(b)
            method_list.append(method)
            y_axis_final.append(counter_y)



        a,b=best_methods(method,best_params[method],'bb_stats',output_text,label_diag)
        if a != np.inf:
            bbbest_mean.append(a)
            bbbest_mean_err.append(b)
            bbmethod_list.append(method)
            bby_axis_final.append(counter_y)

        counter_y+=2


        output_text.close()




    ####################################################################################################
    #                                   PLOTTING
    ####################################################################################################


    '''
    THIS CAN BE IMPROVED, AND PUT AS INDEPENDENT MODUL AFTER HAVING APPLIED ALSO THE GAUSSIAN PROCESSES.
    '''
    counter_y+=2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def add_line_vert(ax, xpos, bum):
        print xpos
        plt.plot([xpos, xpos], [0, 2*bum],
                       color='black',linestyle='dashed',label='<Z>_TRUE=%.2f'%  xpos)

    bum=len(best_mean)
    #for i in N_true.keys():
    #   add_line_vert(ax, N_true[i], bum)

    #plt.axvspan(N_true[i]-0.02, N_true[i]+0.02, alpha=0.3, color='grey')


    fig.subplots_adjust(left=.1*2.8)

    def add_line(ax, xpos, ypos):
        line = plt.Line2D([-0.5, 1], [ypos, ypos],
                      transform=ax.transAxes, color='black',linestyle='dashed')
        line.set_clip_on(False)
        ax.add_line(line)



    def add_line1(ax, xpos, ypos):
        line = plt.Line2D([-0.5, 1.2], [ypos, ypos],
                      transform=ax.transAxes, color='black',linestyle='dashed')
        line.set_clip_on(False)
        ax.add_line(line)




    plt.title('compare methods')




    plt.errorbar(np.array(best_mean),y_axis_final,xerr=np.array(best_mean_err), markersize=5,fmt='o',color='b',label='no bias correction')
    plt.errorbar(np.array(bbbest_mean),np.array(bby_axis_final)+0.1,xerr=np.array(bbbest_mean_err), markersize=5,alpha=1.,fmt='*',color='green',label='bias correction')
    #plt.errorbar(np.array(bgbest_mean),np.array(bgy_axis_final)+0.2,xerr=np.array(bgbest_mean_err),markersize= 5, alpha=1.,fmt='*',color='yellow',label='bias & gaussian correction')
    #plt.errorbar(np.array(best_mean[len(N_true.keys()):-1]),y_axis_final[len(N_true.keys()):-1],xerr=np.array(best_mean_err[len(N_true.keys()):-1]), fmt='o',color='red',label='BPZ')

    #ax.set_xlim([0.25,0.35])

    #plt.yticks(y_axis_final, y_values_final,fontsize=10)
    #plt.yticks(y_axis_final1, y_values_final1,fontsize=10)
    #plt.yticks(y_axis_final2, y_values_final2,fontsize=10)

    plt.yticks(y_axis_final, method_list,fontsize=15)
    #plt.yticks(y_axis_final, y_values_final,fontsize=10)
    plt.margins(0.02)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True,prop={'size':15})

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)



    plt.savefig('./output_dndz/best_Nz/compare_methods.pdf', format='pdf')


    plt.close()



def best_methods(method,best_params,stats,output_text,label_diag):
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
    elif stats=='bb_stats': opt='[bias corrected]'
    else: opt='[bias and gaussian corrected]'

    SN=np.array(SN)
    indexu=np.argmin(SN)

    if not SN[indexu]==np.inf:

        print  ('best of {0} {1}: {2:.3f} +- {3:.3f} [{4},{5},{6}]').format(method,opt,best_params[methods_label[indexu]][stats]['mean_rec'],SN[indexu],best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins'])
        output_text.write (('best of {0} {1}: {2:.3f} +- {3:.3f} [{4},{5},{6}]').format(method,opt,best_params[methods_label[indexu]][stats]['mean_rec'],SN[indexu],best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']))
        output_text.write('\n')

        #print (stats)
        if stats=='stats':
            shutil.copy('./output_dndz/Nz/Nz_{0}_{1}_{2}_{3}.pdf'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/NZ_{0}.pdf'.format(method))
            shutil.copy('./output_dndz/Nz/Nz_{0}_{1}_{2}_{3}.h5'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/NZ_{0}.h5'.format(method))
            shutil.copy('./output_dndz/Nz/statistics_Nz_{0}_{1}_{2}_{3}.pkl'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/statistics_NZ_{0}.pkl'.format(method))

        elif stats=='bb_stats':
            shutil.copy('./output_dndz/Nz/BNz_{0}_{1}_{2}_{3}.pdf'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/BNZ_{0}.pdf'.format(method))
            shutil.copy('./output_dndz/Nz/BNz_{0}_{1}_{2}_{3}.h5'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/BNZ_{0}.h5'.format(method))
            shutil.copy('./output_dndz/Nz/statistics_BNz_{0}_{1}_{2}_{3}.pkl'.format(method,best_params[methods_label[indexu]]['thetmin'],best_params[methods_label[indexu]]['thetmax'],best_params[methods_label[indexu]]['Nbins']), './output_dndz/best_Nz/statistics_BNZ_{0}.pkl'.format(method))


        return best_params[methods_label[indexu]][stats]['mean_rec'],best_params[methods_label[indexu]][stats]['mean_rec_err{0}'.format(label_diag)]
    else:
        return np.inf,np.inf






def make_plot_compare_all(only_diagonal):
    counter=1
    method=[]
    mean=[]
    mean_err=[]
    bbmean=[]
    bbmean_err=[]
    ggmean=[]
    ggmean_err=[]
    ggbbmean=[]
    ggbbmean_err=[]
    y=[]
    bby=[]
    ggy=[]
    ggbby=[]

    label_diag=''
    if only_diagonal:
        label_diag='_diag'
    for file in os.listdir('./output_dndz/best_Nz/'):
        if file.endswith(".h5") and ('gaussian' not in file) and ('BNZ' not in file):

            method_label1=(file.replace(".h5", ""))

            statistics=load_obj('./output_dndz/best_Nz/statistics_{0}'.format(method_label1))
            method_label=method_label1.replace("NZ_", "")

            method.append(method_label)

            mean.append(statistics['mean_rec'])
            mean_err.append(statistics['mean_rec_err'+label_diag])
            y.append(counter)

            if path.exists('./output_dndz/best_Nz/statistics_B{0}.pkl'.format(method_label1)):
                statistics=load_obj('./output_dndz/best_Nz/statistics_B{0}'.format(method_label1))
                bbmean.append(statistics['mean_rec'])
                bbmean_err.append(statistics['mean_rec_err'+label_diag])
                bby.append(counter)
            if path.exists('./output_dndz/best_Nz/statistics_gauss_{0}.pkl'.format(method_label1)):
                statistics=load_obj('./output_dndz/best_Nz/statistics_gauss_{0}'.format(method_label1))
                ggmean.append(statistics['mean_rec'])
                ggmean_err.append(statistics['mean_rec_err'+label_diag])
                ggy.append(counter)
            if path.exists('./output_dndz/best_Nz/statistics_gauss_B{0}.pkl'.format(method_label1)):
                statistics=load_obj('./output_dndz/best_Nz/statistics_gauss_B{0}'.format(method_label1))
                ggbbmean.append(statistics['mean_rec'])
                ggbbmean_err.append(statistics['mean_rec_err'+label_diag])
                ggbby.append(counter)

            counter+=2



    ##############################################################

    ##############################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def add_line_vert(ax, xpos, bum):
            #print xpos
            plt.plot([xpos, xpos], [0, 2*bum],
                           color='black',linestyle='dashed',label='<Z>_TRUE=%.2f'%  xpos)

    fig.subplots_adjust(left=.1*2.8)

    def add_line(ax, xpos, ypos):
            line = plt.Line2D([-0.5, 1], [ypos, ypos],
                          transform=ax.transAxes, color='black',linestyle='dashed')
            line.set_clip_on(False)
            ax.add_line(line)



    def add_line1(ax, xpos, ypos):
            line = plt.Line2D([-0.5, 1.2], [ypos, ypos],
                          transform=ax.transAxes, color='black',linestyle='dashed')
            line.set_clip_on(False)
            ax.add_line(line)




    plt.title('compare methods')




    plt.errorbar(np.array(mean),np.array(y), xerr=np.array(mean_err), markersize=5,fmt='o',color='b',label='no bias correction')
    plt.errorbar(np.array(bbmean),np.array(bby)+0.2,xerr=np.array(bbmean_err), markersize=5,alpha=1.,fmt='*',color='green',label='bias correction')
    plt.errorbar(np.array(ggmean),np.array(ggy)+0.1,xerr=np.array(ggmean_err),markersize= 5, alpha=1.,fmt='*',color='yellow',label='gaussian correction')
    plt.errorbar(np.array(ggbbmean),np.array(ggbby)+0.3,xerr=np.array(ggbbmean_err),markersize= 5, alpha=1.,fmt='*',color='red',label='bias &gaussian correction')
    #plt.errorbar(np.array(best_mean[len(N_true.keys()):-1]),y_axis_final[len(N_true.keys()):-1],xerr=np.array(best_mean_err[len(N_true.keys()):-1]), fmt='o',color='red',label='BPZ')

        #ax.set_xlim([0.25,0.35])

        #plt.yticks(y_axis_final, y_values_final,fontsize=10)
        #plt.yticks(y_axis_final1, y_values_final1,fontsize=10)
        #plt.yticks(y_axis_final2, y_values_final2,fontsize=10)


    plt.yticks(y, method,fontsize=7)

        #plt.yticks(y_axis_final, y_values_final,fontsize=10)
    plt.margins(0.02)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True,prop={'size':15})
    #plt.legend()
    #plt.legend(loc=2,prop={'size':10},fancybox=True)
        # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)



    plt.savefig('./output_dndz/best_Nz/compare_methods_ALL.pdf', format='pdf')


    plt.close()
