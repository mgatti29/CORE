"""
Create datasets given fits files

.. module:: dataset
"""

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import healpy as hp
import fitsio
import pickle
import gc
import time
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import pyfits as pf
from scipy import spatial
from scipy.spatial import distance
import sys
import copy
import os
from scipy.interpolate import interp1d
import scipy.integrate as pyint

def dataset(reference_data, reference_random,
            unknown_data, unknown_random,
            unknown_bins,reference_bins,
            max_objects=30000000,
            label_every=10000,
            randoms_time=0.,
            kind_regions='kmeans',dontsaveplot=False,
            #healpix_nside=32,
            #healpix_filter_size=20,
            #healpix_filter_start=-1,
            number_of_regions=100,
            load_regions='None',
            time0=0,**kwargs):

    """Load up datasets into understandable form

    Parameters
    ----------
    [reference,unknown]_[data,random]_path : string, locations to fits files of
                                             catalogs. They are assumed to have
                                             the entries in reference_columns
                                             or unknown_columns in them.
                                             IF you pass a list of 2 items,
                                             then interpret as hdf5 file, with
                                             entry 1 being file path, entry 2
                                             the key of the dataset.
    [reference,unknown]_[ra,dec,w]_column : string, location of said columns.
                                            w means 'weight' and does NOT have
                                            to be present. These will be
                                            renamed in my dataset

    max_objects : int [default: 30,000,000] Maximum number of objects from
                  catalog to load. Useful if memory issues ariase.
    label_every : int [default: 10,000] Maximum number of labels calculated and
                  assigned when augmenting catalog with `region' values which
                  will be later used in the paircounts. Again, useful if memory
                  issues arise.
    healpix_nside : int [default: 32] nside we divide the paircounts into for
                    later paircounting. This therefore also controls the number
                    of files saved, and thus the harddisk space of the output


    Returns
    -------
    reference_catalogs, unknown_catalogs : list of pandas dataframes of
                                           reference and unknown catalogs

    Notes
    -----
    Cannot handle loading up any rows that are vectors!
    """
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    #progress bar
    update_progress(0.)

    #   redshift arrays - edges and central values. **************************************
    reference_bins_interval,additional_column_ref = interpret_bins(reference_bins)
    unknown_bins_interval,additional_column_unknown   = interpret_bins(unknown_bins)

    #  save the redshift arrays
    save_obj('./pairscount/unknown_bins_interval',unknown_bins_interval)
    save_obj('./pairscount/reference_bins_interval',reference_bins_interval)

    # Loading reference catalog    ********************************************************
    catalog_ref=load_catalogs(reference_data,max_objects,additional_column_ref)
    catalog_ref=check_columns(reference_data['columns'],catalog_ref,additional_column_ref)
    update_progress(0.1)


    # Loading reference randoms catalog ***************************************************
    catalog_ref_rndm=load_catalogs(reference_random,max_objects)
    catalog_ref_rndm=check_columns(reference_random['columns'],catalog_ref_rndm)


    # cut the randoms  of the reference if too numerous
    """
    I want to use the same randoms for each reference.
    """
    '''
    mask=catalog_ref_rndm['RA']<100
    catalog_ref_rndm=catalog_ref_rndm[mask]
    mask=catalog_ref['RA']<100
    catalog_ref=catalog_ref[mask]

    mask=catalog_ref_rndm['DEC']>-60.
    catalog_ref_rndm=catalog_ref_rndm[mask]
    mask=catalog_ref['DEC']>-60.
    catalog_ref=catalog_ref[mask]
    '''
    if randoms_time>0:
        if catalog_ref_rndm['RA'].shape[0]>randoms_time*(catalog_ref['RA'].shape[0]):
            catalog_ref_rndm=catalog_ref_rndm.sample(np.int(randoms_time*(catalog_ref['RA'].shape[0])))


    # Define jackknives over the reference randoms and saving them
    if kind_regions=='kmeans':
        if load_regions!='None':
            centers=np.array(np.loadtxt(load_regions))
            np.savetxt('./pairscount/pairscounts_centers.txt', centers)
            centers_tree=spatial.cKDTree(centers)
        else:
            new_cat=np.array(zip(catalog_ref_rndm['RA'],catalog_ref_rndm['DEC']))
            A=new_cat[np.random.randint(new_cat.shape[0],size=20000),:]
            centers_jck= kmeans_radec.kmeans_sample(A,number_of_regions,maxiter=100,tol=1e-05,verbose=0)

            #saving the senters
            np.savetxt('./pairscount/pairscounts_centers.txt', centers_jck.centers)
            centers_tree=spatial.cKDTree(centers_jck.centers[:,[0,1]])

        update_progress(0.3)
        # Assign jackknives to galaxies
        _,catalog_ref_rndm['HPIX']= centers_tree.query(np.array(zip(catalog_ref_rndm['RA'],catalog_ref_rndm['DEC'])))
        _,catalog_ref['HPIX']= centers_tree.query(np.array(zip(catalog_ref['RA'],catalog_ref['DEC'])))

    elif kind_regions=='healpix':
        pd.options.mode.chained_assignment = None
        catalog_ref_rndm['HPIX'] = radec_to_index(catalog_ref_rndm['DEC'], catalog_ref_rndm['RA'],number_of_regions)
        catalog_ref['HPIX'] = radec_to_index(catalog_ref['DEC'], catalog_ref['RA'],number_of_regions)

        #saving the centers
        unique=np.unique(radec_to_index(catalog_ref_rndm['DEC'], catalog_ref_rndm['RA'],number_of_regions))
        healpix_to_int=(zip(unique,range(1,unique.shape[0])))
        center_ra,center_dec=hp.pix2ang(number_of_regions, unique,nest=False, lonlat=True)
        np.savetxt('./pairscount/pairscounts_centers.txt', zip(center_ra,center_dec))

        #converting from healpix to integer
        for kk_int,kk in enumerate(unique):
            catalog_ref_rndm['HPIX'][catalog_ref_rndm['HPIX']==kk]=kk_int
            catalog_ref['HPIX'][catalog_ref['HPIX']==kk]=kk_int


    update_progress(0.4)

    # remove jackknife_distance_file from previous run for the given number of jackknife

    if  os.path.exists('./pairscount/pairs_dist/'+str(number_of_regions)+'.pkl'):
        os.remove('./pairscount/pairs_dist/'+str(number_of_regions)+'.pkl')

    # create the redshift for the randoms if not provided
    if 'Z' not in catalog_ref_rndm.keys():
        catalog_ref_rndm['Z']=make_z_distribution(catalog_ref['binning_column'],reference_bins_interval['z_edges'],catalog_ref_rndm['HPIX'].shape[0])
        #catalog_ref_rndm['Z']=np.ones(catalog_ref_rndm['HPIX'].shape[0])*0.7         #  make_z_distribution_full(catalog_ref['binning_column'],catalog_ref_rndm['HPIX'].shape[0])

    # binning of
    catalog_ref_rndm['bins']=digitize(catalog_ref_rndm['Z'],  reference_bins_interval)
    catalog_ref['bins']=digitize(catalog_ref['binning_column'], reference_bins_interval)

    if not dontsaveplot:
        plot_redshift_distr(catalog_ref_rndm['Z'],reference_bins_interval['z_edges'],'./pairscount/data_plot/ref_rndm_z.pdf')
        plot_redshift_distr(catalog_ref['binning_column'],reference_bins_interval['z_edges'],'./pairscount/data_plot/ref_z.pdf')



    # saving catalogs
    catalog_ref.to_hdf('./pairscount/dataset.h5', 'ref')
    catalog_ref_rndm.to_hdf('./pairscount/dataset.h5', 'ref_random')
    if not dontsaveplot:
        plot_data(catalog_ref,'ref',np.unique(catalog_ref_rndm['bins']))
        plot_data(catalog_ref_rndm,'ref_rndm',np.unique(catalog_ref_rndm['bins']))
    update_progress(0.5)
    del catalog_ref
    del catalog_ref_rndm
    gc.collect()


    # Loading unknown  catalog    ********************************************************
    catalog_unk=load_catalogs(unknown_data,max_objects,additional_column_unknown)
    catalog_unk=check_columns(unknown_data['columns'],catalog_unk,additional_column_unknown)
    update_progress(0.6)

    # Loading unknown random catalog    ********************************************************
    catalog_unk_rndm=load_catalogs(unknown_random,max_objects)
    catalog_unk_rndm=check_columns(unknown_random['columns'],catalog_unk_rndm)
    update_progress(0.7)
    '''
    mask=catalog_unk_rndm['RA']<100
    catalog_unk_rndm=catalog_unk_rndm[mask]
    mask=catalog_unk['RA']<100
    catalog_unk=catalog_unk[mask]

    mask=catalog_unk_rndm['DEC']>-60.
    catalog_unk_rndm=catalog_unk_rndm[mask]
    mask=catalog_unk['DEC']>-60.
    catalog_unk=catalog_unk[mask]
    '''
    # cut the randoms if too numerous
    if randoms_time>0:
        if catalog_unk_rndm['RA'].shape[0]>randoms_time*(catalog_unk['RA'].shape[0]):
            catalog_unk_rndm=catalog_unk_rndm.sample(np.int(randoms_time*(catalog_unk['RA'].shape[0])))

    # Assign jackknives
    if  kind_regions=='kmeans':
        _,catalog_unk_rndm['HPIX']= centers_tree.query(np.array(zip(catalog_unk_rndm['RA'],catalog_unk_rndm['DEC'])))
        _,catalog_unk['HPIX']= centers_tree.query(np.array(zip(catalog_unk['RA'],catalog_unk['DEC'])))
        update_progress(0.8)
    elif  kind_regions=='healpix':
        catalog_unk_rndm['HPIX'] = radec_to_index(catalog_unk_rndm['DEC'], catalog_unk_rndm['RA'],number_of_regions)
        catalog_unk['HPIX'] = radec_to_index(catalog_unk['DEC'], catalog_unk['RA'],number_of_regions)

        #converting from healpix to integer
        for kk_int,kk in enumerate(unique):
            catalog_unk_rndm['HPIX'][catalog_unk_rndm['HPIX']==kk]=kk_int
            catalog_unk['HPIX'][catalog_unk['HPIX']==kk]=kk_int3

    if 'Z' not in catalog_unk_rndm.keys():
        catalog_unk_rndm['Z']=make_z_distribution(catalog_unk['binning_column'],unknown_bins_interval['z_edges'],catalog_unk_rndm['HPIX'].shape[0])
    catalog_unk_rndm['Z_auto_']=make_z_distribution(catalog_unk[unknown_data['columns']['z_photo_columns'][0]],reference_bins_interval['z_edges'],catalog_unk_rndm['HPIX'].shape[0])

    # binning

    catalog_unk_rndm['bins']=digitize(catalog_unk_rndm['Z'], unknown_bins_interval)
    catalog_unk_rndm['bins_auto_']=digitize(catalog_unk_rndm['Z_auto_'], reference_bins_interval)

    catalog_unk['bins']=digitize(catalog_unk['binning_column'], unknown_bins_interval)
    catalog_unk['bins_auto_']=digitize(catalog_unk[unknown_data['columns']['z_photo_columns'][0]], reference_bins_interval)

    for i,mute in enumerate(unknown_bins_interval['z']):
        if not dontsaveplot:
            plot_redshift_distr(catalog_unk_rndm['Z'][catalog_unk_rndm['bins']==i+1],reference_bins_interval['z_edges'],'./pairscount/data_plot/unk_rndm_z_{0}.pdf'.format(i+1))
        mask_plot=(catalog_unk['binning_column']>unknown_bins_interval['z_edges'][i])& (catalog_unk['binning_column']<unknown_bins_interval['z_edges'][i+1])
        if not dontsaveplot:
            plot_redshift_distr(catalog_unk['binning_column'][mask_plot],reference_bins_interval['z_edges'],'./pairscount/data_plot/unk_z_{0}.pdf'.format(i+1))


    update_progress(0.9)

    # saving catalogs
    catalog_unk.to_hdf('./pairscount/dataset.h5', 'unk')
    catalog_unk_rndm.to_hdf('./pairscount/dataset.h5', 'unk_random')

    if not dontsaveplot:
        plot_data(catalog_unk,'unk',np.unique(catalog_unk_rndm['bins']))
        plot_data(catalog_unk_rndm,'unk_rndm',np.unique(catalog_unk_rndm['bins']))

    update_progress(1.)

    del catalog_unk
    del catalog_unk_rndm
    gc.collect()




# Routines for healpix *****************************************
def hpix_filter(hpix, healpix_filter_start, healpix_filter_size):
    hpix_unique = np.unique(hpix)  # also sorts
    hpix_min = healpix_filter_start * healpix_filter_size
    hpix_max = min([(healpix_filter_start + 1) * healpix_filter_size, len(hpix_unique) - 1])
    conds = (hpix >= hpix_unique[hpix_min]) & (hpix < hpix_unique[hpix_max])
    return conds

def index_to_radec(index, nside):
    theta, phi = hp.pixelfunc.pix2ang(nside, index)

    # dec, ra
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)

def radec_to_index(dec, ra, nside):
    return hp.pixelfunc.ang2pix(nside, np.radians(-dec + 90.), np.radians(ra))



#  Routines to load files *****************************************
def load_catalogs(path,max_objects,additional_column=None):

        #eliminate the None columns
        columns_to_be_read=[]
        for column in path['columns'].keys():
            if column == 'z_photo_columns':
                for z_photo in path['columns'][column]:
                    columns_to_be_read.append('{0}'.format(z_photo))
            else:
                if path['columns'][column]!= 'None':
                    columns_to_be_read.append(path['columns'][column])
        if additional_column:
            columns_to_be_read.append(additional_column)
        columns_to_be_read=np.unique(columns_to_be_read)
        # open file ******************



        # default option fits format
        if 'file_format' not in path.keys():
            file_format_toberead='fits'
        else:
            file_format_toberead=path['file_format']

        #read the correct format
        if file_format_toberead=='fits':
            catalog=load_fits_file(path['path'], columns_to_be_read, additional_column, max_objects)
            #except: print ('Error in reading the fits file. Path and columns need to be provided')

        elif file_format_toberead=='hdf5':

            try: catalog=load_h5_file(path['path'], path['table'], columns_to_be_read, additional_column, max_objects)
            except: print ('Error in reading the hdf5 file. Path, table and columns need to be provided')

        else:
            sys.exit ('Error: format {0} [file: {1}] not implemented yet.'.format(file_format,path['path']))

        return catalog

def load_fits_file(filename, columns, additional_column=None,max_objects=0, **kwargs):
    """Load up fits file with fitsio, returns pandas dataframe

    Parameters
    ----------
    filename : string that fitsio reads
    columns : columns we extract from file
    kwargs : whatever to pass to fitsio to read the filename

    Returns
    -------
    df: pandas dataframe with specified columns

    Notes
    -----

    """
    #data = fitsio.read(filename[0], columns=columns, **kwargs)

    data=pf.open(filename)
    data=data[1].data
    if max_objects > 0 and max_objects < len(data):
        # only take max_objects
        indx = np.random.choice(len(data), max_objects, replace=False)
        data = data[indx]


    df = pd.DataFrame({key: data[key].byteswap().newbyteorder()
                       for key in columns})


    return df

def load_h5_file(filename, tablename, columns, additional_column=None, max_objects=0, **kwargs):
    """Load up h5 file, returns pandas dataframe

    Parameters
    ----------
    filename: string that fitsio reads
    columns : columns we extract from file
    kwargs: whatever to pass to pandas to read the filename

    Returns
    -------
    df: pandas dataframe with specified columns

    Notes
    -----

    """
    try:
        df = pd.read_hdf(filename, tablename, columns=columns)
    except TypeError:
        # problem with fixed pandas tables, so just load the full thing, drop
        df = pd.read_hdf(filename, tablename)
        drop_columns = [column for column in df.columns if column not in columns]
        if len(drop_columns) > 0:
            df.drop(drop_columns, axis=1, inplace=True)
    if max_objects > 0 and max_objects < len(df):
        # only take sample_num objects
        indx = np.random.choice(len(df), max_objects, replace=False)
        df = df.iloc[indx]
        # redo index
        df.index = np.arange(len(df))

    return df

    #check on columns

def check_columns(columns,catalog,additional_column=None):

    # columns in the output file: 'RA','DEC','Z','W','binning_column' and 'z_true_column' (only unknown)
    # the module dataset.py later adds the columns 'HPIX' (=jackknife region) and 'bins'


    #the additional column is for the binning. we have to check that it is not already present.
    if additional_column:
        catalog['binning_column']=copy.deepcopy(catalog[additional_column])
        #catalog.rename(columns={additional_column: 'binning_column'}, inplace=True)

    if columns['ra_column'] != 'RA':
        catalog.rename(columns={columns['ra_column']: 'RA'}, inplace=True)

    if columns['dec_column'] != 'DEC':
        catalog.rename(columns={columns['dec_column']: 'DEC'}, inplace=True)

    if 'z_column' in columns.keys():             #if I am loading randoms they could be provided without redshift
        if columns['z_column'] != 'Z':
                catalog.rename(columns={columns['z_column']: 'Z'}, inplace=True)

    if 'w_column' in columns.keys():
        if columns['w_column']=='None':
            catalog['W']=np.ones(catalog['RA'].shape[0])
        elif columns['w_column'] != 'W':
            catalog.rename(columns={columns['w_column']: 'W'}, inplace=True)
    else:
        catalog['W']=np.ones(catalog['RA'].shape[0])



    if 'z_true_column' in columns.keys():  #only for the unknown catalog
        if columns['z_true_column'] != 'Z_T' and columns['z_true_column']  != columns['z_column']:
            catalog.rename(columns={columns['z_true_column']: 'Z_T'}, inplace=True)
        elif columns['z_true_column'] != 'Z_T'  and columns['z_true_column'] == columns['z_column']:
            catalog['Z_T']=copy.deepcopy(catalog['Z'])
    return catalog


#  Routines for the redshift intervals *****************************

def digitize(cat_z, bins, label_every=0):
    """Digitize catalog assignment

    Parameters
    ----------
    cat_z: z values
    bins: dictionary, containing edges and central values of the redshift bins

    Returns
    -------
    labels: zbin labels

    Notes
    -----
    label == 0 means OUTSIDE zbins to the left
    label == len(zbins) + 1 means OUTSIDE zbins to the right!
    This means that zbins[label] gives you the RIGHT side of the bin
    """
    if label_every > 0:
        labels = np.zeros(len(cat_z), dtype=int) - 1
        n_iter = int(np.ceil(len(cat_z) / label_every))
        for ith in xrange(n_iter):
            indx_lower = ith * label_every
            indx_upper = indx_lower + label_every
            cat_z_i = cat_z[indx_lower: indx_upper]
            labels_i = np.digitize(cat_z_i,  bins['z_edges'])
            labels[indx_lower: indx_upper] = labels_i
    else:

        labels = np.digitize(cat_z, bins['z_edges'])

    return labels

def interpret_bins(bins):
    '''
    It creates the redshift interval according to the input in the config file.
    I am simplifying it, there s no need to keep the "between/equal" info until the end
    bins is not anymore a dict of dict, I am not sure we would use this feature of using different
    intervals at the same time, but it can be put it back.

    output:
    it gives back the redshift interval, the edges and the column used to bin. In theory, other columns could
    be used (colors for instance)
    '''
    results = {}

    if bins['type'] == 'between':
        # if 3 entries, then turn into numpy.linspace
        # I am using linspace because arange is not reliable (e.g.:http://quantumwise.com/forum/index.php?topic=110.0)
        edges=np.linspace(bins['array'][0],bins['array'][1],bins['array'][2]+1,endpoint=True)
        points=0.5*(edges[:-1]+edges[1:])
        results.update({'z_edges':edges,'z':points})
    elif bins['type'] == 'equal':
        edges=np.array(bins['array'])
        points=0.5*(edges[:-1]+edges[1:])
        results.update({'z_edges':edges,'z':points})
    return results,bins['name_column']

def make_z_distribution(data_z,zedges,num_random):
    '''
    It creates the redshift distribution for the randoms, like a histogram.

    '''
    hist, bins = np.histogram(data_z, bins=zedges)

    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(num_random)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    return random_from_cdf

def make_z_distribution_full(data_z,num_random):
    '''
    It creates the redshift distribution for the randoms

    '''
    import warnings
    warnings.filterwarnings("ignore")

    hst_z,edge=np.histogram(data_z,bins=50)
    edge1=edge[:-1] + 0.5*np.diff(edge)

    hst_z_long=np.zeros(len(hst_z)+2)
    edge2=np.zeros(len(hst_z)+2)

    hst_z_long[1:-1]=hst_z[:]

    edge2[1:-1]=edge1[:]
    edge2[0]=edge[0]
    edge2[-1]=edge[-1]

    hst_z_long[0]=((hst_z[1]-hst_z[0])/(edge1[1]-edge1[0]))*(edge2[0]-edge[1])+hst_z[0]
    hst_z_long[-1]=((hst_z[len(hst_z)-1]-hst_z[len(hst_z)-2])/(edge1[len(hst_z)-1]-edge1[len(hst_z)-2]))*(edge2[len(hst_z)-1]-edge[len(hst_z)-2])+hst_z[-1]

    a_int=interp1d(edge2,hst_z_long)
    edge2[0]=edge2[0]+0.0001
    edge2[-1]=edge2[-1]-0.0001
    G=[]
    xx=[]
    x=edge
    y=hst_z

    for k in range(0,50):
            extr=((edge2[-1]-edge2[0])/50.)*k+edge2[0]
            xx.append(extr)
            G.append(pyint.quad(a_int,edge2[0],extr)[0])
    xx=np.array(xx)
    G=np.array(G)
    norm=(pyint.quad(a_int,edge2[0],extr)[0])
    G=G/norm
    Gm1=interp1d(G,xx)

    z_rndm=Gm1(np.random.uniform(0.001, 1, num_random))
    return  z_rndm


#  progress bar ******************************************************

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
        status = "Done...\r\n"
    block = int(round(barLength*progress))



    if progress*100>1. and elapsed_time>0 :
        remaining=((elapsed_time-starting_time)/progress)*(1.-progress)
        text = "\rPercent: [{0}] {1:.2f}% {2}  - elapsed time: {3} - estimated remaining time: {4}".format( "#"*block + "-"*(barLength-block), progress*100, status,time.strftime('%H:%M:%S',time.gmtime(elapsed_time-starting_time)),time.strftime('%H:%M:%S',time.gmtime(remaining)))
    else:
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


# plot jackknives & data  ********************************************
def plot_data(catalog,label,indexes):
    color_i=[]
    for gg in range(1000):
            color_i.append('b')
            color_i.append('g')
            color_i.append('r')
            color_i.append('c')
            color_i.append('m')
            color_i.append('y')
            color_i.append('k')
    #plotting jaccknives
    #print (np.unique(catalog['bins']))

    for i in range(len(indexes)):
        mask=catalog['bins']==i+1
        ra=catalog['RA'][mask]
        dec=catalog['DEC'][mask]
        jck=catalog['HPIX'][mask]
        '''
        plt.plot(ra, dec, 'o', ms=4, alpha=1.,color='b')
        plt.savefig('./pairscount/data_plot/{0}_{1}_tot.png'.format(label,i))
        plt.close()
        '''

        #test
        fig= plt.figure()
        ax = fig.add_subplot(111)
        #ax.text(0.1, 0.9,'#_obj='+str((len(ra))),fontsize=11 , ha='center', transform=ax.transAxes)

        for j in range(len(np.unique(jck))):
            mask2=jck==j
            plt.plot(ra[mask2], dec[mask2], 'o', ms=4, alpha=1., color=color_i[j])
        plt.savefig('./pairscount/data_plot/{0}_{1}.png'.format(label,i+1))
        plt.close()

def  plot_redshift_distr(z,edges,label):

    fig= plt.figure()
    ax = fig.add_subplot(111)
    mask=(z>edges[0])&(z<edges[-1])
    ax.text(0.1, 0.9,'#_obj='+str((len(z[mask]))),fontsize=11 , ha='center', transform=ax.transAxes)
    plt.hist(z,edges)
    plt.savefig(label)
    plt.close()
# save and load python objects ***************************************

def save_obj( name,obj ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=3)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
