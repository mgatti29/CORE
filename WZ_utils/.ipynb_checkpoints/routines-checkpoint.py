import numpy as np
import twopoint
def make_fits(nz,qz,bz,nzs):

    import twopoint

    gt = 1.
    angular_bins = 1 # it's integrated
    redshift_bins_gt_rmg = nz.shape[1]
    redshift_bins_w_rmg = nz.shape[1]
    redshift_bins_WL = nz.shape[0]+1
    

    dv_start = 0

    gt_length = redshift_bins_gt_rmg*redshift_bins_WL*angular_bins
    gt_values = np.zeros(gt_length, dtype=float)

    bin1 = np.zeros(gt_length, dtype=int)
    bin2 = np.zeros_like(bin1)
    angular_bin = np.zeros_like(bin1)
    angle = np.zeros_like(gt_values)

    for l in range(0, (redshift_bins_gt_rmg)):
        for s in range((redshift_bins_WL)):
                bin_pair_inds = np.arange(dv_start, dv_start + angular_bins )
                bin1[bin_pair_inds] = l
                bin2[bin_pair_inds] = s#+redshift_bins_gt_rmg

                angular_bin[bin_pair_inds] = np.arange(angular_bins)
                angle[bin_pair_inds] = 0.
                try:
                    gt_values[bin_pair_inds] = qz[s,l,0]
                except:
                    pass
                dv_start += angular_bins 


    gammat = twopoint.SpectrumMeasurement('gammat', (bin1, bin2),
                                                         (twopoint.Types.galaxy_position_real,
                                                          twopoint.Types.galaxy_shear_plus_real),
                                                         ['no_nz', 'no_nz'], 'SAMPLE', angular_bin, gt_values, angle=angle, angle_unit='arcmin')




    dv_start=0
    print redshift_bins_gt_rmg
    gt_length = redshift_bins_gt_rmg*redshift_bins_WL*angular_bins#+redshift_bins_gt_rmg*angular_bins
    gt_values = np.zeros(gt_length, dtype=float)


    bin1 = np.zeros(gt_length, dtype=int)
    bin2 = np.zeros_like(bin1)
    angular_bin = np.zeros_like(bin1)
    angle = np.zeros_like(gt_values)
    print redshift_bins_gt_rmg,redshift_bins_WL
    for l in range(0, (redshift_bins_gt_rmg)):
        for s in range((redshift_bins_WL)):
                bin_pair_inds = np.arange(dv_start, dv_start + angular_bins )
  
                bin1[bin_pair_inds] = l
                bin2[bin_pair_inds] = s#+redshift_bins_gt_rmg

                angular_bin[bin_pair_inds] = np.arange(angular_bins)
                angle[bin_pair_inds] = 0.
                try:
                    gt_values[bin_pair_inds] = nz[s,l,0]
                except:
                    pass
                dv_start += angular_bins 

    '''
    for l in range(0, (redshift_bins_gt_rmg)):
        
                bin_pair_inds = np.arange(dv_start, dv_start + angular_bins )
                bin1[bin_pair_inds] = l
                bin2[bin_pair_inds] = l

                angular_bin[bin_pair_inds] = np.arange(angular_bins)
                angle[bin_pair_inds] = 0.

                gt_values[bin_pair_inds] = bz[0,l,0]
                dv_start += angular_bins 
    '''
    w = twopoint.SpectrumMeasurement('w', (bin1, bin2),
                                                         (twopoint.Types.galaxy_position_real,
                                                          twopoint.Types.galaxy_position_real),
                                                         ['no_nz', 'no_nz'], 'SAMPLE', angular_bin, gt_values, angle=angle, angle_unit='arcmin')



    dv_start = 0
    gt_length = redshift_bins_gt_rmg*redshift_bins_WL*angular_bins
    gt_values = np.zeros(gt_length, dtype=float)

    bin1 = np.zeros(redshift_bins_gt_rmg, dtype=int)
    bin2 = np.zeros_like(bin1)
    angular_bin = np.zeros_like(bin1)
    angle = np.zeros_like(gt_values)

    for l in range(0, (redshift_bins_gt_rmg)):
                print bin_pair_inds
                bin_pair_inds = np.arange(dv_start, dv_start + angular_bins )
                bin1[bin_pair_inds] = l
                bin2[bin_pair_inds] = l

                angular_bin[bin_pair_inds] = np.arange(angular_bins)
                angle[bin_pair_inds] = 0.

                gt_values[bin_pair_inds] = bz[0,l,0]
                dv_start += angular_bins 


    w_ref = twopoint.SpectrumMeasurement('w_ref', (bin1, bin2),
                                                         (twopoint.Types.galaxy_position_real,
                                                          twopoint.Types.galaxy_position_real),
                                                         ['no_nz', 'no_nz'], 'SAMPLE', angular_bin, gt_values, angle=angle, angle_unit='arcmin')
  

    print 'done'
    cov = None

   # obj = twopoint.TwoPointFile([gammat,w,w_ref], [nz_lenses_20, nz_wl, windows=None, covmat_info=cov)
    obj = twopoint.TwoPointFile([gammat,w,w_ref], [nzs, nz_wl], windows=None, covmat_info=cov)

    return obj,[gammat,w],None









