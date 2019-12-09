def write_values(dz,folder):
    a =open("{0}/priors.ini".format(folder),"w")
    
    a.write("[wl_photoz_errors] \n")
    a.write("bias_1 = gaussian 0. 0.016 \n")
    a.write("bias_2 = gaussian -0.019 0.013 \n")
    a.write("bias_3 = gaussian  0.009 0.011 \n")
    a.write("bias_4 = gaussian  -0.018 0.022 \n")
    a.write("bias_5 = gaussian  -0.018 0.022 \n")
    a.write("bias_6 = gaussian  -0.018 0.022 \n")
    a.write("bias_7 = gaussian  -0.018 0.022 \n")
    a.write("bias_8 = gaussian 0. 0.016 \n")
    a.write("bias_9 = gaussian -0.019 0.013 \n")
    a.write("bias_10 = gaussian  0.009 0.011 \n")
    a.write("bias_11 = gaussian  -0.018 0.022 \n")
    a.write("bias_12 = gaussian  -0.018 0.022 \n")
    a.write("bias_13 = gaussian  -0.018 0.022 \n")
    a.write("bias_14 = gaussian  -0.018 0.022 \n")
    a.write("bias_15 = gaussian 0. 0.016 \n")
    a.write("bias_16 = gaussian -0.019 0.013 \n")
    a.write("bias_17 = gaussian  0.009 0.011 \n")
    a.write("bias_18 = gaussian  -0.018 0.022 \n")
    a.write("bias_19 = gaussian  -0.018 0.022 \n")
    a.write("bias_20 = gaussian  -0.018 0.022 \n")
    a.write("bias_21 = gaussian  -0.018 0.022 \n")
    a.write("bias_22 = gaussian 0. 0.016 \n")
    a.write("bias_23 = gaussian -0.019 0.013 \n")
    a.write("bias_24 = gaussian  0.009 0.011 \n")
    a.write("bias_25 = gaussian  -0.018 0.022 \n")

    
    a.write("[shear_calibration_parameters] \n")
    a.write("m1 = gaussian 0.012 0.023 \n")
    a.write("m2 = gaussian 0.012 0.023 \n")
    a.write("m3 = gaussian 0.012 0.023 \n")
    a.write("m4 = gaussian 0.012 0.023 \n")
    a.write("m5 = gaussian 0.012 0.023 \n")
    a.write("m6 = gaussian 0.012 0.023 \n")
    a.write("m7 = gaussian 0.012 0.023 \n")
    a.write("m8 = gaussian 0.012 0.023 \n")
    a.write("m9 = gaussian 0.012 0.023 \n")
    a.write("m10 = gaussian 0.012 0.023 \n")
    a.write("m11 = gaussian 0.012 0.023 \n")
    a.write("m12 = gaussian 0.012 0.023 \n")
    a.write("m13 = gaussian 0.012 0.023 \n")
    a.write("m14 = gaussian 0.012 0.023 \n")
    a.write("m15 = gaussian 0.012 0.023 \n")
    a.write("m16 = gaussian 0.012 0.023 \n")
    a.write("m17 = gaussian 0.012 0.023 \n")
    a.write("m18 = gaussian 0.012 0.023 \n")
    a.write("m19 = gaussian 0.012 0.023 \n")
    a.write("m20 = gaussian 0.012 0.023 \n")
    a.write("m21 = gaussian 0.012 0.023 \n")
    a.write("m22 = gaussian 0.012 0.023 \n")
    a.write("m23 = gaussian 0.012 0.023 \n")
    a.write("m24 = gaussian 0.012 0.023 \n")
    a.write("m25 = gaussian 0.012 0.023 \n")



    a.write("[lens_photoz_errors] \n")
    a.write("bias_1 = gaussian 0.008 0.007 \n")
    a.write("bias_2 = gaussian -0.005 0.007 \n")
    a.write("bias_3 = gaussian 0.006 0.006 \n")
    a.write("bias_4 = gaussian 0.00 0.01 \n")
    a.write("bias_5 = gaussian 0.0 0.01 \n")
    a.write("bias_6 = gaussian 0.008 0.007 \n")
    a.write("bias_7 = gaussian -0.005 0.007 \n")
    a.write("bias_8 = gaussian 0.006 0.006 \n")
    a.write("bias_9 = gaussian 0.00 0.01 \n")
    a.write("bias_10 = gaussian 0.0 0.01 \n")
    a.write("bias_11 = gaussian 0.008 0.007 \n")
    a.write("bias_12 = gaussian -0.005 0.007 \n")
    a.write("bias_13 = gaussian 0.006 0.006 \n")
    a.write("bias_14 = gaussian 0.00 0.01 \n")
    a.write("bias_15 = gaussian 0.0 0.01 \n")
    a.write("bias_16 = gaussian 0.008 0.007 \n")
    a.write("bias_17 = gaussian -0.005 0.007 \n")
    a.write("bias_18 = gaussian 0.006 0.006 \n")
    a.write("bias_19 = gaussian 0.00 0.01 \n")
    a.write("bias_20 = gaussian 0.0 0.01 \n")
    a.write("bias_21 = gaussian 0.008 0.007 \n")
    a.write("bias_22 = gaussian -0.005 0.007 \n")
    a.write("bias_23 = gaussian 0.006 0.006 \n")
    a.write("bias_24 = gaussian 0.00 0.01 \n")
    a.write("bias_25 = gaussian 0.0 0.01 \n")

    a.write("[planck]\n")
    a.write("a_planck = gaussian 1.0 0.0025\n")

    

    a.close()


    a =open("{0}/values.ini".format(folder),"w")
    
    a.write("[cosmological_parameters] \n")
    a.write("omega_m = 0.05 0.286 0.9 \n")
    a.write("h0 = 0.3 0.7 1.0 \n")
    a.write("omega_b = 0.047 \n")
    a.write("sigma8_input = 0.2  0.82  1.6 \n")
    a.write("tau = 0.089 \n")
    a.write("n_s = 0.96 \n")
    a.write("A_s = 2.215e-9 \n")
    a.write("omega_k = 0.0 \n")
    a.write("w = -1.0 \n")
    a.write("wa = 0.0 \n")
    a.write("omnuh2 = 0.00065 \n")
    a.write("massless_nu = 2.046 \n")
    a.write("massive_nu = 1 \n")

    a.write("; And now the parameters we use to marginalize over systematics. \n")

    a.write("[intrinsic_alignment_parameters] \n")
    a.write("A = -5. 1.0  5.0 \n")


    a.write("[cosmological_parameters] \n")
    a.write("omega_m = 0.05 0.286 0.9 \n")
    a.write("h0 = 0.3 0.7 1.0 \n")
    a.write("omega_b = 0.047 \n")
    a.write("sigma8_input = 0.2  0.82  1.6 \n")
    a.write("tau = 0.089 \n")
    a.write("n_s = 0.96 \n")
    a.write("A_s = 2.215e-9 \n")
    a.write("omega_k = 0.0 \n")
    a.write("w = -1.0 \n")
    a.write("wa = 0.0 \n")
    a.write("omnuh2 = 0.00065 \n")
    a.write("massless_nu = 2.046 \n")
    a.write("massive_nu = 1 \n")


    a.write("[intrinsic_alignment_parameters] \n")
    a.write("A = -5. 1.0  5.0 \n")



    a.write("[wl_photoz_errors] \n")
    a.write("bias_1 = {0} \n".format(dz))
    a.write("bias_2 = {0} \n".format(dz))
    a.write("bias_3 = {0} \n".format(dz))
    a.write("bias_4 = {0} \n".format(dz))
    a.write("bias_5 = {0} \n".format(dz))
    a.write("bias_6 = {0} \n".format(dz))
    a.write("bias_7 = {0} \n".format(dz))
    a.write("bias_8 = {0} \n".format(dz))
    a.write("bias_9 = {0} \n".format(dz))
    a.write("bias_10 = {0} \n".format(dz))
    a.write("bias_11 = {0} \n".format(dz))
    a.write("bias_12 = {0} \n".format(dz))
    a.write("bias_13 = {0} \n".format(dz))
    a.write("bias_14 = {0} \n".format(dz))
    a.write("bias_15 = {0} \n".format(dz))
    a.write("bias_16 = {0} \n".format(dz))
    a.write("bias_17 = {0} \n".format(dz))
    a.write("bias_18 = {0} \n".format(dz))
    a.write("bias_19 = {0} \n".format(dz))
    a.write("bias_20 = {0} \n".format(dz))
    a.write("bias_21 = {0} \n".format(dz))
    a.write("bias_22 = {0} \n".format(dz))
    a.write("bias_23 = {0} \n".format(dz))
    a.write("bias_24 = {0} \n".format(dz))
    a.write("bias_25 = {0} \n".format(dz))
    
    
    a.write("[shear_calibration_parameters] \n")
    a.write("m1 = 0. \n")
    a.write("m2 = 0. \n")
    a.write("m3 = 0. \n")
    a.write("m4 = 0. \n")
    a.write("m5 = 0. \n")
    a.write("m6 = 0. \n")
    a.write("m7 = 0. \n")
    a.write("m8 = 0. \n")
    a.write("m9 = 0. \n")
    a.write("m10 = 0. \n")
    a.write("m11 = 0. \n")
    a.write("m12 = 0. \n")
    a.write("m13 = 0. \n")
    a.write("m14 = 0. \n")
    a.write("m15 = 0. \n")
    a.write("m16 = 0. \n")
    a.write("m17 = 0. \n")
    a.write("m18 = 0. \n")
    a.write("m19 = 0. \n")
    a.write("m20 = 0. \n")
    a.write("m21 = 0. \n")
    a.write("m22 = 0. \n")
    a.write("m23 = 0. \n")
    a.write("m24 = 0. \n")
    a.write("m25 = 0. \n")

    
    
    a.write("[lens_photoz_errors] \n")
    a.write("bias_1 = 0. \n")
    a.write("bias_2 = 0. \n")
    a.write("bias_3 = 0. \n")
    a.write("bias_4 = 0. \n")
    a.write("bias_5 = 0. \n")
    a.write("bias_6 = 0. \n")
    a.write("bias_7 = 0. \n")
    a.write("bias_8 = 0. \n")
    a.write("bias_9 = 0. \n")
    a.write("bias_10 = 0. \n")
    a.write("bias_11 = 0. \n")
    a.write("bias_12 = 0. \n")
    a.write("bias_13 = 0. \n")
    a.write("bias_14 = 0. \n")
    a.write("bias_15 = 0. \n")
    a.write("bias_16 = 0. \n")
    a.write("bias_17 = 0. \n")
    a.write("bias_18 = 0. \n")
    a.write("bias_19 = 0. \n")
    a.write("bias_20 = 0. \n")
    a.write("bias_21 = 0. \n")
    a.write("bias_22 = 0. \n")
    a.write("bias_23 = 0. \n")
    a.write("bias_24 = 0. \n")
    a.write("bias_25 = 0. \n")

    a.write("[bin_bias] \n")
    a.write("b1 = 1. \n")
    a.write("b2 = 1. \n")
    a.write("b3 = 1. \n")
    a.write("b4 = 1. \n")
    a.write("b5 = 1. \n")
    a.write("b6 = 1. \n")
    a.write("b7 = 1. \n")
    a.write("b8 = 1. \n")
    a.write("b9 = 1. \n")
    a.write("b10 = 1. \n")
    a.write("b11 = 1. \n")
    a.write("b12 = 1. \n")
    a.write("b13 = 1. \n")
    a.write("b14 = 1. \n")
    a.write("b15 = 1. \n")
    a.write("b16 = 1. \n")
    a.write("b17 = 1. \n")
    a.write("b18 = 1. \n")
    a.write("b19 = 1. \n")
    a.write("b20 = 1. \n")
    a.write("b21 = 1. \n")
    a.write("b22 = 1. \n")
    a.write("b23 = 1. \n")
    a.write("b24 = 1. \n")
    a.write("b25 = 1. \n")


    a.write("[galaxy_luminosity_function] \n")
    #a.write("alpha_binned = np.array(2.,20)\n")
    #"""
    a.write("alpha_binned_1 = 2. \n")
    a.write("alpha_binned_2 = 2. \n")
    a.write("alpha_binned_3 = 2. \n")
    a.write("alpha_binned_4 = 2. \n")
    a.write("alpha_binned_5 = 2. \n")
    a.write("alpha_binned_6 = 2. \n")
    a.write("alpha_binned_7 = 2. \n")
    a.write("alpha_binned_8 = 2. \n")
    a.write("alpha_binned_9 = 2. \n")
    a.write("alpha_binned_10 = 2. \n")
    a.write("alpha_binned_11 = 2. \n")
    a.write("alpha_binned_12 = 2. \n")
    a.write("alpha_binned_13 = 2. \n")
    a.write("alpha_binned_14 = 2. \n")
    a.write("alpha_binned_15 = 2. \n")
    a.write("alpha_binned_16 = 2. \n")
    a.write("alpha_binned_17 = 2. \n")
    a.write("alpha_binned_18 = 2. \n")
    a.write("alpha_binned_19 = 2. \n")
    a.write("alpha_binned_20 = 2. \n")
    a.write("alpha_binned_21 = 2. \n")
    a.write("alpha_binned_22 = 2. \n")
    a.write("alpha_binned_23 = 2. \n")
    a.write("alpha_binned_24 = 2. \n")
    a.write("alpha_binned_25 = 2. \n")
   # """
    a.close()
    


def write_options(path_fits,folder,nz_source="nz_source"):
    
    a =open("{0}/demo17.ini".format(folder),"w")
    
    a.write("[runtime] \n")
    a.write("sampler = test \n")
    a.write("root = ${COSMOSIS_SRC_DIR} \n")

    a.write("[DEFAULT] \n")
    a.write("BASELINE_DIR= {0}/ \n".format(folder))
    a.write("planck_like_path=${COSMOSIS_SRC_DIR}/cosmosis-standard-library/likelihood/planck2015/data/plc_2.0 \n")
#2PT_FILE = %(BASELINE_DIR)s/../../../lensing_z/try_cosmosis_10.fits
    a.write("2PT_FILE = {0}".format(path_fits)+" \n")

    a.write("2PT_DATA_SETS = w w_ref gammat \n")
    a.write("RUN_NAME = 1 \n")

    a.write("[campaign] \n")
    a.write("time_hours = 120 \n")



    a.write("[pmaxlike] \n")
    a.write("maxiter = 10000 \n")
    a.write("tolerance = 0.1 \n")
    a.write("output_ini = best_fit.ini \n")

    a.write("[star] \n")
    a.write("nsample_dimension = 100 \n")


    a.write("[minuit] \n")
    a.write("maxiter=5000 \n")
    a.write("strategy=medium \n")
    a.write("algorithm=fallback \n")
    a.write("tolerance=1000 \n")
    a.write("output_ini=ml_results.ini \n")
    a.write("save_dir=ml_results \n")
    a.write("save_cov=cov.txt \n")
    a.write("verbose=T \n")

    a.write("[multinest] \n")
    a.write("max_iterations=50000 \n")
    a.write("multinest_outfile_root=mn_%(RUN_NAME)s \n")
    a.write("resume=F \n")
    a.write("live_points=1000 \n")
    a.write("efficiency=0.05 \n")
    a.write("tolerance=0.1    \n")
    a.write("constant_efficiency=T \n")
    a.write("[test] \n")
    a.write("save_dir={0}/ \n".format(folder))
    a.write("fatal_errors=T \n")
    a.write("cut_wtheta = 1,1 \n")

    a.write("[output] \n")
    a.write("filename=%(2PT_FILE)s_%(RUN_NAME)s_chain.txt \n")
    a.write("format=text \n")

    a.write("[grid] \n")
    a.write("nsample_dimension = 10 \n")
    a.write("save_dir=grid_output \n")

    a.write("[emcee] \n")
    a.write("burn=0.3 \n")
    a.write("walkers = 160 \n")
    a.write("samples = 10000 \n")
    a.write("nsteps = 5 \n")


    a.write("[pipeline] \n")
    a.write("quiet=T \n")
    a.write("timing=F \n")
    a.write("debug=F \n")
    a.write("priors = %(BASELINE_DIR)s/priors.ini \n")

    a.write("modules = consistency camb halofit growth extrapolate fits_nz unbiased_galaxies  pk_to_cl pk_to_cl_auto bin_bias 2pt_gal 2pt_gal_shear 2pt_gal_auto 2pt_gal_shear_auto 2pt_gal_shear_auto_2 2pt_mag_shear 2pt_mag_pos pk_to_cl22 2pt_mag_pos_2 2pt_mag_mag \n")


    a.write("values = %(BASELINE_DIR)s/values.ini \n")
    a.write("likelihoods =  \n")
    a.write("extra_output = cosmological_parameters/sigma_8 \n")

    a.write("[2pt_like_allscales] \n")
    a.write("file = cosmosis-standard-library/likelihood/2pt/2pt_like.py \n")
    a.write("include_norm=T \n")
    a.write("data_file = %(2PT_FILE)s \n")
    a.write("data_sets = %(2PT_DATA_SETS)s \n")
    a.write("make_covariance=F \n")
    a.write("covmat_name=COVMAT \n")

    a.write("[2pt_like] \n")
    a.write("file = cosmosis-standard-library/likelihood/2pt/2pt_like.py \n")
    a.write("include_norm=T \n")
    a.write("data_file = %(2PT_FILE)s \n")
    a.write("data_sets = %(2PT_DATA_SETS)s \n")
    a.write("make_covariance=F \n")
    a.write("covmat_name=COVMAT \n")



    a.write("[2ptlike_large_scales] \n")
    a.write("file = cosmosis-standard-library/likelihood/2pt/2pt_like.py \n")
    a.write("data_file = %(2PT_FILE)s \n")
    a.write("data_sets = %(2PT_DATA_SETS)s \n")
    a.write("make_covariance=F \n")
    a.write("covmat_name=COVMAT \n")





    a.write("[bias] \n")
    a.write("file=${COSMOSIS_SRC_DIR}/cosmosis-des-library/tcp/simple_bias/bias.py \n")

    a.write("[IA] \n")
    a.write("file=cosmosis-standard-library/intrinsic_alignments/la_model/linear_alignments_interface.py \n")
    a.write("do_galaxy_intrinsic=T \n")
    a.write("method=bk_corrected \n")
    a.write("[add_intrinsic] \n")
    a.write("file=cosmosis-standard-library/shear/add_intrinsic/add_intrinsic.py \n")
    a.write("shear-shear=T \n")
    a.write("perbin=F \n")
    a.write("position-shear=T \n")

    a.write("[stitch] \n")
    a.write("file=${COSMOSIS_SRC_DIR}/cosmosis-des-library/IAs/stitch/stitch_ia.py \n")
    a.write("name_1=red \n")
    a.write("name_2=blue \n")

    a.write("[consistency] \n")
    a.write("file = cosmosis-standard-library/utility/consistency/consistency_interface.py \n")

    a.write("[camb] \n")
    a.write("file = cosmosis-standard-library/boltzmann/camb/camb.so \n")
    a.write("mode=all \n")
    a.write("lmax=2500 \n")
    a.write("feedback=0 \n")
    a.write("kmin=1e-5 \n")
    a.write("kmax=10.0 \n")
    a.write("nk=200 \n")


    a.write("[camb_wmap] \n")
    a.write("file = cosmosis-standard-library/boltzmann/camb/camb.so \n")
    a.write("mode=all \n")
    a.write("lmax=1300 \n")
    a.write("feedback=0 \n")

    a.write("[camb_planck] \n")
    a.write("file = cosmosis-standard-library/boltzmann/camb/camb.so \n")
    a.write("mode=all \n")
    a.write("lmax=2650 \n")
    a.write("feedback=0 \n")
    a.write("kmin=1e-5 \n")
    a.write("kmax=10.0 \n")
    a.write("nk=200 \n")
    a.write("do_lensing = T \n")
    a.write("do_tensors = F \n")
    a.write("do_nonlinear = T \n")
    a.write("high_ell_template = ${COSMOSIS_SRC_DIR}/cosmosis-standard-library/boltzmann/camb/camb_Jan15/HighLExtrapTemplate_lenspotentialCls.dat   \n")
    a.write("accuracy_boost=1.1   \n")
    a.write("high_accuracy_default = T \n")

    a.write("[extrapolate] \n")
    a.write("file = cosmosis-standard-library/boltzmann/extrapolate/extrapolate_power.py  \n")
    a.write("kmax = 500. \n")

    a.write("[sigma8_rescale] \n")
    a.write("file = cosmosis-standard-library/utility/sample_sigma8/sigma8_rescale.py \n")

    a.write("[halofit] \n")
    a.write("file = cosmosis-standard-library/boltzmann/halofit_takahashi/halofit_interface.so \n")
    a.write("nk=700 \n")

    a.write("[unbiased_galaxies] \n")
    a.write("file = cosmosis-standard-library/bias/no_bias/no_bias.py \n")


    a.write("[pk_to_cl]\n")
    a.write("file = cosmosis-standard-library/structure/projection/project_2d.py\n")
    a.write("ell_min = 0.1\n")
    a.write("ell_max = 5.0e5\n")
    a.write("n_ell = 400\n")

    a.write("magnification-shear = lens-source \n")
   
    a.write("magnification-position = source-lens \n")
    a.write("position-position =  lens-source \n")
    a.write("position-shear = lens-source \n")

    a.write("verbose = F \n")
    a.write("get_kernel_peaks=F \n")

    a.write("[pk_to_cl22]\n")
    a.write("file = cosmosis-standard-library/structure/projection/project_2d.py\n")
    a.write("ell_min = 0.1\n")
    a.write("ell_max = 5.0e5\n")
    a.write("n_ell = 400\n")

    a.write("magnification-shear = lens-source \n")
    a.write("magnification-magnification = lens-source \n")
    a.write("magnification-position = lens-source \n")
    a.write("position-position =  lens-source \n")
    a.write("position-shear = lens-source \n")

    a.write("verbose = F \n")
    a.write("get_kernel_peaks=F \n")

    
    
    a.write("[pk_to_cl_auto] \n")
    a.write("file = cosmosis-standard-library/structure/projection/project_2d.py \n")
    a.write("ell_min = 0.1 \n")
    a.write("ell_max = 5.0e5 \n")
    a.write("n_ell = 400 \n")
    a.write("auto_single = True \n")
    a.write("position-position =  lens-lens:auto_galaxy_cl \n")
    a.write("position-shear =  lens-lens:auto_shear_cl \n")
    a.write("verbose = F \n")
    a.write("get_kernel_peaks=F \n")

    a.write("[save_2pt] \n")
    a.write("file = cosmosis-standard-library/likelihood/2pt/save_2pt_new.py \n")
    a.write("theta_min = 2.5 \n")
    a.write("theta_max = 250.0 \n")
    a.write("n_theta = 2000 \n")
    a.write("real_space = T \n")
    a.write("make_covariance = F \n")
    a.write("shear_nz_name = nz_source \n")
    a.write("position_nz_name = nz_lens \n")
    a.write("filename =  ${COSMOSIS_SRC_DIR}lensingz/lensingz.fits \n")
    a.write("clobber = T \n")
    a.write("number_density_shear_bin =  2.0  2.0  2.0  2.0  2.0 \n")
    a.write("number_density_lss_bin = 2.0  2.0  2.0 \n")
    a.write("sigma_e_bin = 0.2  0.2  0.2  0.2  0.2 \n")
    a.write("survey_area = 1500.0 \n")


    a.write("[ia_z_field] \n")
    a.write("file = ${COSMOSIS_SRC_DIR}/cosmosis-standard-library/intrinsic_alignments/z_powerlaw/ia_z_powerlaw.py \n")
    a.write("do_galaxy_intrinsic = T \n")

    a.write("[save_c_ell_fits] \n")
    a.write("file = cosmosis-standard-library/likelihood/2pt/save_2pt.py \n")
    a.write("ell_min = 100.0 \n")
    a.write("ell_max = 2000.0 \n")
    a.write("n_ell = 10 \n")
    a.write("shear_nz_name = nz_source \n")
    a.write("position_nz_name = nz_lens \n")
    a.write("filename = internal_simulation.fits \n")
    a.write("clobber = T \n")
    a.write("number_density_shear_bin =  2.0  2.0  2.0  2.0  2.0 \n")
    a.write("number_density_lss_bin = 2.0  2.0  2.0 \n")
    a.write("sigma_e_bin = 0.2  0.2  0.2  0.2  0.2 \n")
    a.write("survey_area = 1500.0 \n")



    a.write("[bin_bias] \n")
    a.write("file = cosmosis-standard-library/bias/binwise_bias/bin_bias.py \n")
    a.write("perbin=T \n")

    a.write("[load_nz_source] \n")
    a.write("file=cosmosis-standard-library/number_density/load_nz/load_nz.py \n")
    a.write("filepath= %(BASELINE_DIR)s/../../comparison/comparison_details_v1/source_4.nz.txt \n")
    a.write("output_section= {0} \n".format(nz_source))
    a.write("upsampling=1 \n")
    a.write("histogram=T \n")

    a.write("[load_nz_lens] \n")
    a.write("file=cosmosis-standard-library/number_density/load_nz/load_nz.py \n")
    a.write("output_section= nz_lens \n")
    a.write("filepath=%(BASELINE_DIR)s/../../comparison/comparison_details_v1/lens_5.nz.txt \n")
    a.write("upsampling=1 \n")
    a.write("histogram=T \n")

    a.write("[unbiased_galaxies] \n")
    a.write("file = cosmosis-standard-library/bias/no_bias/no_bias.py \n")
    a.write("use_lin_power=False \n")


    a.write("[fits_nz] \n")
    a.write("file = cosmosis-standard-library/number_density/load_nz_fits/load_nz_fits.py \n")
    a.write("nz_file = %(2PT_FILE)s \n")
    a.write("data_sets = lens source \n")
    a.write("prefix_section = T \n")
    a.write("prefix_extension = T \n")



    a.write("[2pt_shear] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("corr_type = 0 \n")

    a.write("[2pt_gal] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("corr_type = 1 \n")


    a.write("[2pt_gal_auto] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = galaxy_cl_auto_galaxy_cl \n")
    a.write("output_section_name = matter_auto_xi \n")

    a.write("corr_type = 1 \n")


    a.write("[2pt_gal_shear_auto_2] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = galaxy_cl_auto_galaxy_cl \n")
    a.write("output_section_name = matter_auto_xi_2 \n")
    a.write("corr_type = 2 \n")

    a.write("[2pt_gal_shear_auto] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = galaxy_shear_cl_auto_shear_cl \n")
    a.write("output_section_name = galaxy_shear_auto_xi \n")
    a.write("corr_type = 2 \n")


    a.write("[2pt_gal_shear] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("corr_type = 2 \n")



    a.write("[2pt_mag_pos] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = magnification_galaxy_cl \n")
    a.write("output_section_name = mag_pos_xi \n")
    a.write("corr_type = 1 \n")


    a.write("[2pt_mag_pos_2] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = magnification_galaxy_cl \n")
    a.write("output_section_name = mag_pos_xi_2 \n")
    a.write("corr_type = 1 \n")

    a.write("[2pt_mag_mag] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = magnification_cl \n")
    a.write("output_section_name = mag_mag_xi_2 \n")
    a.write("corr_type = 1 \n")

    a.write("[2pt_mag_shear] \n")
    a.write("file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so \n")
    a.write("input_section_name = magnification_shear_cl \n")
    a.write("output_section_name = mag_shear_xi \n")
    a.write("corr_type = 2 \n")

    a.write("[shear_m_bias] \n")
    a.write("file = cosmosis-standard-library/shear/shear_bias/shear_m_bias.py \n")
    a.write("m_per_bin = True \n")
    a.write("verbose = F \n")

    a.write("[source_photoz_bias] \n")
    a.write("file = cosmosis-standard-library/number_density/photoz_bias/photoz_bias.py \n")
    a.write("mode = additive \n")
    a.write("sample = nz_source \n")
    a.write("bias_section = wl_photoz_errors \n")
    a.write("interpolation = linear \n")

    a.write("[lens_photoz_bias] \n")
    a.write("file = cosmosis-standard-library/number_density/photoz_bias/photoz_bias.py \n")
    a.write("mode = additive \n")
    a.write("sample = nz_lens \n")
    a.write("bias_section = lens_photoz_errors \n")
    a.write("interpolation = linear \n")
            
            
    a.write("[growth] \n")
    a.write("file=cosmosis-standard-library/structure/growth_factor/interface.so \n")
    a.write("zmin=0. \n")
    a.write("zmax=4. \n")
    a.write("nz=401 \n")

    a.write("[extract] \n")
    a.write("file = ${PWD}/datavector.py \n")
    a.write("outfile = datavector.txt \n")

    a.write("[bias_neutrinos] \n")
    a.write("file=cosmosis-des-library/lss/braganca-neutrino-bias/interface.so \n")
    a.write("feedback=true \n")
    a.write("LINEAR_GROWTH_EPS_ABS = 0.0 \n")
    a.write("LINEAR_GROWTH_EPS_REL = 1.0e-6 \n")
    a.write("LINEAR_GROWTH_RK_TYPE = RK45 \n")
    a.write("LINEAR_GROWTH_SPLINE_ZMIN = 0.0 \n")
    a.write("LINEAR_GROWTH_SPLINE_ZMAX = 1.00 \n")
    a.write("LINEAR_GROWTH_SPLINE_DELTA_Z = 0.02 \n")
          
            
def write_params_v(om,h0,ob,s8,ns,folder):
    filew = open("{0}/values.ini".format(folder),"w")
    filew.write("[cosmological_parameters] \n")
    filew.write("omega_m = 0.05 {0} 0.9 \n".format(om))
    filew.write("h0 = 0.3 {0} 1.0 \n".format(h0))
    filew.write("omega_b = 0.03 {0} 0.12 \n".format(ob))
    filew.write("sigma8_input= 0.2  {0} 1.6 \n".format(s8))
    filew.write("tau = 0.089 \n")
    filew.write("n_s = {0} \n".format(ns))
    filew.write("omega_k = 0.0 \n")
    filew.write("w = -1.0 \n")
    filew.write("wa = 0.0 \n")
    filew.write("omnuh2 = 0.00065 \n")
    filew.write("massless_nu = 2.046 \n")
    filew.write("massive_nu = 1 \n")
    filew.write("A_s = 2.215e-9\n")
    
    filew.write("\n")
    filew.write("[intrinsic_alignment_parameters]\n")
    filew.write("A = -5. 1.0  5.0\n")
    
    filew.write("\n")
    filew.write("[wl_photoz_errors]\n")
    for i in range(4):
        filew.write("bias_{0} = 0.\n".format(i+1))
    filew.write("\n")


    filew.write("\n")
    filew.write("[shear_calibration_parameters]\n")
    for i in range(4):
        filew.write("m{0} = 0.\n".format(i+1))
    filew.write("\n")


    filew.write("\n")
    filew.write("[lens_photoz_errors]\n")
    for i in range(25):
        filew.write("bias_{0} = 0.\n".format(i+1))
    filew.write("\n")
    
    filew.write("\n")
    filew.write("[bin_bias]\n")
    for i in range(25):
        filew.write("b{0} = 1.\n".format(i+1))
    filew.write("\n")




    filew.write("\n")
    filew.write("[galaxy_luminosity_function]\n")
    for i in range(25):
        filew.write("alpha_binned_{0} = 1.\n".format(i+1))
    filew.write("\n")


    filew.close()