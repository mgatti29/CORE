"""
Plot redshift distribution

.. module:: plot
"""

import numpy as np

#from .dataset import catalog_to_histogram, _modify_indices_redshift
from .compare import _chi2

def plot(pcs, covs, stats, tags, unknown_catalog, redshift_bins,
         plot_path='', label='', label_cat='',
         redshift_column='Z', redshift_catalog_columns=['Z'],
         z_min=0, z_max=1.0, time0=0,
         **kwargs):
    """Create plots comparing the paircounts with the catalogs

    Parameters
    ----------

    redshift_column : string [default: 'Z'], when using non-angular units, where
            do we look to find the central redshift
    redshift_catalog_columns : list of strings [default: ['Z']], which list all
                               of the catalog estimators we want to compare against
    z_min : float [default: 0] minimum redshift of samples
    z_max : float [default: 1] maximum redshift of samples
    Returns
    -------
    """

    if time0: from time import time


    # determine the tag
    unique_pc_tags = pcs[tags].drop_duplicates().reset_index(drop=True)

    tags_unknown = [tag for tag in tags if tag in unknown_catalog.columns]
    # make the different plots based on the tags
    for index, key in unique_pc_tags.iterrows():
        if time0:
            print(index, key, time() - time0)
            conds_cat = (unknown_catalog[tags_unknown] == key[tags_unknown]).all(axis=1).values
            print(unknown_catalog[conds_cat].iloc[0])


        # make figure
        fig, ax = make_figure()

        # get these values from data
        conds = (pcs[tags] == key).all(axis=1).values
        pc = pcs[conds]

        # get corresponding stats
        stat = stats[(stats[tags] == key).all(axis=1)]

        # get corresponding cov
        cov = covs[np.ix_(conds, conds)]

        z = pc[redshift_column].values
        pz = pc['phi'].values
        pz_err = pc['phi_err'].values

        # create label
        label_plot = label + ' $<x> = {{{0:.3f}}} \pm {{{1:.3f}}}$'.format(stat['z_mean'].values[0], stat['z_mean_error'].values[0])

        pc_kwargs = {'x': z, 'y': pz, 'yerr': pz_err,
                     'marker': 'o',
                     'linestyle': 'None',
                     'color': 'black', 'alpha': 0.8, 'label': label_plot}
        ax.errorbar(**pc_kwargs)

        for ith, redshift_catalog_column in enumerate(redshift_catalog_columns):


            colors = ['black', 'red', 'blue', 'magenta', 'cyan']
            color = colors[ith % len(colors)]

            # convert catalog to corresponding pdf
            conds_cat = (unknown_catalog[tags_unknown] == key[tags_unknown]).all(axis=1).values
            z_cat = unknown_catalog[conds_cat][redshift_catalog_column]
            if 'W' in unknown_catalog:
                weights = unknown_catalog[conds_cat]['W']
            else:
                weights = None
            pz_cat, z_vals = catalog_to_histogram(z_cat, redshift_bins,
                                                  weights=weights,
                                                  z_min=z_min, z_max=z_max)

            # get catalog stats
            pz_z_vals = pz_cat * z_vals
            z_mean_cat = (0.5 * (z_vals[1:] - z_vals[:-1]) * (pz_z_vals[1:] + pz_z_vals[:-1])).sum()
            z_width_squared = (pz_z_vals - z_mean_cat * pz_cat) ** 2
            z_width = np.sqrt((0.5 * (z_vals[1:] - z_vals[:-1]) * (z_width_squared[1:] + z_width_squared[:-1])).sum())

            # compare chi2
            pc_conds, cat_conds = _modify_indices_redshift(z, z_vals)
            chi2 = _chi2(pz_cat[cat_conds], pz[pc_conds], cov[np.ix_(pc_conds, pc_conds)])

            label_cat_plot = label_cat + ' ' + redshift_catalog_column + ' $\chi^2 = {{{0:.1f}}}, \mathrm{{d.o.f.}} = {{{1}}}, <x> = {{{2:.3f}}}$'.format(chi2, sum(pc_conds), z_mean_cat)


            cat_kwargs = {'x': z_vals, 'y': pz_cat, 'marker': 'None',
                                                   'linestyle': '-', 'linewidth': 2,
                                                    'color': color, 'alpha': 0.8, 'label': label_cat_plot}

            # plot
            #ax.errorbar(**cat_kwargs)
            ax.errorbar(**cat_kwargs)

        # make legend transparent
        ax.legend(loc='lower right', fancybox=True).get_frame().set_alpha(0.5)
        # ax.set_ylim(None, None)
        ax.set_xlim(z_min, z_max)

        # save figures
        if len(plot_path) > 0:
            # make file name
            out_name = ''
            for value, index in zip(key.values, key.index):
                out_name += '__{0}_{1}'.format(index, value)
            fig.savefig('{0}/redshift_distribution{1}.png'.format(plot_path, out_name))


def make_figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('$z$', fontsize=32)
    ax.set_ylabel(r'$\frac{dN}{dz}$', fontsize=32, rotation=0)
    ax.yaxis.labelpad = 25


    return fig, ax
