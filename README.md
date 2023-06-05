# CORE

This repository contains code for computing clustering redshift, a method used to estimate the redshift distribution of galaxies based on their spatial clustering. 
The Software has been used in the photo-z analysis of the DES Y1 data (https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.1664G/abstract)

Usage

- Prepare the input data:
Ensure your galaxy survey data is in a compatible format (e.g., FITS). Minimum requirements are ra,dec information for the target file, and ra,dec,z (or anything else to 'bin' in distance) for the reference sample. A random sample for either the targets or the reference sample is also required.


Open the config.yaml file and set the desired parameters for the clustering redshift computation.
Customize options such as correlation function type, binning scheme, etc. To run the test example, just do python run.py (the basic example infers the n(z) of COSMOS photometric galaxies using a collection of spectra from the Laigle2016 data)
The settings of the run are described in config.yaml.


Below: example of simulated redshift distribution recovered using the code. Image taken from Gatti et al., 2018
![alt text](https://github.com/mgatti29/CORE/blob/py3/input_files/clustering_z.png)



# Contributing
Contributions to this project are welcome! If you encounter any issues, have ideas for improvements, or would like to add new features, please submit a pull request or open an issue on the GitHub repository.

# License
This project is licensed under the MIT License. See the LICENSE file for more information.
