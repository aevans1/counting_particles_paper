# counting_particles_paper
Figures, scripts for the paper "Counting particles in cryo-electron microscopy may result in incorrect population estimates", biorxiv preprint available [here](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1).



- Each script titled `figure_x` computes figure_x from the manuscript.
Note: To recreate the figure exactly, you will need `tex` to enable it in the matplotlib settings, as mentioned in the script. Otherwise, you may have to adjust some font sizes.

- Each script titled `compute_weights_x` will compute the data used directly in the figure corresponding to dataset x.

- These scripts ^ all use some helper functions from `utils.py`, which includes the code for deconvolution and ensemble reweighting.

The input data required for the `compute_` scripts,  was computed via the [`RECOVAR` library](https://github.com/ma-gilles/recovar) and [cryoSPARC](https://cryosparc.com/). The pipelines for these is described in the manuscript, and the scripts used to make the data are in `data_generation_scripts`:
- simulate the synthetic datasets in RECOVAR, or manually add noise to datasets (`make_x`)
- compute likelihoods in RECOVAR (`assign_x`)
These functions were all run on a cluster with a lot of memory available, and were using older versions of RECOVAR, so if one wants to repurpose them it will take some adjusting!  

