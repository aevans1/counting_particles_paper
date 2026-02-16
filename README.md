# counting_particles_paper
Figures, scripts for the paper "Counting particles in cryo-electron microscopy may result in incorrect population estimates", biorxiv preprint available [here](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1).



- Each script titled `figure_x` computes figure_x from the manuscript.
Note: To recreate the figure exactly, you will need `tex` to enable it in the matplotlib settings, as mentioned in the script. Otherwise, you may have to adjust some font sizes.

- Each script titled `compute_weights_x` will compute the data used directly in the figure corresponding to dataset x.

- These scripts ^ all use some helper functions from `utils.py`, which includes the code for deconvolution and ensemble reweighting.

The input data required for the `compute_` scripts,  was computed via the [`RECOVAR` library](https://github.com/ma-gilles/recovar) and [cryoSPARC](https://cryosparc.com/). The pipelines for these is described in the manuscript, and the scripts used to make the data are in `data_generation_scripts`:
- simulate the synthetic datasets in RECOVAR, or manually add noise to datasets (`make_x`)
- compute likelihoods in RECOVAR (`assign_x`)


These ^ scripts are included as reference for future repurposing of some of these pipelines here. These were all run on a cluster with a lot of memory available, and were using older versions of RECOVAR, so if one wants to repurpose them it will take some adjusting! They cannot be run directly from this repo.

The data used for these `data_generating/` scripts are from cryo-EM image datasets that are available in this [Zenodo repo](https://zenodo.org/records/18664625?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZlMWE5NmU2LTQyZmItNGJmOC05ZWQ1LTNlZWI5MGZhODgyYiIsImRhdGEiOnt9LCJyYW5kb20iOiIzZjAwM2U3Y2YyZGFhZDk5YjI5MzkxZmU2Y2NkYjhkMSJ9.XtYougOfMXs4NGlgh_734gGOJqF3cKEXv3pkeUOJT1b1nxWkOuezLg9WRQZZ2wDyB5sQtZYtiS8eGmawlMcIfg). 


Finally, see the `.toml` file included for a list of minimal dependencies for running the scripts.



