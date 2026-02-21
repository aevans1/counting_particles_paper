# Data Generation and Analysis Scripts for *Counting particles in cryo-electron microscopy may result in incorrect population estimates*  
Evans, 2025

This repository contains the data generation and analysis scripts required to reproduce the results presented in the bioRxiv preprint:

> **Counting particles in cryo-electron microscopy may result in incorrect population estimates**  
> https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1

The repository is organized into three main sections:

1. **Data generation**  
   Scripts used to generate the precomputed datasets available on Zenodo:  
   https://zenodo.org/records/18724862

2. **Reweighting / Deconvolution analysis**  
   Scripts implementing ensemble reweighting and deconvolution methods used in the manuscript (primarily `compute_weights.py` and `utils.py`).

3. **Figure generation**  
   Scripts used to generate manuscript figures from precomputed data.  
   (LE: is this in zenodo or data in this repo?)

---

# Installation

The `pyproject.toml` file specifies the minimal dependencies required to run the analysis and figure-generation scripts.

To install using `venv`:

```bash
python -m venv my_venvs/counting_particles_venv
source my_venvs/counting_particles_venv/bin/activate
cd counting_particles_paper/
pip install .
```

## 1. Data Generation 

The `data_generation_scripts` files are included as reference for future repurposing of some of these pipelines. These are the scripts used to generate the `data` available in Zenodo at  
https://zenodo.org/records/18724862. (LE: please explain what is available in data repo??)

Note that these cannot run without installing several external cryo-EM software packages, including RECOVAR, cryoDRGN, and [cryoSPARC](https://cryosparc.com/).  
(LE: include references please).

However, one does not need to re-generate the data in order to reproduce the results of the manuscript, since it is already available.  
(LE: where? zenodo or here).

### 1.1 Synthetic Data Generation

More specifically, we generate the synthetic data with:

- `make_x` script: simulates the synthetic datasets in RECOVAR, where `_x` denotes a particular dataset.  
  (LE: please add details, output simulated images and star file?)  
  It also allows adding noise to each dataset.

### 1.2 Likelihood and Misclassification Computation

Then, we compute the likelihoods and misclassification assignments in RECOVAR using:

- `assign_x`: computes likelihoods and misclassification given a `.star` file as input.  
  (LE: please add details).

The outputs are stored  
(LE: what and where?)  

and are used as inputs in the following section.

We also note that these scripts were run on a cluster with substantial memory resources and were executed using older versions of RECOVAR. Therefore, if one wishes to repurpose them, some adjustments will likely be necessary. They cannot be run directly from this repository without additional setup.

## 2. Reweighting / Deconvolution Analysis Scripts

After generating (or downloading) the data, we obtain the inputs required for the `compute_weights` script.  
(LE: add the names of files)

The `compute_weights` script:

- Computes probability weights from a likelihood matrix input  
- Performs ensemble reweighting using a multiplicative gradient approach  
  (LE: please clarify and correct)
- Alternatively performs deconvolution using the [`RECOVAR` library](https://github.com/ma-gilles/recovar)

The full computational pipelines for these procedures are described in the manuscript.  
(LE: what sections?)

This script uses helper functions defined in `utils.py`, which contains basic utilities for both deconvolution and ensemble reweighting.

(LE: explain the output)

---

## 3. Figure Generation

Each script titled:

- `figure_x`

computes **Figure X** from the manuscript.

To recreate the figure exactly as in the manuscript, you will need LaTeX installed to enable `mpl.rcParams['text.usetex'] = True`
     in the matplotlib settings. If you don't have LaTeX installed, you can keep `mpl.rcParams['text.usetex'] = False` but
     may have to adjust some font sizes.


The output figures are saved in the `figures` folder.


