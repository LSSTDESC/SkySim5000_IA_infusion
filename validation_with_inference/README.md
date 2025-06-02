# IA Model Inference with SkySim5000

This repository contains data and analysis for the validation work described
in **Harnois-Déraps & Šarčević _et al._ ([arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx))**. 
It focuses on validating the inference of the intrinsic alignment (IA) models, specifically recovering both cosmological
and IA parameters from SkySim5000 simulation data that has been infused with IA following
the recipe outlined in the paper. This work demonstrates whether the injected IA signals and cosmological parameters
can be accurately recovered through inference.


In this repository, you will find both the `CosmoSIS`-generated chains and all necessary configuration files 
to reproduce them through `CosmoSIS` sampling.

Below, we provide a brief overview of the repository structure, instructions on how to run the analysis,
and a summary of the IA models included.

The workflow is built on top of the [`CosmoSIS`](https://bitbucket.org/joezuntz/cosmosis) framework
and includes full pipeline configuration, chain processing, plotting, and comparison of IA models.

## Overview

We compare several IA models against the TATT (Tidal Alignment Tidal Torque) model and one another to evaluate their performance using 2-point weak lensing statistics. The included models are:

- **NLA** (Nonlinear Linear Alignment)
- **δNLA** (Extended Nonlinear Linear Alignment)
- **TT** (Tidal Torque)
- **δTT** (Extended Tidal Torque)
- **HOD NLA** (Halo Occupation Distribution for NLA)
- **HOD TT** (Halo Occupation Distribution for TT)
- **TATT** (Full tidal alignment + torque model)

Each IA model is run and compared against relevant targets (usually TATT, or a matching IA model), and their inference results are analyzed and visualized.

## Usage

### Requirements

- `Python ≥ 3.9`
- [`CosmoSIS`](https://bitbucket.org/joezuntz/cosmosis)
- `getdist`, `numpy`, `matplotlib`, `astropy`

### Running the Pipeline

1. Set up the `CosmoSIS` environment (e.g., activate the environment).
2. Navigate to the `runs/` directory.
3. Choose a `.ini` file corresponding to the model-target pair (e.g., `NLA_with_TATT.ini`).
4. Run `CosmoSIS`:
   ```bash
   cosmosis path_to_your_runs_directory/NLA_with_TATT.ini

### ⚠️ Important Note on Paths

**WARNING:** The paths to input files (fiducial values, 2PCF FITS, n(z), priors, etc.) are hardcoded
in the CosmoSIS `.ini` files in the `runs/` directory.
**You must edit these paths to match your local setup.**

### ⚠️ Important Note on Chains

**You do not need to rerun the chains unless you intend to modify analysis choices**  
(for example, adjusting priors or central parameter values).

The results shown in Section 5 of the paper are based on the chains already stored in the `output/chains/` directory.

You can use the `corner_plots_skysim500_ia_infusion.ipynb` notebook to load these chains and generate corner plots.

The processing and preparation of chains for GetDist are handled by the `helpers/chains_processing.py` module,  
while the plotting functionality is implemented in `helpers/plot_handlers.py`.


### Directory Structure

This setup uses the standard `CosmoSIS` cosmology library. 
Aside from the settings and configurations provided in this directory,
all other configurations follow the default `CosmoSIS` setup.

```commandline
├── helpers/ # Python modules for chain processing and plotting
│ ├── chains_processing.py # Functions to load and process CosmoSIS chains
│ ├── plot_handlers.py # Functions to customize corner plots using getdist
│ └── init.py
├── output/ # Outputs from CosmoSIS and plotting
│ ├── chains/ # Chain files organized by model name (e.g., NLA_chains/, TATT_chains/)
│ ├── corner_plots/ # Generated corner plots
│ └── sampler_outputs/ # CosmoSIS sampler outputs (log files, weights)
├── cosmosis_setup/ # Input files for CosmoSIS
│ ├── 2pcf_fits_files/ # FITS files for each model's 2PCF measurements
│ ├── fiducial_values/ # ini files with fiducial values and priors
│ ├── extra_config/ # Additional configs (n(z), scale cuts, priors)
│ └── runs/ # CosmoSIS .ini configuration files
├── corner_plots_skysim500_ia_infusion.ipynb # Jupyter notebook for generating corner plots
└── README.md # This file
```

## Analysis Choices

Priors and fiducial values are defined in `fiducial_values/` .ini files. A typical block includes:
`[cosmological_parameters]`, `[intrinsic_alignment_parameters]` and other relevant parameters such as
`[shear_calibration_parameters]` or `[wl_photoz_errors]`.

### 📋 Fiducial Values, Priors, and Truth Values of Relevant Parameters

| Parameter     | Prior Range (Fiducial Values (Prior Range) | [Outer Rim](https://arxiv.org/abs/1904.11970) Truth Value |
|---------------|--------------------------------------------|-----------------------------------------------------------|
| $\Omega_\mathrm{m}$ | 0.3 (0.1 – 0.5)                            | 0.2648                                                    |
| $S_8$             | 0.7525 (0.6 – 0.9)                         | 0.7525                                                    |
| $A_\mathrm{IA}$   | 1.0 (-5.0 – 5.0)                           | 1.0                                                       |
| $C_2$             | 1.0 (-5.0 – 5.0)                           | 0.0                                                       |
| $b_\mathrm{TA}$   | 1.0 (0.0 – 5.0)                            | 1.0                                                       |


### 📋 **Table of Varied Parameters**

| Model      | File Key               | Cosmological Parameters       | IA Parameters Varied            |
|------------|------------------------|-------------------------------|---------------------------------|
| **NLA**    | NLA_with_NLA           | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$                 |
|            | NLA_with_TATT          | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
|            | NLA_with_TATT_w        | $\Omega_\mathrm{m}$, $S_8$, $w$ | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
| **δNLA**   | deltaNLA_with_deltaNLA | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $b_\mathrm{TA}$ |
|            | deltaNLA_with_TATT     | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
| **TT**     | TT_with_TT             | $\Omega_\mathrm{m}$, $S_8$    | $C_2$                           |
|            | TT_with_TATT           | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
| **δTT**    | deltaTT_with_TT        | $\Omega_\mathrm{m}$, $S_8$    | $C_2$                           |
|            | deltaTT_with_deltaTT   | $\Omega_\mathrm{m}$, $S_8$    | $C_2$, $b_\mathrm{TA}$          |
|            | deltaTT_with_TATT      | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
| **HOD NLA**| HOD_NLA_with_deltaNLA  | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $b_\mathrm{TA}$ |
|            | HOD_NLA_with_TATT      | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |
| **HOD TT** | HOD_TT_with_TT         | $\Omega_\mathrm{m}$, $S_8$    | $C_2$                           |
|            | HOD_TT_with_deltaTT    | $\Omega_\mathrm{m}$, $S_8$    | $C_2$, $b_\mathrm{TA}$          |
|            | HOD_TT_with_TATT       | $\Omega_\mathrm{m}$, $S_8$    | $A_\mathrm{IA}$, $C_2$, $b_\mathrm{TA}$ |


## Generating the Results from the Paper

To generate the results presented in the paper (_Section 5: Validation with full Inference_), 
you can run the notebook  `corner_plots_skysim500_ia_infusion.ipynb`.

To list the available samples and their keys, you can use the following code snippet:


```python
import helpers.chains_processing as cp

samples = cp.samples
for key, value in samples.items():
    print(key)
    if isinstance(value, dict):
        for nested_key in value:
            print(f"  {nested_key}")
```

In the sme manner, you can access the labels, truth values, and predifend color schemes for the corner plots 
All of them are dictionaries with keys corresponding to the file names in the `output/chains/` directory.

```python
import helpers.plot_handlers as ph
import helpers.chains_processing as cp

truth_values = cp.truth_values  # dictionary with truth values for each model
labels = ph.labels  # dictionary with labels for each model
colors = ph.get_colors()  # dictionary with colors for each model
```
