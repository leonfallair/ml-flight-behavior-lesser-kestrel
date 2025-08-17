# Predicting flight behavior of lesser kestrel using supervised machine learning

* This project contains a supervised machine learning approach to predict three behavior classes (resting/flying(searching)/migrating) based primarily on spatial and temporal features. In this repo the notebook, environment and dataset is provided, as needed to run the code itself.

## Motivation and General information

* Overall, understanding animal behavior can potentially help improve animal conservation. In the case of a migratory bird like the lesser kestrel, it can be useful to analyze flight behavior and identify potential external influences. A supervised classification approach was chosen here in order to gain a deeper analytical overview.

* Beyond this project, other machine learning methods could also be used, and more environmental factors could be incorporated to potentially get more extensive results. Unfortunately, in this case with the large dataset of the lesser kestrel, many variable queries did not work in the Movebank.

## Getting started

* The code is entirely contained in a Jupyter notebook and was developed with Python in VS Code. The working environment is managed with the [Miniforge](https://github.com/conda-forge/miniforge) distribution using **mamba/conda**, and the complete environment configuration is provided as a `.yaml` file in this repository for easy reproducibility.

### Dependencies

Before running the notebook, you need to have:
- Python 3.11.*
- [Miniforge](https://github.com/conda-forge/miniforge), Anaconda/Miniconda or other preferred environment
- JupyterLab or Jupyter Notebook
- The required packages listed in the `environment.yaml`

### Installing and executing programm

After following the steps of installing the miniforge dsitribution
1. Create the environment from the provided `.yaml` file:
    ```
    [mamba or conda] env create -f environment.yaml
    ```
2.  Activate the environment:
    ```
    [mamba or conda] activate tracking_env
    ```
3. Open the Jupyter Notebook in VS Code or JupyterLaband select right kernel (tracking_env) and run all cells

## Data source and license

This project uses data from the Movebank:

**Dataset:** [Lesser Kestrel trackin data](https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study2398637362)

**Download Dataset (40 MB):** https://heibox.uni-heidelberg.de/f/0eada8bfd7f64dfca8e3/?dl=1

**Source:** [www.movebank.org](https://www.movebank.org/cms/webapp?gwt_fragment=page=search_map)

**License from source:** [![License: CC BY-NC](https://img.shields.io/badge/License-CC%20BY--NC-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


## References

* Bustamante, J. (2025). (EBD) Lesser Kestrel (Falco naumanni) Senegal, MERCURIO-SUMHAL. Movebank Data Repository. Available: https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study2398637362. Licensed under CC BY-NC.
