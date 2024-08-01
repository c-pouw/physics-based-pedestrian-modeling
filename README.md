# Data-driven physics-based modeling of pedestrian dynamics
<p align="center">
    <a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/" alt="Repository">
	   <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" /></a>
    <a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml" alt="Unit Tests">
       <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml/badge.svg" /></a>
	<a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/integration-tests.yaml" alt="Integration Tests">
       <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/integration-tests.yaml/badge.svg" /></a>
	<a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/pages/pages-build-deployment" alt="pages-build-deployment">
	   <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/pages/pages-build-deployment/badge.svg" /></a>
    <a href="https://pypi.python.org/pypi/physics-based-pedestrian-modeling" alt="pypi version">
       <img src="https://img.shields.io/pypi/v/physics-based-pedestrian-modeling.svg" /></a>
    <a href="#">
       <img src="https://img.shields.io/pypi/pyversions/physics-based-pedestrian-modeling" alt="PyPI - Python Version" /></a>
    <a href="https://github.com/psf/black">
       <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" /></a>
	<a href="https://colab.research.google.com/github/c-pouw/physics-based-pedestrian-modeling/blob/master/usage_notebooks/physped_quick_start.ipynb">
	   <img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
	<a href="https://arxiv.org/abs/2407.20794">
	   <img src="https://img.shields.io/badge/arXiv-2407.20794-b31b1b.svg?style=flat" alt="arXiv" /></a>
</p>

# Project Overview

Python package to create physics-based pedestrian models from pedestrian trajectory measurements. This package is an implementation of the data-driven generalized pedestrian model presented in:

Pouw, C. A. S., van der Vleuten, G., Corbetta, A., & Toschi, F. (2024). Data-driven physics-based modeling of pedestrian dynamics. Preprint, https://arxiv.org/abs/2407.20794


## Documentation

* Documentation: https://c-pouw.github.io/physics-based-pedestrian-modeling.


## Usage Notebooks
We provide the following usage notebook on Google Colab:

<h2 align="left" style="vertical-align: middle;">
    <a href="https://colab.research.google.com/github/c-pouw/physics-based-pedestrian-modeling/blob/master/usage_notebooks/physped_quick_start.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a><sup><sub> - Quick-start with the generalized pedestrian model. </sub></sup> <br>
</h2>

The notebook can be used to create a model for the following environments:

- Walking paths in a narrow corridor.
- Intersecting walking paths.
- Walking paths on a train station platform.

## Using the CLI

Install the package from source

```bash
git clone https://github.com/c-pouw/physics-based-pedestrian-modeling.git
cd physics-based-pedestrian-modeling
pip install -e .
```

Run the main processing script for one of the available parameter files (listed below)

```bash
python physped/main.py params=PARAM_NAME
```

### Parameter Files
Parameter configurations are handled by ![Hydra](https://github.com/facebookresearch/hydra). Default parameter files are provided for the following cases:
* **single_paths:** Trajectories in a narrow corridor.
* **intersecting_paths:** Trajectories intersecting in the origin.
* **asdz_pf12:** Complex trajectories on the Amsterdam Zuid train platform 1 and 2.

# License
* Free software: 3-clause BSD license
