# Data-driven physics-based modeling of pedestrian dynamics
<p align="center">
    <a href="https://img.shields.io/badge/build-passing-brightgreen?logo=github" alt="build passing">
       <img src="https://img.shields.io/badge/build-passing-brightgreen?logo=github" /></a>
    <a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml" alt="build status">
       <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml/badge.svg" /></a>
    <a href="https://pypi.python.org/pypi/physics-based-pedestrian-modeling" alt="pypi version">
       <img src="https://img.shields.io/pypi/v/physics-based-pedestrian-modeling.svg" /></a>
    <a href="#">
       <img src="https://img.shields.io/pypi/pyversions/physics-based-pedestrian-modeling" alt="PyPI - Python Version" /></a>
    <a href="https://github.com/psf/black">
       <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" /></a>
    <a href="https://doi.org/10.1016/j.trc.2023.104468">
       <img src="https://img.shields.io/badge/DOI-10.1016/j.trc.2023.104468-blue.svg" alt="doi paper" /></a>
</p>

# Project Overview

Python package to create physics-based pedestrian models from pedestrian trajectory measurements. This package is an implementation of the data-driven generalized pedestrian model presented in:

Pouw, C. A. S., van der Vleuten, G., Corbetta, A., & Toschi, F. (2024). Data-driven physics-based modeling of pedestrian dynamics. To appear xx.


# Getting started

Install the package from PyPI

```bash
pip install physics-based-pedestrian-modeling
```

Download the configuration files into your working directory.

Run the main script for one of the available parameter files (listed below)

```bash
python main.py params=single_paths
```

# Features
### Preprocessing of trajectories
Calculate slow dynamics

### Learn potential from the preprocessed trajectories
Learn the potential

### Simulate new trajectories using the learned potential
Simulate new trajectories

## Parameter Files
Configuration of parameter files is handled by ![Hydra](https://github.com/facebookresearch/hydra). Default parameter files are provided for the following cases:
* **single_paths:** Trajectories in a narrow corridor.
* **parallel_paths:** Trajectories in a wide corridor.
* **curved_paths_synthetic:** Trajectories along a closed elliptical path.
* **intersecting_paths:** Trajectories intersecting in the origin.
* **station_paths:** Complex trajectories in a train station.

## Featured Notebooks
A couple of usage notebooks are available for the following cases:
* Narrow corridor paths
* Station paths
* User input paths

# Documentation
* Documentation: https://c-pouw.github.io/physics-based-pedestrian-modeling.

# License
* Free software: 3-clause BSD license
