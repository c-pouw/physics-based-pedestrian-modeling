# Data-driven physics-based modeling of pedestrian dynamics
<p align="center">
    <a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/" alt="Repository">
	   <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" /></a>
	<a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/integration-tests.yaml" alt="Integration Tests">
	   <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/integration-tests.yaml/badge.svg" /></a>
    <a href="https://c-pouw.github.io/physics-based-pedestrian-modeling" alt="docs">
	   <img src="https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/docs.yml/badge.svg" /></a>
    <a href="https://c-pouw.github.io/physics-based-pedestrian-modeling/reports/junit/report.html?sort=result">
	   <img src="https://c-pouw.github.io/physics-based-pedestrian-modeling/reports/junit/tests-badge.svg?dummy=8484744" alt="tests" /></a>
    <a href="https://c-pouw.github.io/physics-based-pedestrian-modeling/reports/coverage/index.html?">
	   <img src="https://c-pouw.github.io/physics-based-pedestrian-modeling/reports/coverage/coverage-badge.svg?dummy=8484744" alt="coverage" /></a>
    <a href="https://pypi.python.org/pypi/physics-based-pedestrian-modeling" alt="pypi version">
       <img src="https://img.shields.io/pypi/v/physics-based-pedestrian-modeling.svg" /></a>
    <a href="#">
       <img src="https://img.shields.io/pypi/pyversions/physics-based-pedestrian-modeling" alt="PyPI - Python Version" /></a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
       <img src="https://img.shields.io/badge/License-BSD%203--Clause-orange.svg" alt="Licence" /></a>
    <a href="https://github.com/psf/black">
       <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" /></a>
</p>


Python package to create physics-based pedestrian models from pedestrian trajectory measurements. This package is an implementation of the data-driven generalized pedestrian model presented in:

> Pouw, C. A. S., van der Vleuten, G., Corbetta, A., & Toschi, F. (2024). Data-driven physics-based modeling of pedestrian dynamics. Phys. Rev. E 110 (6 Dec. 2024), p. 064102. DOI: [10.1103/PhysRevE.110.064102](https://doi.org/10.1103/PhysRevE.110.064102).

**Abstract.** We introduce a data-driven physics-based generalized
Langevin model that allows robust and generic modeling of pedestrian
behavior across any environment where extensive pedestrian trajectory
data is available. Our model effectively captures the complex
interplay between the deterministic movements and stochastic
fluctuations associated with walking.

<h3 align="center" style="vertical-align: middle;">
	<a href="https://journals.aps.org/pre/accepted/ec075Ra2H081202d17c11029a2e965c33e4471521">
	   <img src="https://img.shields.io/badge/PRE-Manuscript-b31b1b?style=for-the-badge" alt="PRE" /></a>
	<a href="https://github.com/c-pouw/physics-based-pedestrian-modeling/" alt="Repository"><img src="https://img.shields.io/badge/Github-Software-%23181717?style=for-the-badge" /></a>
    <a href="https://doi.org/10.5281/zenodo.13784588">
	   <img src="https://img.shields.io/badge/Zenodo-Dataset-%231682D4?style=for-the-badge" alt="Zenodo dataset"></a>
    <a href="https://colab.research.google.com/github/c-pouw/physics-based-pedestrian-modeling/blob/master/usage_notebooks/physped_quick_start.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Demonstration-%23F9AB00?style=for-the-badge"></a>
</h3>

<!-- index.rst homepage end -->
## Documentation

* Documentation: https://c-pouw.github.io/physics-based-pedestrian-modeling.

<!-- index.rst usage start -->

# Usage Notebook
This Goolge Colab notebook can be used to create pedestrians models without the need to install the package or download any data. The notebook can be used with all the environments discussed below, or can be easily adapted for any other data set.

* Notebook: https://colab.research.google.com/github/c-pouw/physics-based-pedestrian-modeling/blob/master/usage_notebooks/physped_quick_start.ipynb

# Installation

You can install the package from PyPI

```bash
pip install physics-based-pedestrian-modeling
```

# Using the CLI
Run the main processing script for one of the available environments by overwriting the `params` variable with the configuration filename of the environment. The configuration filenames associated to every environment are specified below. These parameter configurations are handled by Hydra, see their documentation for more details ![Hydra](https://github.com/facebookresearch/hydra).

```bash
physped_cli params=CONFIGURATION_FILE_NAME
```

Similarly, we can overwrite all the other parameter directly from the command line. For instance, if we want to process the narrow corridor trajectories with a different noice intensity, e.g. sigma=0.7, we can simply run

```bash
physped_cli params=narrow_corridor params.model.sigma=0.7
```

Creating the model for multiple parameter values can be achieved by adding `-m` and listing the variables. For example

```bash
physped_cli -m params=narrow_corridor params.model.sigma=0.5,0.7,0.9
```

# Available environments

Every environment discussed in the paper that relies a on public data set can be modeled using the cli by overwriting the 'params' variable with one of the following configuration filenames:

## Narrow corridor
The dynamics of a single pedestrian walking undisturbed through a narrow corridor is possibly one of the simplest conceivable dynamics; an almost one-dimensional movement solely confined by two parallel corridor walls.

To model this environment we use the public pedestrian trajectory dataset: [Crowdflow – diluted pedestrian dynamics in the Metaforum building of Eindhoven University of Technology](https://data.4tu.nl/datasets/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/1).

Configuration filename: **narrow_corridor**

## Intersecting walking paths
Trajectories of intersecting walking paths. We synthesize these from the narrow corridor trajectory data set for which half of the trajectories are rotated by 90 degrees. Note that these trajectories were measured in the dilute limit, consequently there is no interaction between them.

Configuration filename: **intersecting_paths**

## Train station platforms
Complex dynamics of pedestrians walking across real-life train platforms.

### Eindhoven train station track 3-4.
Configuration filename: **eindhoven_pf34**.

To model this environment we use the public pedestrian trajectory dataset: [Data-driven physics-based modeling of pedestrian dynamics - dataset: Pedestrian trajectories at Eindhoven train station](https://doi.org/10.5281/zenodo.13784587)

### Amsterdam Zuid train station track 1-2.
Configuration filename: **asdz_pf12**.

To model this environment we use the public pedestrian trajectory dataset: [Trajectory data Amsterdam Zuid (track 1-2) underlying the PhD-thesis: Mind your passenger! The passenger capacity of platforms at railway stations in the Netherlands](https://doi.org/10.4121/20683062)

<!-- index.rst usage end -->

# License
* Free software: 3-clause BSD license

# Contact
<p align="left">
	<a href="https://github.com/c-pouw" alt="Github-profile">
		<img src="https://img.shields.io/badge/Github-black?style=for-the-badge&logo=github&logoColor=white"/></a>
	<a href="mailto:c.a.s.pouw@tue.nl" alt="Email">
		<img src="https://img.shields.io/badge/Email-%230008a1?style=for-the-badge&logo=gmail&logoColor=white" /></a>
    <a href="https://www.linkedin.com/in/caspouw/" alt="LinkedIn">
	   <img src="https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=fff&style=for-the-badge" /></a>
    <a href="https://scholar.google.com/citations?user=JoBuJXgAAAAJ&hl=nl&oi=ao" alt="Google Scholar Badge">
	   <img src="https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge" /></a>
    <a href="https://www.researchgate.net/profile/Caspar-Pouw-2" alt="ResearchGate">
	   <img src="https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge" /></a>
    <a href="https://orcid.org/0000-0002-3041-4533" alt="ORCID">
	   <img src="https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge" /></a>
</p>
