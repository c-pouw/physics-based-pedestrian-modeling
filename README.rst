=========================================================
Data-driven physics-based modeling of pedestrian dynamics
=========================================================
.. image:: https://img.shields.io/pypi/v/physics-based-pedestrian-modeling.svg
   :target: https://pypi.python.org/pypi/physics-based-pedestrian-modeling
.. image:: https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/c-pouw/physics-based-pedestrian-modeling/actions/workflows/testing.yml
.. image:: https://img.shields.io/badge/build-passing-brightgreen?logo=github
   :alt: Static Badge
Project Overview
-----------------------------------------------------------------------

Python package to create physics-based pedestrian models from pedestrian trajectory measurements. This package is an implementation of the data-driven generalized pedestrian model as presented in:

Pouw, C. A. S., van der Vleuten, G., Corbetta, A., & Toschi, F. (2024). Data-driven physics-based modeling of pedestrian dynamics. To appear xx.


Getting started
-----------------------------------------------------------------------

Install the package from PyPI

.. code-block:: bash

   pip install physped

Run the main script for one of the available parameter files (listed below)

.. code-block:: bash

   python main.py params=single_paths

Features
-----------------------------------------------------------------------
- Read trajectory data set
- Calculate slow dynamics
- Learn potential from the trajectories
- Simulate new trajectories with the learned potential

Parameter Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **single_paths:** Trajectories in a narrow corridor.
- **parallel_paths:** Trajectories in a wide corridor.
- **curved_paths_synthetic:** Trajectories along a closed elliptical path.
- **intersecting_paths:** Trajectories intersecting in the origin.
- **station_paths:** Complex trajectories in a train station.

Featured Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Usage notebook for narrow corridor paths
- Usage notebook for station paths
- Usage notebook for custom paths

License
-----------------------------------------------------------------------
* Free software: 3-clause BSD license


Documentation
-----------------------------------------------------------------------
* Documentation: (COMING SOON!) https://c-pouw.github.io/physics-based-pedestrian-modeling.
