"""Module for the PiecewisePotential class.

"""

import logging

import numpy as np
from omegaconf import DictConfig

from physped.core.distribtution_approximator import GaussianApproximation
from physped.core.lattice import Lattice

log = logging.getLogger(__name__)


class PiecewisePotential:
    def __init__(self, bins: DictConfig):
        """A class for the piecewise potential.

        Creates the lattice to discretize the slow dynamics and fit the potential.

        Args:
            bins: A dictionary containing the bin edges for each dimension.
        """
        self.lattice = Lattice(bins)
        self.dist_approximation = GaussianApproximation()
        self.histogram = np.zeros(self.lattice.shape)
        self.histogram_slow = np.zeros(self.lattice.shape)
        self.parametrization = self.initialize_parametrization()

    def __repr__(self):
        return (
            f"PiecewisePotential with dimensions {self.lattice.dimensions}"
            f", fit dimensions {self.dist_approximation.fit_dimensions},"
            f"and parameters {self.dist_approximation.fit_parameters}"
        )

    def initialize_parametrization(self):
        """Initialize the potential parametrization."""
        shape_of_the_potential = self.lattice.shape + (
            len(self.dist_approximation.fit_dimensions),
            len(self.dist_approximation.fit_parameters),
        )
        return np.zeros(shape_of_the_potential) * np.nan
