"""Module for the PiecewisePotential class.

"""

import logging

import numpy as np
from omegaconf import DictConfig

from physped.core.distribution_approximator import DistApproximation
from physped.core.lattice import Lattice

log = logging.getLogger(__name__)


class PiecewisePotential:
    def __init__(self, lattice: Lattice, dist_approximation: DistApproximation):
        """A class for the piecewise potential.

        Creates the lattice to discretize the slow dynamics and fit the potential.

        Args:
            bins: A dictionary containing the bin edges for each dimension.
        """
        self.lattice = lattice
        self.dist_approximation = dist_approximation
        self.histogram = np.zeros(self.lattice.shape)
        self.histogram_slow = np.zeros(self.lattice.shape)
        self.initialize_parametrization()

    def __repr__(self):
        return (
            f"PiecewisePotential with dimensions {self.lattice.dimensions}"
            f", fit dimensions {self.dist_approximation.fit_dimensions},"
            f"and parameters {self.dist_approximation.fit_parameters}"
        )

    def initialize_parametrization(self):
        """Initialize the potential parametrization.

        to initialize the potential parametrization with the following shape:
        (lattice_shape, len(fit_dimensions), len(fit_parameters))
        """
        shape_of_the_potential = self.lattice.shape + (
            len(self.dist_approximation.fit_dimensions),
            len(self.dist_approximation.fit_parameters),
        )
        self.parametrization = np.zeros(shape_of_the_potential) * np.nan

    def reparametrize_to_curvature(self, config: DictConfig):
        """Reparametrize the potential.

        From (mu, var) to (mu, curvature).

        Args:
            config: The configuration.

        Raises:
            ValueError: If the fit parameters are not mu and sigma
        """
        if self.dist_approximation.fit_parameters != ("mu", "sigma"):
            raise ValueError("The fit parameters should be mu and sigma.")

        var = config.params.model.sigma**2
        xvar = self.parametrization[..., 0, 1]
        yvar = self.parametrization[..., 1, 1]
        uvar = self.parametrization[..., 2, 1]
        vvar = self.parametrization[..., 3, 1]

        self.parametrization[..., 0, 1] = uvar / (2 * xvar)
        self.parametrization[..., 1, 1] = vvar / (2 * yvar)
        self.parametrization[..., 2, 1] = var / (4 * uvar)
        self.parametrization[..., 3, 1] = var / (4 * vvar)
        self.dist_approximation.fit_parameters = ["mu", "curvature"]
