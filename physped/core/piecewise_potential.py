"""Module for the PiecewisePotential class.

"""

import logging
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, Dict, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm

from physped.utils.functions import get_bin_middle

# from dataclasses import dataclass


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


class Lattice:
    def __init__(self, bins: Dict[str, np.ndarray]):
        """A class for the lattice.

        Args:
            bins: A dictionary containing the bin edges for each dimension.
        """
        self.bins = bins
        self.dimensions = tuple(bins.keys())
        self.bin_centers = self.get_bin_centers()
        self.shape = self.get_lattice_shape()
        self.cell_volume = self.compute_cell_volume()

    def __repr__(self):
        return f"Lattice(bins={pformat(OmegaConf.to_container(self.bins, resolve=True), depth=1)})"

    def get_bin_centers(self) -> Dict[str, np.ndarray]:
        """Return the middle of the input bins.

        Returns:
            The middle of the input bins.
        """
        return {key: get_bin_middle(self.bins[key]) for key in self.bins}

    def get_lattice_shape(self) -> Tuple[int]:
        """Return the shape of the lattice.

        Returns:
            The shape of the lattice.
        """
        return tuple(len(self.bin_centers[key]) for key in self.bin_centers)

    def compute_cell_volume(self) -> np.ndarray:
        """Compute the volume of each cell in the lattice.

        Returns:
            The volume of each cell in the lattice.
        """
        dx = np.diff(self.bins["x"])
        dy = np.diff(self.bins["y"])
        dr = np.diff(self.bins["r"])
        r = self.bin_centers["r"]
        dtheta = np.diff(self.bins["theta"])
        dk = np.diff(self.bins["k"])

        i, j, k, l, m = np.meshgrid(
            np.arange(len(self.bins["x"]) - 1),
            np.arange(len(self.bins["y"]) - 1),
            np.arange(len(self.bins["r"]) - 1),
            np.arange(len(self.bins["theta"]) - 1),
            np.arange(len(self.bins["k"]) - 1),
            indexing="ij",
        )

        # return the volume for each cell using broadcasting
        return dx[i] * dy[j] * r[k] * dr[k] * dtheta[l] * dk[m]


@dataclass
class DistApproximation:
    """A class for the distribution approximation of the potential."""

    fit_dimensions: Tuple[str, ...]
    fit_parameters: Tuple[str, ...]
    function: Callable


class GaussianApproximation(DistApproximation):
    def __init__(self):
        predefined_kwargs = {"fit_dimensions": ("x", "y", "u", "v"), "fit_parameters": ("mu", "sigma"), "function": norm.fit}
        super().__init__(**predefined_kwargs)
