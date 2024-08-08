"""Module for the PiecewisePotential class.

"""

import logging
from pprint import pformat
from typing import Dict

import numpy as np
from omegaconf import OmegaConf

from physped.utils.functions import get_bin_middle

# from dataclasses import dataclass


log = logging.getLogger(__name__)


# TODO: Devide this class into two classes: one for the potential and one for the lattice


class PiecewisePotential:
    def __init__(self, bins: Dict[str, np.ndarray]):
        """A class for the piecewise potential.

        Creates the lattice to discretize the slow dynamics and fit the potential.

        Args:
            bins: A dictionary containing the bin edges for each dimension.
        """
        self.bins = bins
        self.bin_centers = {key: get_bin_middle(bins[key]) for key in bins}
        self.grid_shape = tuple(len(self.bin_centers[key]) for key in self.bin_centers)
        self.dimensions = tuple(self.bins.keys())
        self.histogram = np.zeros(self.grid_shape)
        self.histogram_slow = np.zeros(self.grid_shape)
        self.fit_dimensions = ("x", "y", "u", "v")
        self.parameters = ["mu", "sigma"]

        # Initialize potential paramterization
        shape_of_the_potential = self.grid_shape + (len(self.fit_dimensions), len(self.parameters))
        self.parametrization = np.zeros(shape_of_the_potential) * np.nan
        self.cell_volume = self.compute_cell_volume()

    def __repr__(self):
        return f"PiecewisePotential(bins={pformat(OmegaConf.to_container(self.bins, resolve=True), depth=1)})"

    def compute_cell_volume(self) -> np.ndarray:
        """Compute the volume of each cell in the lattice."""
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

    # TODO: turn this into methods
    # def bin_centers(self):
    #     return {key: get_bin_middle(self.bins[key]) for key in self.bins}

    # def grid_shape(self):
    #     return tuple(len(self.bin_centers[key]) for key in self.bin_centers)
