"""Discrete grid class"""

import logging
from typing import Dict

import numpy as np

from physped.utils.functions import get_bin_middle

# from dataclasses import dataclass


log = logging.getLogger(__name__)


class PiecewisePotential:
    # ? Should this be a (data)class? Possibly just a numpy ndarray with some metadata?
    """
    A class for creating a discrete grid based on a set of bin edges.

    Attributes:
    - bins (Dict[str, np.ndarray]): A dictionary of bin edges for each dimension of the grid.
    - bin_centers (Dict[str, np.ndarray]): A dictionary of bin centers for each dimension of the grid.
    - grid_shape (tuple): The shape of the grid.
    - grid (np.ndarray): The grid of discrete values.
    """

    def __init__(self, bins: Dict[str, np.ndarray]):
        """
        Initialize a piecewise potential object.

        Parameters:
        - bins (Dict[str, np.ndarray]): A dictionary of bin edges for each dimension of the grid.
        """
        self.bins = bins
        self.bin_centers = {key: get_bin_middle(bins[key]) for key in bins}
        self.grid_shape = tuple(len(self.bin_centers[key]) for key in self.bin_centers)
        self.dimensions = tuple(self.bins.keys())
        self.histogram = np.zeros(self.grid_shape)
        self.histogram_slow = np.zeros(self.grid_shape)
        self.fit_dimensions = ("x", "y", "u", "v")
        self.fit_param_names = [
            "xmu",
            "xvar",
            "ymu",
            "yvar",
            "umu",
            "uvar",
            "vmu",
            "vvar",
        ]
        self.no_fit_params = len(self.fit_param_names)  # (mu, sigma) for ('x','y','u','v')
        # Initialize potential grid
        self.fit_params = np.zeros(self.grid_shape + (self.no_fit_params,))

    # TODO: turn this into methods
    # def bin_centers(self):
    #     return {key: get_bin_middle(self.bins[key]) for key in self.bins}

    # def grid_shape(self):
    #     return tuple(len(self.bin_centers[key]) for key in self.bin_centers)
