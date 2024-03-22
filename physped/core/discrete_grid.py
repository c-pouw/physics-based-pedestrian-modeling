"""Discrete grid class"""

from typing import Dict
import logging

import numpy as np

from physped.utils.functions import get_bin_middle

log = logging.getLogger(__name__)


class DiscreteGrid:
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
        Initialize a new CreateDiscreteGrid object.

        Parameters:
        - bins (Dict[str, np.ndarray]): A dictionary of bin edges for each dimension of the grid.
        """
        self.bins = {key: bins[key] for key in bins}
        self.bin_centers = {key: get_bin_middle(bins[key]) for key in bins}
        self.grid_shape = tuple(len(self.bin_centers[key]) for key in self.bin_centers)
        self.dimensions = tuple(self.bin_centers.keys())
        self.histogram = np.zeros(self.grid_shape)
        self.histogram_slow = np.zeros(self.grid_shape)
        self.no_fit_params = 8  # (mu, sigma) for ('x','y','u','v')
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

        self.fit_params = np.zeros(self.grid_shape + (self.no_fit_params,))
