"""Module for the Latice class.

"""

import logging
from pprint import pformat
from typing import Dict, Tuple

import numpy as np
from omegaconf import OmegaConf

from physped.utils.functions import get_bin_middle

log = logging.getLogger(__name__)


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
        # self.cell_volume = self.compute_cell_volume()

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
