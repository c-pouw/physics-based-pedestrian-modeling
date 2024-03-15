"""Langevin model class."""

import os
import sys
import logging
import pickle

# from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from physped.utils.functions import (
    pol2cart,
    cart2pol,
    digitize_values_to_grid,
)
from physped.core.discrete_grid import DiscreteGrid
from physped.core.langevin_model import LangevinModel
from physped.io.readers import read_parameter_file


log = logging.getLogger("mylog")


def random_uniform_value_in_bin(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Generate random uniform values within each bin.

    Args:
        values (np.ndarray): Array of bin indices.
        bins (np.ndarray): Array of bin edges.

    Returns:
        np.ndarray: Array of random uniform values within each bin.
    """
    left = bins[values]
    right = bins[values + 1]
    return np.random.uniform(left, right)


# def get_initial_values_from_trajectories(df: pd.DataFrame) -> np.ndarray:
#     """
#     Get initial values for a simulation from a DataFrame of trajectories.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame of trajectories.

#     Returns:
#     - A NumPy array of initial values for the simulation.
#     """
#     pidlist = df['Pid'].unique()
#     X_0 = df.loc[df['Pid'] == np.random.choice(pidlist), ['xf', 'yf', 'uf', 'vf', 'xs', 'ys', 'us', 'vs']].values[0]
#     return X_0


def convert_grid_indices_to_coordinates(
    grids: DiscreteGrid, X_0: np.ndarray
) -> np.ndarray:
    """
    Convert grid indices to Cartesian coordinates.

    Parameters:
    - grids (Grids): The Grids object containing the grid definitions.
    - X_0 (List[int]): A list of grid indices.

    Returns:
    - A list of Cartesian coordinates.
    """
    # TODO: Add some noise within the bin.
    # xf, yf, rf, thetaf, k = (grids.bins[dim][X_0[:,i]] for i, dim in enumerate(grids.dimensions))
    xf = random_uniform_value_in_bin(X_0[:, 0], grids.bins["x"])
    yf = random_uniform_value_in_bin(X_0[:, 1], grids.bins["y"])
    rf = random_uniform_value_in_bin(X_0[:, 2], grids.bins["r"])
    thetaf = random_uniform_value_in_bin(X_0[:, 3], grids.bins["theta"])
    k = grids.bins["k"][X_0[:, 4]]

    uf, vf = pol2cart(rf, thetaf)
    return np.array([xf, yf, uf, vf, k]).T


def sample_from_ndarray(origin_histogram: np.ndarray, N_samples: int = 1) -> np.ndarray:
    """
    Sample origin positions from an initial position heatmap.

    Parameters:
    - origin_histogram (np.ndarray): The initial position heatmap.
    - N_samples (int): The number of samples to generate. Default is 1.

    Returns:
    - A tuple of NumPy arrays representing the sampled origin positions.
    """
    flat_origin_histogram = origin_histogram.ravel()
    indices1d = np.random.choice(
        a=range(len(flat_origin_histogram)),
        size=N_samples,
        replace=True,
        p=flat_origin_histogram / np.sum(flat_origin_histogram),
    )
    return np.array(np.unravel_index(indices1d, origin_histogram.shape)).T
