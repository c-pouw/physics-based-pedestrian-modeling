"""Functions for the analysis of the data."""

from typing import Tuple
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)


def _create_folderpath(params: dict) -> Path:
    """
    OBSOLETE
    Create a folder path for a model based on the specified parameters.

    Parameters:
    - parameters (dict): A dictionary of parameters for the model.

    Returns:
    - A string representing the file path for the model.
    """
    #### TODO: Create folderpath directly in the config file
    gridname = f"{params['env_name']}-tau{params['taux']}-sigma{params['sigma']}"
    return Path.cwd() / "data" / "models" / gridname


def create_folder_if_not_exists(folderpath: Path) -> None:
    """
    Create a folder at the specified path if it does not exist.

    params:
    - folderpath (str): The path to the folder to create.

    Returns:
    - None
    """
    ### TODO: change function name to ensure_folder_exists
    if not folderpath.exists():
        folderpath.mkdir()
        log.info("Folder created at %s", folderpath)


def cart2pol(x: float, y: float) -> tuple:
    """Convert cartesian coordinates to polar coordinates."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho: float, phi: float) -> tuple:
    """Convert polar coordinates to cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def get_slice_of_multidimensional_matrix(a, slice_x, slice_y, slice_theta, slice_r):
    """Get slice."""
    for slice_range in [slice_x, slice_y, slice_theta, slice_r]:
        if slice_range[0] > slice_range[1]:
            print("Slice values must be ascending.")
    sl0 = np.array(range(slice_x[0], slice_x[1])).reshape(-1, 1, 1, 1) % a.shape[0]
    sl1 = np.array(range(slice_y[0], slice_y[1])).reshape(1, -1, 1, 1) % a.shape[1]
    sl2 = np.array(range(slice_theta[0], slice_theta[1])).reshape(1, 1, -1, 1) % a.shape[2]
    sl3 = np.array(range(slice_r[0], slice_r[1])).reshape(1, 1, 1, -1) % a.shape[3]
    return a[sl0, sl1, sl2, sl3]


def get_bin_middle(bins):
    """Return the middle of the input bins."""
    return (bins[1:] + bins[:-1]) / 2


def add_velocity(df, groupby="Pid", xpos="xf", ypos="yf"):
    """Add velocity to dataframe."""
    unit_conversion = 10
    pos_to_vel = {"xf": "uf", "yf": "vf"}
    for direction in [xpos, ypos]:
        df.loc[:, pos_to_vel[direction]] = df.groupby([groupby])[direction].transform(
            lambda x: savgol_filter(x, 49, polyorder=1, deriv=1, mode="interp") * unit_conversion
        )
    return df


# def weighted_matrix_mean(a, aC, b, bC):
def weighted_mean_of_two_matrices(a, aC, b, bC):
    vals_stack = np.stack([a, b], axis=0)
    counts_stack = np.stack([aC, bC], axis=0)
    counts_sum = np.nansum(counts_stack, axis=0)
    counts_sum_stack = np.stack([counts_sum] * 2, axis=0)
    weights = np.true_divide(
        counts_stack,
        counts_sum_stack,
        where=(counts_stack != 0) | (counts_sum_stack != 0),
    )
    weighted_vals = np.multiply(vals_stack, weights)
    weighted_mean = np.nansum(weighted_vals, axis=0)

    # If both values are nan, return nan
    both_nan = np.sum(np.stack([np.isnan(a), np.isnan(b)], axis=0), axis=0) == 2
    weighted_mean = np.where(both_nan, np.nan, weighted_mean)
    return weighted_mean


def test_weighted_mean_of_two_matrices(a, aC):
    """If values and counts are equal, the weighted average should be equal to the input values."""
    out = weighted_mean_of_two_matrices(a, aC, a, aC)
    assert np.array_equal(out, a, equal_nan=True)
    print("Test passed.")


def sample_nd_histogram(origin_histogram, N_samples=1):
    """Sample origin positions from initial position heatmap."""
    flat_origin_histogram = origin_histogram.flatten()
    flat_indices = range(len(flat_origin_histogram))
    indices1d = np.random.choice(
        a=flat_indices,
        size=N_samples,
        replace=True,
        p=flat_origin_histogram / np.sum(flat_origin_histogram),
    )

    return np.unravel_index(indices1d, origin_histogram.shape)


def digitize_values_to_grid(values: pd.Series, grid: np.ndarray) -> np.ndarray:
    indices = np.digitize(values, grid) - 1
    indices = np.where(indices < 0, 0, indices)
    indices = np.where(indices > len(grid) - 2, len(grid) - 2, indices)
    return indices


def weighted_mean_of_matrix(field: np.ndarray, histogram: np.ndarray, axes: Tuple = (2, 3, 4)) -> np.ndarray:
    weighted_field = np.nansum(field * histogram, axis=axes)
    position_histogram = np.nansum(histogram, axis=axes)
    weighted_field /= np.where(position_histogram != 0, position_histogram, np.inf)
    return weighted_field
