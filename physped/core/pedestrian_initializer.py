import logging
from typing import List

import numpy as np
import pandas as pd

from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.functions import polar_to_cartesian_coordinates

log = logging.getLogger(__name__)


def sample_trajectory_origins_from_trajectory_state_n(
    parameters: dict, measured_trajectories: pd.DataFrame, state_n, piecewise_potential: PiecewisePotential
) -> np.ndarray:
    ntrajs = parameters.simulation.ntrajs

    if state_n == -1:
        # Sample from random point along each path
        states_to_sample_from = measured_trajectories.groupby("Pid").apply(lambda x: x.sample(1)).reset_index(drop=True)
    elif state_n >= 0:
        # Sample from state n along each path
        states_to_sample_from = measured_trajectories[measured_trajectories["k"] == state_n].copy()

    # Make sure that the potential is defined for the states we sample
    states_to_sample_from = potential_defined_at_slow_state(states_to_sample_from, piecewise_potential)
    states_to_sample_from = states_to_sample_from[states_to_sample_from["potential_defined"]].copy()
    sampled_states = states_to_sample_from[["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"]].sample(n=ntrajs, replace=True)
    log.info("Sampled %d origins from the input trajectories.", ntrajs)
    return sampled_states.to_numpy()


def potential_defined_at_slow_state(paths: pd.DataFrame, piecewise_potential: PiecewisePotential) -> pd.DataFrame:
    # REQUIRED: Dataframe must have a column with the slow grid indices
    # TODO : Move this function to another module. Change the slow_grid_indices to lists rather than tuples
    indices = np.array(list(paths["slow_grid_indices"]))
    potential_defined = np.where(np.isnan(piecewise_potential.parametrization), False, True)
    # All free parameters must be defined
    potential_defined = np.all(potential_defined, axis=(-2, -1))
    paths["potential_defined"] = potential_defined[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 4]]
    return paths


def sample_trajectory_origins_from_heatmap(piecewise_potential: PiecewisePotential, parameters: dict) -> np.ndarray:
    sampled_origins = sample_from_ndarray(piecewise_potential.histogram[..., 0], parameters.simulation.ntrajs)
    sampled_origins = np.hstack((sampled_origins, np.zeros((sampled_origins.shape[0], 1), dtype=int)))
    sampled_origins = convert_grid_indices_to_coordinates(piecewise_potential, sampled_origins)
    sampled_origins = np.hstack((sampled_origins, sampled_origins))
    sampled_origins = np.delete(sampled_origins, 4, axis=1)
    return sampled_origins


def sample_from_ndarray(origin_histogram: np.ndarray, N_samples: int = 1) -> np.ndarray:
    """Sample origin positions from a heatmap with initial positions.

    ! Used for simulation purposes

    Parameters:
    - origin_histogram: The initial position heatmap.
    - N_samples: The number of samples to generate. Default is 1.

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


def random_uniform_value_in_bin(lattice_indices: List[int], bins: np.ndarray) -> np.ndarray:
    """Generate random uniform values within each bin.

    ! Used for simulation purposes

    This function is used to sample origins from a histogram. Given the index of the
    sampled lattice cell the function will return a real-world variable.

    For example:
    with bins v_r = [0,0.5,1] and lattice_indices [1] the function returns a random uniformly
    sampled value between 0.5 and 1.

    Args:
        lattice_indices: A list with lattice indices.
        bins: Array of bin edges.

    Returns:
        Array of random uniform values within each bin.
    """
    left = bins[lattice_indices]
    right = bins[lattice_indices + 1]
    return np.random.uniform(left, right)


def convert_grid_indices_to_coordinates(potential_grid: PiecewisePotential, X_0: np.ndarray) -> np.ndarray:
    """Convert grid indices to Cartesian coordinates.

    ! Used for simulation purposes

    Parameters:
    - grids (Grids): The Grids object containing the grid definitions.
    - X_0 (List[int]): A list of grid indices.

    Returns:
    - A list of Cartesian coordinates.
    """
    # TODO: Add some noise within the bin.
    # xf, yf, rf, thetaf, k = (grids.bins[dim][X_0[:,i]] for i, dim in enumerate(grids.dimensions))
    xf = random_uniform_value_in_bin(X_0[:, 0], potential_grid.bins["x"])
    yf = random_uniform_value_in_bin(X_0[:, 1], potential_grid.bins["y"])
    rf = random_uniform_value_in_bin(X_0[:, 2], potential_grid.bins["r"])
    thetaf = random_uniform_value_in_bin(X_0[:, 3], potential_grid.bins["theta"])
    k = potential_grid.bins["k"][X_0[:, 4]]

    uf, vf = polar_to_cartesian_coordinates(rf, thetaf)
    return np.array([xf, yf, uf, vf, k]).T
