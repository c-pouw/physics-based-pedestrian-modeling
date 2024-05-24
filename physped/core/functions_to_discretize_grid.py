"""Infer force fields from trajectories."""

import copy
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

from physped.core.piecewise_potential import PiecewisePotential
from physped.io.readers import read_piecewise_potential_from_file
from physped.utils.functions import digitize_values_to_grid, pol2cart, weighted_mean_of_two_matrices

log = logging.getLogger(__name__)


# def create_grid_bins_from_config(config: dict) -> dict:
#     grid_conf = config.params.grid
#     xbins = np.arange(grid_conf.x.min, grid_conf.x.max + 0.01, grid_conf.x.step)
#     ybins = np.arange(grid_conf.y.min, grid_conf.y.max + 0.01, grid_conf.y.step)
#     rbins = np.arange(grid_conf.r.min, grid_conf.r.max + 0.01, grid_conf.r.step)
#     thetabins = np.linspace(
#         grid_conf.theta.min, grid_conf.theta.min + 2 * np.pi + 0.0001, grid_conf.theta.segments + 1
#     )
#     kbins = np.array([0, 1, 10**10])
#     gridbins = {"x": xbins, "y": ybins, "r": rbins, "theta": thetabins, "k": kbins}
#     log.info("Bins succesfully created with limits: %s", grid_conf)
#     log.debug("Grid bins: %s", gridbins)
#     return gridbins


def learn_potential_from_trajectories(trajectories: pd.DataFrame, config: dict) -> PiecewisePotential:
    """
    Convert trajectories to a grid of histograms and parameters.

    Parameters:
    - trajectories (pd.DataFrame): A DataFrame of trajectories.
    - grid_bins (dict): A dictionary of grid values for each dimension.

    Returns:
    - A dictionary of DiscreteGrid objects for storing histograms and parameters.
    """
    grid_bins = dict(config.params.grid.bins)
    filepath = Path.cwd().parent / config.filename.piecewise_potential
    if config.read.simulated_trajectories:
        log.debug("Configuration 'read.simulated_trajectories' is set to True.")
        try:
            piecewise_potential = read_piecewise_potential_from_file(filepath)
            log.warning("Piecewise potential read from file")
            log.debug("Filepath %s", filepath.relative_to(config.root_dir))
            return piecewise_potential
        except FileNotFoundError as e:
            log.error("Piecewise potential not found: %s", e)

    log.info("Start learning the piecewise potential")
    piecewise_potential = PiecewisePotential(grid_bins)
    trajectories = digitize_trajectories_to_grid(piecewise_potential.bins, trajectories)
    piecewise_potential.histogram = add_trajectories_to_histogram(
        piecewise_potential.histogram, trajectories, "fast_grid_indices"
    )
    piecewise_potential.histogram_slow = add_trajectories_to_histogram(
        piecewise_potential.histogram_slow, trajectories, "slow_grid_indices"
    )
    if config.params.simulation.sample_origins_from == "trajectories":
        piecewise_potential.trajectory_origins = trajectories.groupby("Pid").first()[["xf", "yf", "uf", "vf"]]
    piecewise_potential.fit_params = fit_trajectories_on_grid(piecewise_potential.fit_params, trajectories)
    log.info("Finished learning piecewise potential from trajectories.")
    piecewise_potential = calculate_curvature_of_the_potential(piecewise_potential, config)
    piecewise_potential = derive_potential_center(piecewise_potential, config)
    piecewise_potential.position_based_offset = calculate_position_based_emperic_potential(piecewise_potential.histogram_slow)
    # piecewise_potential = calculate_position_based_offset(piecewise_potential, config)
    return piecewise_potential


def calculate_curvature_of_the_potential(piecewise_potential: PiecewisePotential, config: dict) -> PiecewisePotential:
    var = config.params.sigma**2
    var_indices = [1, 3, 5, 7]
    variances = [piecewise_potential.fit_params[..., i] for i in var_indices]

    # Replace 0 with np.nan
    xvar, yvar, uvar, vvar = [np.where(v == 0, np.nan, v) for v in variances]

    piecewise_potential.curvature_x = uvar / xvar
    piecewise_potential.curvature_y = vvar / yvar
    piecewise_potential.curvature_u = var / (2 * uvar)
    piecewise_potential.curvature_v = var / (2 * vvar)
    return piecewise_potential


def derive_potential_center(piecewise_potential: PiecewisePotential, config: dict) -> PiecewisePotential:
    piecewise_potential.center_x = piecewise_potential.fit_params[..., 0]
    piecewise_potential.center_y = piecewise_potential.fit_params[..., 2]
    piecewise_potential.center_u = piecewise_potential.fit_params[..., 4]
    piecewise_potential.center_v = piecewise_potential.fit_params[..., 6]
    return piecewise_potential


# def calculate_position_based_offset(piecewise_potential: PiecewisePotential, config: dict) -> PiecewisePotential:
#     position_counts = np.nansum(piecewise_potential.histogram_slow, axis=(2, 3, 4))
#     position_counts = np.where(position_counts == 0, np.nan, position_counts)
#     A = 0.01  # TODO: Move to config
#     piecewise_potential.position_based_offset = A * (
#         -np.log(position_counts) + np.log(np.nansum(piecewise_potential.histogram_slow))
#     )
#     return piecewise_potential


def calculate_position_based_emperic_potential(histogram_slow):
    position_counts = np.nansum(histogram_slow, axis=(2, 3, 4))
    position_counts = np.where(position_counts == 0, np.nan, position_counts)
    A = 0.02  # TODO: Move to config
    position_based_emperic_potential = A * (-np.log(position_counts) + np.log(np.nansum(histogram_slow)))
    return position_based_emperic_potential


def accumulate_grids(cummulative_grids: PiecewisePotential, grids_to_add: PiecewisePotential) -> PiecewisePotential:
    """
    Accumulate grids by taking a weighted mean of the fit parameters.

    Parameters:
    - cummulative_grids (DiscreteGrid): The cumulative grids to add to.
    - grids_to_add (DiscreteGrid): The grids to add to the cumulative grids.

    Returns:
    - The updated cumulative grids.
    """

    for p in range(cummulative_grids.no_fit_params):  # Loop over all fit parameters
        # accumulate fit parameters
        cummulative_grids.fit_params[:, :, :, :, :, p] = weighted_mean_of_two_matrices(
            first_matrix=copy.deepcopy(cummulative_grids.fit_params[:, :, :, :, :, p]),
            counts_first_matrix=copy.deepcopy(cummulative_grids.histogram),
            second_matrix=copy.deepcopy(grids_to_add.fit_params[:, :, :, :, :, p]),
            counts_second_matrix=copy.deepcopy(grids_to_add.histogram),
        )
    # accumlate histogram
    cummulative_grids.histogram += grids_to_add.histogram
    cummulative_grids.histogram_slow += grids_to_add.histogram_slow
    return cummulative_grids


def get_slice_of_multidimensional_matrix(a: np.ndarray, slices: List[tuple]) -> np.ndarray:
    """
    Get a slice of a multidimensional NumPy array with periodic boundary conditions.

    Parameters:
    - a (np.ndarray): The input array to slice.
    - slices (List[Tuple[int, int]]): A list of slice ranges for each dimension of the array.

    Returns:
    - The sliced array.
    """
    if any(slice[0] > slice[1] for slice in slices):
        print("Slice values must be ascending.")
    reshape_dimension = (-1,) + (1,) * (len(slices) - 1)
    slices = [np.arange(*slice).reshape(np.roll(reshape_dimension, i)) % a.shape[i] for i, slice in enumerate(slices)]
    return a[tuple(slices)]


def digitize_trajectories_to_grid(grid_bins: dict, trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Digitize trajectories to a grid.

    Parameters:
    - grid_bins (dict): The grid bins to digitize the trajectories to.
    - trajectories (pd.DataFrame): The trajectories to digitize.

    Returns:
    - pd.DataFrame: The trajectories with the digitized grid indices.
    """
    indices = {}
    for obs, dynamics in [(obs, dynamics) for obs in grid_bins.keys() for dynamics in ["f", "s"]]:
        if obs == "k":
            dobs = obs
        else:
            dobs = obs + dynamics
        inds = digitize_values_to_grid(trajectories[dobs], grid_bins[obs])
        indices[dobs] = inds

    indices["thetaf"] = np.where(indices["rf"] == 0, 0, indices["thetaf"])
    indices["thetas"] = np.where(indices["rs"] == 0, 0, indices["thetas"])

    trajectories["fast_grid_indices"] = list(zip(indices["xf"], indices["yf"], indices["rf"], indices["thetaf"], indices["k"]))
    trajectories["slow_grid_indices"] = list(zip(indices["xs"], indices["ys"], indices["rs"], indices["thetas"], indices["k"]))
    return trajectories


def fit_probability_distributions(group: pd.DataFrame) -> list:
    """
    Fits normal distribution to a group of data points and returns fitting parameters.

    Parameters:
    - group (pd.DataFrame): The group of data points to fit the normal distribution to.

    Returns:
    - A list of fitting parameters.
    """
    if len(group) < 10:
        return None  # ? Can we do better if we have multiple files?
    fit_func = norm.fit  # * Other functions can be implemented here
    params = []
    for i in ["xf", "yf", "uf", "vf"]:
        mu, std = fit_func(group[i])  # fit normal distribution to fast modes
        params += [mu, std**2]  # store mean and variance of normal distribution
    return params


def fit_trajectories_on_grid(param_grid, trajectories: pd.DataFrame):
    """
    Fit trajectories to the parameter grid.

    Parameters:
    - param_grid (): The parameter grid to fit the trajectories on.
    - trajectories (pd.DataFrame): The trajectories to fit.

    Returns:
    - The grid with fit parameters.
    """
    fit_params = trajectories.groupby("slow_grid_indices").apply(fit_probability_distributions).dropna().to_dict()
    for key, value in fit_params.items():
        param_grid[key] = value
    return param_grid


def add_trajectories_to_histogram(histogram: np.ndarray, trajectories: pd.DataFrame, groupbyindex: str) -> np.ndarray:
    """
    Add trajectories to a histogram.

    Parameters:
    - histogram (): The histogram to add the trajectories to.
    - trajectories (pd.DataFrame): The trajectories to add to the histogram.

    Returns:
    - The updated histogram.
    """
    for grid_index, group in trajectories.groupby(groupbyindex):
        histogram[grid_index] += len(group)
    return histogram


# def create_grid_bins(grid_vals: dict) -> dict:
#     """
#     Create bins for a grid.

#     Parameters:
#     - grid_vals (dict): The values of the grid.

#     Returns:
#     - A dictionary of bins for each dimension of the grid.
#     """
#     grid_bins = {}
#     if "x" in grid_vals and "y" in grid_vals:
#         grid_bins = {key: np.arange(*grid_vals[key]) for key in ["x", "y"]}
#     if "r" in grid_vals:
#         grid_bins["r"] = np.array(grid_vals["r"])
#     if "theta" in grid_vals:
#         grid_bins["theta"] = np.arange(-np.pi, np.pi + 0.01, grid_vals["theta"])
#     if "k" in grid_vals:
#         grid_bins["k"] = np.array(grid_vals["k"])
#     return grid_bins


def get_grid_indices(potential_grid: PiecewisePotential, X: List[float]) -> np.ndarray:
    """
    Given a point (xs, ys, thetas, rs), returns the grid index of the point.

    Parameters:
    - grid (Grids): The Grids object containing the grid definitions.
    - X (List[float]): A list of Cartesian and polar coordinates.

    Returns:
    - A tuple of grid indices.
    """
    indices = np.array([], dtype=int)
    for val, obs in zip(X, potential_grid.dimensions):
        grid = potential_grid.bins[obs]
        indices = np.append(indices, digitize_values_to_grid(val, grid))

    if indices[2] == 0:
        indices[3] = 0  # merge grid cells for low r_s

    return indices


def get_grid_index_single_value(value: float, bins: np.array) -> int:
    """
    Returns the index of the grid cell that corresponds to the given value.
    If a value is outside the grid, it is wrapped around to the other side.
    Following a periodic boundary condition.

    Parameters:
    value (float): The value to be discretized.
    bins (np.array): An array of bin edges defining the grid cells.

    Returns:
    int: The index of the grid cell that corresponds to the given value.
    """

    if value > np.max(bins):
        value -= np.max(bins) - np.min(bins)
        grid_idx = np.digitize(value, bins)
        grid_idx += len(bins) - 1
    elif value < np.min(bins):
        value += np.max(bins) - np.min(bins)
        grid_idx = np.digitize(value, bins)
        grid_idx -= len(bins) - 1
    else:
        grid_idx = np.digitize(value, bins)
    return grid_idx


def return_grid_ids(grid, value):
    """Return the grid indices of a given observable and value."""
    # grid = grids.bins[observable]  # TODO Change attribute to 1d Grid
    # grid = self.grid_observable[observable]

    if isinstance(value, int):
        value = float(value)
    if isinstance(value, float):
        grid_id = get_grid_index_single_value(value, grid)
        grid_idx = [grid_id - 1, grid_id]

    elif (isinstance(value, list)) and (len(value) == 2):
        grid_idx = [
            get_grid_index_single_value(value[0], grid) - 1,
            get_grid_index_single_value(value[1], grid),
        ]
    else:
        grid_idx = [0, len(grid) - 1]
    return {
        "grid_idx": grid_idx,
    }


def get_boundary_coordinates_of_selection(bins, observable, values):
    """Return the grid bounds of a given observable and value."""
    if observable == "theta":
        grid_sides = [
            bins[(values[0] - 1) % (len(bins) - 1)],
            bins[(values[1]) % (len(bins) - 1)],
        ]
    else:
        grid_sides = [bins[values[0] - 1], bins[values[1]]]
    return grid_sides


def selection_to_bounds(bins, selection_coordinates, dimension):
    selection_grid_indices = [get_grid_index_single_value(x, bins) for x in selection_coordinates]
    selection_boundary_coordinates = get_boundary_coordinates_of_selection(bins, dimension, selection_grid_indices)
    return selection_boundary_coordinates


def make_grid_selection(piecewise_potential, selection):
    """Make selection."""
    # TODO: Remove dependency on PiecewisePotential object
    grid_selection = {}
    for observable, value in selection.items():
        print(observable, value)
        # for observable, value in zip(["x", "y", "r", "theta"], [x, y, r, theta, k]):
        grid_selection[observable] = {}
        # grid = grid.grid_observable[observable]
        grid_bins = piecewise_potential.bins.get(observable)
        print(grid_bins)

        if not value:  # if None select full grid
            value = [grid_bins[0], grid_bins[-2]]
        elif isinstance(value, int):  # if int select single value on grid
            value = [float(value), float(value)]
        elif isinstance(value, float):  # if float select single value on grid
            value = [value, value]

        grid_selection[observable]["selection"] = value
        grid_ids = return_grid_ids(grid_bins, value)["grid_idx"]
        grid_selection[observable]["grid_ids"] = grid_ids
        grid_boundaries = get_boundary_coordinates_of_selection(grid_bins, observable, grid_ids)
        grid_selection[observable]["bounds"] = grid_boundaries

        if observable == "theta":
            while grid_boundaries[1] < grid_boundaries[0]:
                grid_boundaries[1] += 2 * np.pi

        grid_selection[observable]["periodic_bounds"] = grid_boundaries
    return grid_selection


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


def sample_from_ndarray(origin_histogram: np.ndarray, N_samples: int = 1) -> np.ndarray:
    """
    Sample origin positions from a heatmap with initial positions.

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


def convert_grid_indices_to_coordinates(potential_grid: PiecewisePotential, X_0: np.ndarray) -> np.ndarray:
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
    xf = random_uniform_value_in_bin(X_0[:, 0], potential_grid.bins["x"])
    yf = random_uniform_value_in_bin(X_0[:, 1], potential_grid.bins["y"])
    rf = random_uniform_value_in_bin(X_0[:, 2], potential_grid.bins["r"])
    thetaf = random_uniform_value_in_bin(X_0[:, 3], potential_grid.bins["theta"])
    k = potential_grid.bins["k"][X_0[:, 4]]

    uf, vf = pol2cart(rf, thetaf)
    return np.array([xf, yf, uf, vf, k]).T
