"""Infer force fields from trajectories."""

import copy
import logging

# from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import norm

from physped.core.digitizers import digitize_coordinates_to_lattice
from physped.core.distribution_approximator import GaussianApproximation
from physped.core.lattice import Lattice
from physped.core.piecewise_potential import PiecewisePotential

# from physped.io.readers import read_piecewise_potential_from_file
from physped.utils.functions import compose_functions, periodic_angular_conditions, weighted_mean_of_two_matrices

log = logging.getLogger(__name__)


def learn_potential_from_trajectories(trajectories: pd.DataFrame, config: DictConfig) -> PiecewisePotential:
    """
    Convert trajectories to a grid of histograms and parameters.

    Args:
        trajectories: A DataFrame of trajectories.
        grid_bins: A dictionary of grid values for each dimension.

    Returns:
        A dictionary of DiscreteGrid objects for storing histograms and parameters.
    """
    log.info("Start learning the piecewise potential")
    lattice = Lattice(config.params.grid.bins)
    dist_approximation = GaussianApproximation()
    piecewise_potential = PiecewisePotential(lattice, dist_approximation)

    trajectories = prepare_trajectories_for_lattice_parametrization(trajectories, lattice=lattice)

    piecewise_potential.histogram = add_trajectories_to_histogram(
        piecewise_potential.histogram, trajectories, "fast_grid_indices"
    )
    piecewise_potential.histogram_slow = add_trajectories_to_histogram(
        piecewise_potential.histogram_slow, trajectories, "slow_grid_indices"
    )

    piecewise_potential.parametrization = parameterize_trajectories(piecewise_potential.parametrization, trajectories, config)

    piecewise_potential.reparametrize_to_curvature(config)
    return piecewise_potential


def apply_periodic_angular_conditions(trajectories: pd.DataFrame, lattice: Lattice) -> pd.DataFrame:
    """Apply periodic angular conditions to the trajectories.

    This function makes sure that the angles are within the range of the angular lattice bins.

    Args:
        trajectories: the trajectory data set.
        lattice: the lattice object

    Returns:
        The trajectories with the angular conditions applied.
    """
    theta_cols = [col for col in trajectories.columns if "theta" in col]
    for col in theta_cols:
        trajectories[col] = periodic_angular_conditions(trajectories[col], lattice.bins["theta"])
    log.info("Periodic angular conditions applied to columns %s", theta_cols)
    return trajectories


def digitize_trajectories_to_grid(trajectories: pd.DataFrame, lattice: Lattice) -> pd.DataFrame:
    """Digitize trajectories to a lattice.

    Adds a column to the dataframe with the trajectories that contains the slow indices

    Args:
        grid_bins: The bins which define the lattice.
        trajectories: The trajectories to digitize.

    Returns:
        The trajectories with an extra column for the slow indices.
    """
    indices = {}
    for obs, dynamics in [(obs, dynamics) for obs in lattice.bins.keys() for dynamics in ["f", "s"]]:
        if obs == "k":
            dobs = obs
        else:
            dobs = obs + dynamics
        inds = digitize_coordinates_to_lattice(trajectories[dobs], lattice.bins[obs])
        indices[dobs] = inds

    indices["thetaf"] = np.where(indices["rf"] == 0, 0, indices["thetaf"])
    indices["thetas"] = np.where(indices["rs"] == 0, 0, indices["thetas"])

    trajectories["fast_grid_indices"] = list(zip(indices["xf"], indices["yf"], indices["rf"], indices["thetaf"], indices["k"]))
    trajectories["slow_grid_indices"] = list(zip(indices["xs"], indices["ys"], indices["rs"], indices["thetas"], indices["k"]))
    return trajectories


prepare_trajectories_for_lattice_parametrization = compose_functions(
    apply_periodic_angular_conditions, digitize_trajectories_to_grid
)


def add_trajectories_to_histogram(histogram: np.ndarray, trajectories: pd.DataFrame, groupbyindex: str) -> np.ndarray:
    """Add trajectories to a histogram.

    Args:
        histogram: The histogram to add the trajectories to.
        trajectories: The trajectories to add to the histogram.

    Returns:
        The updated histogram.
    """
    for grid_index, group in trajectories.groupby(groupbyindex):
        histogram[grid_index] += len(group)
    return histogram


def parameterize_trajectories(parametrization: np.ndarray, trajectories: pd.DataFrame, config: DictConfig):
    """Fit trajectories to the lattice.

    Fit the fast dynamics conditioned to the slow dynamics.

    Args:
        parametrization: The initialized, empty, parametrization matrix.
        trajectories: The trajectories to fit.
        config: The configuration parameters.

    Returns:
        The updated parametrization matrix.
    """
    fit_output = trajectories.groupby("slow_grid_indices").apply(fit_probability_distributions, config=config).dropna().to_dict()
    for key, value in fit_output.items():
        parametrization[key[0], key[1], key[2], key[3], key[4], :, :] = value
    log.info("Finished learning piecewise potential from trajectories.")
    return parametrization


def calculate_position_based_emperic_potential(histogram_slow, config: DictConfig):
    position_counts = np.nansum(histogram_slow, axis=(2, 3, 4))
    position_counts = np.where(position_counts < config.params.model.minimum_fitting_threshold, np.nan, position_counts)
    A = 0.02  # TODO: Move to config
    position_based_emperic_potential = A * (-np.log(position_counts) + np.log(np.nansum(histogram_slow)))
    return position_based_emperic_potential


def accumulate_grids(cummulative_grids: PiecewisePotential, grids_to_add: PiecewisePotential) -> PiecewisePotential:
    """Accumulate grids by taking a weighted mean of the fit parameters.

    The goal of this function is to sum PiecewisePotential objects.

    Args:
        cummulative_grids: The cumulative grids to add to.
        grids_to_add: The grids to add to the cumulative grids.

    Returns:
        The updated cumulative grids.
    """
    # ! WARNING: This function needs to be tested. Seems to have a bug.
    # ! Perhaps this needs to be a dunder class method i.e. __add__
    for p, _ in enumerate(cummulative_grids.fit_param_names):  # Loop over all fit parameters
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


def extract_submatrix(matrix: np.ndarray, slicing_indices: List[tuple]) -> np.ndarray:
    """Extract a submatrix from a nd-matrix using periodic boundary conditions.

    Periodicity is needed for the angular dimension.

    Args:
        matrix: The input nd-matrix to slice.
        slicing_indices: A list of slice tuples for each dimension of the nd-matrix.

    Returns:
        The submatrix.
    """
    if any(slice[0] > slice[1] for slice in slicing_indices):
        raise ValueError("Slicing indices must be ascending.")

    reshape_dimension = (-1,) + (1,) * (len(slicing_indices) - 1)
    slicing_indices = [
        np.arange(*slice).reshape(np.roll(reshape_dimension, i)) % matrix.shape[i] for i, slice in enumerate(slicing_indices)
    ]
    return matrix[tuple(slicing_indices)]


def fit_probability_distributions(group: pd.DataFrame, config: DictConfig) -> np.ndarray:
    """Fits a group of data points and returns the fit parameters.

    Args:
        group: The group of data points to fit the normal distribution to.
        config: The configuration parameters.

    Returns:
        A matrix containing the fit parameters.
    """
    if len(group) < config.params.model.minimum_fitting_threshold:
        return np.nan  # ? Can we do better if we have multiple files?
    fit_func = norm.fit  # * Other functions can be implemented here
    fit_parameters = np.zeros((4, 2)) * np.nan
    for i, variable in enumerate(["xf", "yf", "uf", "vf"]):
        mu, std = fit_func(group[variable])  # fit normal distribution to fast modes
        fit_parameters[i, :] = [mu, std**2]  # store mean and variance of normal distribution
    return fit_parameters


def get_grid_indices(piecewise_potential: PiecewisePotential, point: List[float]) -> np.ndarray:
    """Given a point (xs, ys, thetas, rs), return the associated lattice indices.

    This function is 4-dimensional.

    If the radial velocity is in the lowest bin, the angular velocity is automatically
    also added to the lowest bin. In other words, the angular velocities are not
    discretized for low radial velocity.

    Args:
        potential: The piecewise potential.
        point: A list with slow positions and velocities (xs, ys, thetas, rs).

    Returns:
        A tuple of grid indices.
    """
    # ! Write a test for this function
    indices = np.array([], dtype=int)
    for val, obs in zip(point, piecewise_potential.lattice.bins.keys()):
        grid = piecewise_potential.lattice.bins[obs]
        indices = np.append(indices, digitize_coordinates_to_lattice(val, grid))

    # For r = 0 all theta are 0
    if indices[2] == 0:
        indices[3] = 0  # merge grid cells for low r_s

    return indices


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
    selection_grid_indices = digitize_coordinates_to_lattice(selection_coordinates, bins)
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
        grid_bins = piecewise_potential.lattice.bins.get(observable)
        print(grid_bins)

        if not value:  # if None select full grid
            value = [grid_bins[0], grid_bins[-2]]
        elif isinstance(value, int):  # if int select single value on grid
            value = [float(value), float(value)]
        elif isinstance(value, float):  # if float select single value on grid
            value = [value, value]

        grid_selection[observable]["selection"] = value
        grid_ids = digitize_coordinates_to_lattice(value, grid_bins)
        grid_selection[observable]["grid_ids"] = grid_ids
        grid_boundaries = get_boundary_coordinates_of_selection(grid_bins, observable, grid_ids)
        grid_selection[observable]["bounds"] = grid_boundaries

        if observable == "theta":
            while grid_boundaries[1] < grid_boundaries[0]:
                grid_boundaries[1] += 2 * np.pi

        grid_selection[observable]["periodic_bounds"] = grid_boundaries
    return grid_selection
