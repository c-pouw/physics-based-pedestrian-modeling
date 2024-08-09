"""This module contains functions to select a single point or a range of values in the configuration file.

The only purpose is to let the user make a selection along the discretization lattice.
"""

import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf

from physped.core.digitizers import digitize_coordinates_to_lattice
from physped.utils.functions import periodic_angular_conditions

log = logging.getLogger(__name__)


# ! The functions below are used to select a single point in the configuration file


def validate_value_within_lattice(selected_value: float, bins: dict, dimension: str) -> None:
    """Check if the selected value is outside the lattice.

    Args:
        selected_value: The selected value in one of the dimensions.
        bins: The bins of the lattice in all the dimensions.
        dimension: The dimension of the given value.

    Raises:
        ValueError: If the selected value is outside the lattice.
    """
    left_bound = bins[dimension][0]
    right_bound = bins[dimension][-1]
    outside_lattice = (selected_value < left_bound) or (selected_value > right_bound)
    if outside_lattice:
        raise ValueError(
            f"Selected {dimension} value ({selected_value}) is outside the lattice. "
            f"Please select a value within the range [{left_bound},{right_bound}]."
        )


def validate_point_within_lattice(selected_point: DictConfig, grid_bins: dict) -> None:
    """Validate the selection.

    Args:
        selected_point: The selected point.
        grid_bins: The bins of the lattice in all the dimensions.
    """
    for dimension, value in selected_point.items():
        validate_value_within_lattice(value, grid_bins, dimension)
    log.info("The selected point is located within the grid.")


def get_boundaries_that_enclose_the_selected_bin(bin_index: int, bins: np.ndarray) -> dict:
    left_bound = bins[bin_index]
    right_bound = bins[bin_index + 1]
    return [left_bound, right_bound]


def evaluate_selection_point(config: DictConfig) -> DictConfig:
    selection = config.params.selection
    selected_point = selection.point
    grid_bins = config.params.grid.bins

    selected_point.theta_periodic = periodic_angular_conditions(selected_point.theta, grid_bins["theta"])
    validate_point_within_lattice(selected_point, grid_bins)
    selected_point.x_index = digitize_coordinates_to_lattice(selected_point.x, grid_bins["x"])
    selected_point.y_index = digitize_coordinates_to_lattice(selected_point.y, grid_bins["y"])
    selected_point.r_index = digitize_coordinates_to_lattice(selected_point.r, grid_bins["r"])
    selected_point.theta_index = digitize_coordinates_to_lattice(selected_point.theta_periodic, grid_bins["theta"])
    selected_point.k_index = digitize_coordinates_to_lattice(selected_point.k, grid_bins["k"])
    return config


# ! The functions below are used to select a range of values in the configuration file


def is_selected_range_within_grid(selected_range: OmegaConf, grid_bins: dict) -> None:
    for dimension in ["x", "y", "r", "theta", "k"]:
        for value in selected_range[dimension]:
            validate_value_within_lattice(value, grid_bins, dimension)
    log.info("The selected range is located within the grid.")


def is_range_decreasing(selected_range: OmegaConf):
    """
    Check if the range is increasing.

    Raises:
        ValueError: If the range is decreasing

    Args:
        selected_range (OmegaConf): The selection.
    """
    range_decreasing = selected_range[0] > selected_range[1]
    if range_decreasing:
        raise ValueError("The selected range is invalid. The range must be increasing.")


def is_selected_range_valid(selected_range: OmegaConf) -> None:
    """
    Validate the selection.

    Args:
        selection: The selection.
    """
    for dimension in ["x", "y", "r", "theta", "k"]:
        is_range_decreasing(selected_range[dimension])
    log.info("The selected range is valid.")


def get_boundaries_that_enclose_the_selected_range(selected_range: OmegaConf, bins: dict) -> dict:
    left_bound = bins[selected_range[0]]
    right_bound = bins[selected_range[1] + 1]
    return [float(left_bound), float(right_bound)]


def evaluate_selection_range(config: DictConfig) -> DictConfig:
    selected_range = config.params.selection.range
    grid_bins = config.params.grid.bins
    selected_range.theta_periodic = periodic_angular_conditions(selected_range.theta, grid_bins["theta"]).tolist()

    is_selected_range_valid(selected_range)
    is_selected_range_within_grid(selected_range, grid_bins)
    selected_range.x_indices = digitize_coordinates_to_lattice(selected_range.x, grid_bins["x"]).tolist()
    selected_range.y_indices = digitize_coordinates_to_lattice(selected_range.y, grid_bins["y"]).tolist()
    selected_range.r_indices = digitize_coordinates_to_lattice(selected_range.r, grid_bins["r"]).tolist()
    selected_range.theta_indices = digitize_coordinates_to_lattice(selected_range.theta_periodic, grid_bins["theta"]).tolist()
    selected_range.k_indices = digitize_coordinates_to_lattice(selected_range.k, grid_bins["k"]).tolist()

    selected_range.x_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.x_indices, grid_bins["x"])
    selected_range.y_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.y_indices, grid_bins["y"])
    selected_range.r_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.r_indices, grid_bins["r"])
    selected_range.theta_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.theta_indices, grid_bins["theta"])
    selected_range.k_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.k_indices, grid_bins["k"])

    log.debug("x bins: %s", [np.round(x, 2) for x in grid_bins["x"]])
    log.debug(
        "range: %s ---> %s ---> %s",
        selected_range.x,
        selected_range.x_indices,
        [np.round(x, 2) for x in selected_range.x_bounds],
    )
    log.debug("y bins: %s", [np.round(x, 2) for x in grid_bins["y"]])
    log.debug(
        "range: %s ---> %s ---> %s",
        selected_range.y,
        selected_range.y_indices,
        [np.round(x, 2) for x in selected_range.y_bounds],
    )
    log.debug("r bins: %s", [np.round(x, 2) for x in grid_bins["r"]])
    log.debug(
        "range: %s ---> %s ---> %s",
        selected_range.r,
        selected_range.r_indices,
        [np.round(x, 2) for x in selected_range.r_bounds],
    )
    log.debug("theta bins: %s", [np.round(x, 2) for x in grid_bins["theta"]])
    log.debug(
        "range: %s ---> %s ---> %s",
        selected_range.theta,
        selected_range.theta_indices,
        [np.round(x, 2) for x in selected_range.theta_bounds],
    )
    log.debug("k bins: %s", [np.round(x, 2) for x in grid_bins["k"]])
    log.debug(
        "range: %s ---> %s ---> %s",
        selected_range.k,
        selected_range.k_indices,
        [np.round(x, 2) for x in selected_range.k_bounds],
    )
    return config
