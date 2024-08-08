"""This module contains functions to select a single point or a range of values in the configuration file.

The only purpose is to let the user make a selection along the discretization lattice.
"""

import logging
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def apply_periodic_conditions_to_the_angle_theta(theta: float) -> float:
    """Apply periodic conditions to the angle theta.

    The output will be in the range [-pi, pi].

    Args:
        theta: The angle theta.

    Returns:
        The angle theta in the range [-pi, pi].
    """
    theta += np.pi
    return theta % (2 * np.pi) - np.pi


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

    # validate_value_within_lattice(selected_point.x, grid_bins, "x")
    # validate_value_within_lattice(selected_point.y, grid_bins, "y")
    # validate_value_within_lattice(selected_point.r, grid_bins, "r")
    # validate_value_within_lattice(selected_point.theta_periodic, grid_bins, "theta")
    # validate_value_within_lattice(selected_point.k, grid_bins, "k")
    # selected_point_outside_bins = [
    #     x_value_outside_bins,
    #     y_value_outside_bins,
    #     r_value_outside_bins,
    #     theta_value_outside_bins,
    #     k_value_outside_bins,
    # ]
    # if any(selected_point_outside_bins):
    #     raise ValueError("The selected point is not located within the grid.")
    # else:


def get_boundaries_that_enclose_the_selected_bin(bin_index: OmegaConf, bins: dict) -> dict:
    left_bound = bins[bin_index]
    right_bound = bins[bin_index + 1]
    return [left_bound, right_bound]


def get_index_of_the_enclosing_bin(selected_value: float, bins: np.ndarray) -> int:
    """Get the index of the bin that encloses the value.

    Args:
        selected_value: A single integer or float value.
        bins: An array of bin edges.

    Returns:
        The index of the bin that encloses the value.
    """
    # ! Note that the value can be outside the range of the bins.
    # ! In this case it returns nan.
    if selected_value < bins[0]:
        return np.nan
    if selected_value > bins[-1]:
        return np.nan
    shifted_bins = np.copy(bins) - bins[0]
    selected_value = selected_value - bins[0]
    return int(np.digitize(selected_value, shifted_bins, right=False) - 1)


def evaluate_selection_point(config):
    selection = config.params.selection
    selected_point = selection.point
    grid_bins = config.params.grid.bins

    selected_point.theta_periodic = apply_periodic_conditions_to_the_angle_theta(selected_point.theta)
    validate_point_within_lattice(selected_point, grid_bins)
    selected_point.x_index = get_index_of_the_enclosing_bin(selected_point.x, grid_bins["x"])
    selected_point.y_index = get_index_of_the_enclosing_bin(selected_point.y, grid_bins["y"])
    selected_point.r_index = get_index_of_the_enclosing_bin(selected_point.r, grid_bins["r"])
    selected_point.theta_index = get_index_of_the_enclosing_bin(selected_point.theta_periodic, grid_bins["theta"])
    selected_point.k_index = get_index_of_the_enclosing_bin(selected_point.k, grid_bins["k"])
    return config
    # log.info(f"Selection : {OmegaConf.to_yaml(selection)}")


# ! The functions below are used to select a range of values in the configuration file


def is_selected_range_within_grid(selected_range: OmegaConf, grid_bins: dict) -> None:
    x_values_outside_bins = [validate_value_within_lattice(x, grid_bins, "x") for x in selected_range.x]
    y_values_outside_bins = [validate_value_within_lattice(y, grid_bins, "y") for y in selected_range.y]
    r_values_outside_bins = [validate_value_within_lattice(r, grid_bins, "r") for r in selected_range.r]
    theta_values_outside_bins = [
        validate_value_within_lattice(theta, grid_bins, "theta") for theta in selected_range.theta_periodic
    ]
    k_values_outside_bins = [validate_value_within_lattice(k, grid_bins, "k") for k in selected_range.k]
    selected_point_outside_bins = [
        x_values_outside_bins,
        y_values_outside_bins,
        r_values_outside_bins,
        theta_values_outside_bins,
        k_values_outside_bins,
    ]
    if any([any(x) for x in selected_point_outside_bins]):
        raise ValueError("The selected range is not located within the grid.")
    else:
        log.info("The selected range is located within the grid.")


def is_range_decreasing(selected_range: OmegaConf) -> bool:
    """
    Check if the range is increasing.

    Args:
        selected_range (OmegaConf): The selection.

    Returns:
        bool: True if the range is increasing, False otherwise.
    """
    range_decreasing = selected_range[0] > selected_range[1]
    if range_decreasing:
        log.error(
            "The selected range [%s] must be increasing.",
            selected_range,
        )
    return range_decreasing


def is_selected_range_valid(selected_range: OmegaConf) -> None:
    """
    Validate the selection.

    Args:
        selection (OmegaConf): The selection.

    Returns:
        None
    """
    x_decreasing = is_range_decreasing(selected_range.x)
    y_decreasing = is_range_decreasing(selected_range.y)
    r_decreasing = is_range_decreasing(selected_range.r)
    theta_decreasing = is_range_decreasing(selected_range.theta)
    k_decreasing = is_range_decreasing(selected_range.k)

    if any([x_decreasing, y_decreasing, r_decreasing, theta_decreasing, k_decreasing]):
        raise ValueError("The selected range is invalid.")
    else:
        log.info("The selected range is valid.")


def get_indices_of_the_enclosing_range(selected_range: List[float], bins: np.ndarray) -> List[int]:
    return [get_index_of_the_enclosing_bin(x, bins) for x in selected_range]


def get_boundaries_that_enclose_the_selected_range(selected_range: OmegaConf, bins: dict) -> dict:
    left_bound = bins[selected_range[0]]
    right_bound = bins[selected_range[1] + 1]
    return [float(left_bound), float(right_bound)]


def evaluate_selection_range(config):
    selected_range = config.params.selection.range
    grid_bins = config.params.grid.bins
    selected_range.theta_periodic = [apply_periodic_conditions_to_the_angle_theta(theta) for theta in selected_range.theta]
    is_selected_range_valid(selected_range)
    is_selected_range_within_grid(selected_range, grid_bins)
    selected_range.x_indices = get_indices_of_the_enclosing_range(selected_range.x, grid_bins["x"])
    selected_range.y_indices = get_indices_of_the_enclosing_range(selected_range.y, grid_bins["y"])
    selected_range.r_indices = get_indices_of_the_enclosing_range(selected_range.r, grid_bins["r"])
    selected_range.theta_indices = get_indices_of_the_enclosing_range(selected_range.theta_periodic, grid_bins["theta"])
    selected_range.k_indices = get_indices_of_the_enclosing_range(selected_range.k, grid_bins["k"])

    selected_range.x_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.x_indices, grid_bins["x"])
    selected_range.y_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.y_indices, grid_bins["y"])
    selected_range.r_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.r_indices, grid_bins["r"])
    selected_range.theta_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.theta_indices, grid_bins["theta"])
    selected_range.k_bounds = get_boundaries_that_enclose_the_selected_range(selected_range.k_indices, grid_bins["k"])

    # log.info(f"Selection : {OmegaConf.to_yaml(selected_range)}")
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
