import logging
from typing import List

import numpy as np
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def apply_periodic_conditions_to_the_angle_theta(theta: float):
    """
    Apply periodic conditions to the angle theta.

    Args:
        theta (float): The angle theta.

    Returns:
        float: The angle theta after applying the periodic conditions.
    """
    theta += np.pi
    return theta % (2 * np.pi) - np.pi


def is_selected_value_outside_bins(selected_value: float, bins: dict, grid_dimension: str) -> bool:
    outside_bins = (selected_value < bins[grid_dimension][0]) or (selected_value > bins[grid_dimension][-1])
    if outside_bins:
        log.error(
            "The selected %s value (%s) must be within the range of the bins [%s,%s].",
            grid_dimension,
            selected_value,
            bins[grid_dimension][0],
            bins[grid_dimension][-1],
        )
    return outside_bins


def is_selected_point_within_grid(selected_point: OmegaConf, grid_bins: dict) -> None:
    """
    Validate the selection.

    Args:
        selection (OmegaConf): The selection.

    Returns:
        None
    """
    x_value_outside_bins = is_selected_value_outside_bins(selected_point.x, grid_bins, "x")
    y_value_outside_bins = is_selected_value_outside_bins(selected_point.y, grid_bins, "y")
    r_value_outside_bins = is_selected_value_outside_bins(selected_point.r, grid_bins, "r")
    theta_value_outside_bins = is_selected_value_outside_bins(selected_point.theta_periodic, grid_bins, "theta")
    k_value_outside_bins = is_selected_value_outside_bins(selected_point.k, grid_bins, "k")
    selected_point_outside_bins = [
        x_value_outside_bins,
        y_value_outside_bins,
        r_value_outside_bins,
        theta_value_outside_bins,
        k_value_outside_bins,
    ]
    if any(selected_point_outside_bins):
        raise ValueError("The selected point is not located within the grid.")
    else:
        log.info("The selectied point is located within the grid.")


def get_boundaries_that_enclose_the_selected_bin(bin_index: OmegaConf, bins: dict) -> dict:
    left_bound = bins[bin_index]
    right_bound = bins[bin_index + 1]
    return [left_bound, right_bound]


def get_index_of_the_enclosing_bin(selected_value: float, bins: np.ndarray) -> int:
    """
    Get the index of the bin that encloses the value.

    Args:
        selected_value (float): A single integer or float value.
        bins (np.ndarray): An array of bin boundaries.

    Returns:
        int: The index of the bin that encloses the value.
    """
    # ! Note that the value can be outside the range of the bins.
    # ! In this case it returns the extrema i.e. 0 or len(bins).
    if selected_value < bins[0]:
        return np.nan
    if selected_value > bins[-1]:
        return np.nan
    shifted_bins = np.copy(bins) - bins[0]
    # print(shifted_bins)
    selected_value = selected_value - bins[0]
    # print(selected_value)
    return int(np.digitize(selected_value, shifted_bins, right=False) - 1)


def evaluate_selection_point(config):
    selection = config.params.selection
    selected_point = selection.point
    grid_bins = config.params.grid.bins

    selected_point.theta_periodic = apply_periodic_conditions_to_the_angle_theta(selected_point.theta)
    is_selected_point_within_grid(selected_point, grid_bins)
    selected_point.x_index = get_index_of_the_enclosing_bin(selected_point.x, grid_bins["x"])
    selected_point.y_index = get_index_of_the_enclosing_bin(selected_point.y, grid_bins["y"])
    selected_point.r_index = get_index_of_the_enclosing_bin(selected_point.r, grid_bins["r"])
    selected_point.theta_index = get_index_of_the_enclosing_bin(selected_point.theta_periodic, grid_bins["theta"])
    selected_point.k_index = get_index_of_the_enclosing_bin(selected_point.k, grid_bins["k"])
    return config
    # log.info(f"Selection : {OmegaConf.to_yaml(selection)}")


def is_selected_range_within_grid(selected_range: OmegaConf, grid_bins: dict) -> None:
    x_values_outside_bins = [is_selected_value_outside_bins(x, grid_bins, "x") for x in selected_range.x]
    y_values_outside_bins = [is_selected_value_outside_bins(y, grid_bins, "y") for y in selected_range.y]
    r_values_outside_bins = [is_selected_value_outside_bins(r, grid_bins, "r") for r in selected_range.r]
    theta_values_outside_bins = [
        is_selected_value_outside_bins(theta, grid_bins, "theta") for theta in selected_range.theta_periodic
    ]
    k_values_outside_bins = [is_selected_value_outside_bins(k, grid_bins, "k") for k in selected_range.k]
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
    log.info("x bins: %s", [np.round(x, 2) for x in grid_bins["x"]])
    log.info(
        "range: %s ---> %s ---> %s",
        selected_range.x,
        selected_range.x_indices,
        [np.round(x, 2) for x in selected_range.x_bounds],
    )
    log.info("y bins: %s", [np.round(x, 2) for x in grid_bins["y"]])
    log.info(
        "range: %s ---> %s ---> %s",
        selected_range.y,
        selected_range.y_indices,
        [np.round(x, 2) for x in selected_range.y_bounds],
    )
    log.info("r bins: %s", [np.round(x, 2) for x in grid_bins["r"]])
    log.info(
        "range: %s ---> %s ---> %s",
        selected_range.r,
        selected_range.r_indices,
        [np.round(x, 2) for x in selected_range.r_bounds],
    )
    log.info("theta bins: %s", [np.round(x, 2) for x in grid_bins["theta"]])
    log.info(
        "range: %s ---> %s ---> %s",
        selected_range.theta,
        selected_range.theta_indices,
        [np.round(x, 2) for x in selected_range.theta_bounds],
    )
    log.info("k bins: %s", [np.round(x, 2) for x in grid_bins["k"]])
    log.info(
        "range: %s ---> %s ---> %s",
        selected_range.k,
        selected_range.k_indices,
        [np.round(x, 2) for x in selected_range.k_bounds],
    )
    return config
