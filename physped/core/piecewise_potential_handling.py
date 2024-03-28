from typing import List, Union

import numpy as np


def get_most_left_boundary(values: Union[float, List[float]], bins: np.ndarray) -> int:
    """
    Get the index of the leftmost bin boundary that contains values.

    Args:
        values : A single integer or float value, or a list of integer and/or float values.
        bins : An array of bin boundaries.

    Returns:
        The index of the leftmost bin boundary that contains the minimum value.
    """

    shifted_bins = np.copy(bins) - bins[0]

    # If values is a single number (int or float), convert it to a list
    if isinstance(values, (int, float)):
        values = [values, values]

    if len(values) != 2:
        raise ValueError("The 'values' argument must be a list of two elements or a single number")

    values = [x - bins[0] for x in values]

    binspan = bins.max() - bins.min()

    if not np.diff(values) < binspan:
        raise ValueError("The difference between the two elements in 'values' is too large.")

    values = values % binspan
    return np.digitize(values[0], shifted_bins) - 1


def get_most_right_boundary(values: Union[int, float, List[Union[int, float]]], bins: np.ndarray) -> int:
    """
    Get the index of the rightmost bin boundary that contains values.

    Args:
        values (Union[int, float, List[Union[int, float]]]): A single integer or float value,
        or a list of integer and/or float values.
        bins (np.ndarray): An array of bin boundaries.

    Returns:
        int: The index of the rightmost bin boundary that contains the maximum value.
    """

    shifted_bins = np.copy(bins) - bins[0]

    # If values is a single number (int or float), convert it to a list
    if isinstance(values, (int, float)):
        values = [values, values]

    if len(values) != 2:
        raise ValueError("The 'values' argument must be a list of two elements or a single number")

    values = [x - bins[0] for x in values]

    binspan = bins.max() - bins.min()

    if not np.diff(values) < binspan:
        raise ValueError("The difference between the two elements in 'values' is too large.")

    values = values % binspan
    return np.digitize(values[1], shifted_bins)


def get_the_boundaries_that_enclose_the_selected_values(
    values: Union[int, float, List[Union[int, float]]], bins: np.ndarray
) -> List[int]:
    """
    Get the boundaries that enclose the selected values.

    Args:
        values (Union[int, float, List[Union[int, float]]]): The selected values.
        bins (np.ndarray): The array of bin boundaries.

    Returns:
        List[int]: A list containing the left and right boundaries that enclose the selected values.
    """
    left_boundary_index = get_most_left_boundary(values, bins)
    right_boundary_index = get_most_right_boundary(values, bins)
    left_boundary = bins[left_boundary_index]
    right_boundary = bins[right_boundary_index]
    if right_boundary < left_boundary:
        right_boundary += bins.max() - bins.min()
    return [left_boundary, right_boundary]


def get_the_indices_of_the_gird_that_enclose_the_selected_values(
    values: Union[int, float, List[Union[int, float]]], bins: np.ndarray
) -> List[int]:
    """
    Get the boundaries that enclose the selected values.

    Args:
        values (Union[int, float, List[Union[int, float]]]): The selected values.
        bins (np.ndarray): The array of bin boundaries.

    Returns:
        List[int]: A list containing the left and right boundaries that enclose the selected values.
    """
    left_boundary_index = get_most_left_boundary(values, bins)
    right_boundary_index = get_most_right_boundary(values, bins)
    if right_boundary_index < left_boundary_index:
        right_boundary_index += len(bins)
    return [left_boundary_index, right_boundary_index]
