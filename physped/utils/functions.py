import logging
from functools import reduce
from typing import Any, Callable, Tuple

import numpy as np

log = logging.getLogger(__name__)

Composable = Callable[[Any], Any]


def compose_functions(*functions: Composable) -> Composable:
    return lambda x, **kwargs: reduce(lambda df, fn: fn(df, **kwargs), functions, x)


def cartesian_to_polar_coordinates(x: float, y: float) -> tuple:
    """Convert cartesian coordinates to polar coordinates.

    Parameters:
    - x: The x-coordinate.
    - y: The y-coordinate.

    Returns:
    - A tuple with the polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def polar_to_cartesian_coordinates(rho: float, phi: float) -> tuple:
    """Convert polar coordinates to cartesian coordinates.

    Parameters:
    rho: The radial distance from the origin to the point.
    phi: The angle in radians.

    Returns:
    - A tuple containing the cartesian coordinate.s
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def get_slice_of_multidimensional_matrix(
    a: np.ndarray, slice_x: Tuple, slice_y: Tuple, slice_theta: Tuple, slice_r: Tuple
) -> np.ndarray:
    """
    Get a slice of a multidimensional matrix.

    Parameters:
    a (ndarray): The input multidimensional matrix.
    slice_x (tuple): The range of indices to slice along the x-axis.
    slice_y (tuple): The range of indices to slice along the y-axis.
    slice_theta (tuple): The range of indices to slice along the theta-axis.
    slice_r (tuple): The range of indices to slice along the r-axis.

    Returns:
    ndarray: The sliced multidimensional matrix.

    Raises:
    ValueError: If the slice values are not in ascending order.

    """
    for slice_range in [slice_x, slice_y, slice_theta, slice_r]:
        if slice_range[0] > slice_range[1]:
            raise ValueError("Slice values must be in ascending order.")
    sl0 = np.array(range(slice_x[0], slice_x[1])).reshape(-1, 1, 1, 1) % a.shape[0]
    sl1 = np.array(range(slice_y[0], slice_y[1])).reshape(1, -1, 1, 1) % a.shape[1]
    sl2 = np.array(range(slice_theta[0], slice_theta[1])).reshape(1, 1, -1, 1) % a.shape[2]
    sl3 = np.array(range(slice_r[0], slice_r[1])).reshape(1, 1, 1, -1) % a.shape[3]
    return a[sl0, sl1, sl2, sl3]


def get_bin_middle(bins: np.ndarray) -> np.ndarray:
    """Return the middle of the input bins.

    Args:
        bins: The input bins.

    Returns:
        The middle of the input bins.

    """
    return (bins[1:] + bins[:-1]) / 2


def weighted_mean_of_two_matrices(
    first_matrix: np.ndarray,
    counts_first_matrix: np.ndarray,
    second_matrix: np.ndarray,
    counts_second_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculates the weighted mean of two matrices.

    Args:
        first_matrix (numpy.ndarray): First input matrix.
        counts_first_matrix (numpy.ndarray): Counts for the first input matrix.
        second_matrix (numpy.ndarray): Second input matrix.
        counts_second_matrix (numpy.ndarray): Counts for the second input matrix.

    Returns:
        numpy.ndarray: Weighted mean of the two input matrices.
    """
    vals_stack = np.stack([first_matrix, second_matrix], axis=0)
    counts_stack = np.stack([counts_first_matrix, counts_second_matrix], axis=0)
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
    both_nan = np.sum(np.stack([np.isnan(first_matrix), np.isnan(second_matrix)], axis=0), axis=0) == 2
    weighted_mean = np.where(both_nan, np.nan, weighted_mean)
    return weighted_mean


def test_weighted_mean_of_two_matrices(first_matrix: np.ndarray, counts_first_matrix: np.ndarray) -> None:
    """
    Test function for weighted_mean_of_two_matrices.

    This function tests whether the weighted average of two matrices is equal to the input values.

    Parameters:
    first_matrix (numpy.ndarray): The first input matrix.
    counts_first_matrix (numpy.ndarray): The counts matrix corresponding to the first input matrix.
    """
    # ! Move this ot tests
    out = weighted_mean_of_two_matrices(first_matrix, counts_first_matrix, first_matrix, counts_first_matrix)
    assert np.array_equal(out, first_matrix, equal_nan=True)
    print("Test passed.")


# def sample_nd_histogram(origin_histogram: np.ndarray, sample_count: int = 1) -> np.ndarray:
#     """
#     Sample origin positions from heatmap with initial position.

#     Parameters:
#         origin_histogram (ndarray): The heatmap with initial position.
#         sample_count (int): The number of samples to generate. Default is 1.

#     Returns:
#         ndarray: The sampled origin positions.

#     """
#     flat_origin_histogram = origin_histogram.flatten()
#     flat_indices = range(len(flat_origin_histogram))
#     indices1d = np.random.choice(
#         a=flat_indices,
#         size=sample_count,
#         replace=True,
#         p=flat_origin_histogram / np.sum(flat_origin_histogram),
#     )

#     return np.unravel_index(indices1d, origin_histogram.shape)


def periodic_angular_conditions(angle: np.ndarray, angular_bins: np.ndarray) -> np.ndarray:
    """Apply periodic boundary conditions to a list of angular coordinates.

    Args:
        angle: A list of angular coordinates.
        angular_bins: An array of bin edges defining the angular grid cells.

    Returns:
        The angular coordinates cast to the angular grid.
    """
    angle -= angular_bins[0]
    angle = angle % (2 * np.pi)
    angle += angular_bins[0]
    return angle


def weighted_mean_of_matrix(field: np.ndarray, histogram: np.ndarray, axes: Tuple = (2, 3, 4)) -> np.ndarray:
    """
    Calculate the weighted mean of a matrix based on a given histogram.

    Parameters:
        field (np.ndarray): The input matrix.
        histogram (np.ndarray): The histogram used for weighting.
        axes (Tuple): The axes along which to calculate the mean. Default is (2, 3, 4).

    Returns:
        np.ndarray: The weighted mean of the matrix.

    """
    weights = histogram / np.sum(histogram)
    values = field

    weighted_sum = np.sum(values * weights, axis=axes)
    sum_of_weights = np.sum(weights, axis=axes)

    weighted_average = weighted_sum / sum_of_weights

    # weighted_field = np.nansum(field * histogram, axis=axes)
    # position_histogram = np.nansum(histogram, axis=axes)
    # weighted_field /= np.where(position_histogram != 0, position_histogram, np.inf)
    return weighted_average
