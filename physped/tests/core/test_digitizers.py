import numpy as np

from physped.core.digitizers import digitize_coordinates_to_lattice


def test_digitize_coordinates_to_lattice():
    """Test the digitize_coordinates_to_lattice function."""
    coordinates = np.array([-1, 0, 0.4, 1, 5.3])
    lattice_bins = np.array([0, 1, 2])

    expected_indices = np.array([-1, 0, 0, 1, -1])
    result_indices = digitize_coordinates_to_lattice(coordinates, lattice_bins)
    np.testing.assert_array_equal(result_indices, expected_indices)
