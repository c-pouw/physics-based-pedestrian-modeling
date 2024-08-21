import numpy as np
import pytest

from physped.utils.functions import periodic_angular_conditions

angular_bins_1 = np.array([np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2])
angles_1 = np.array([0, np.pi / 2, 5 * np.pi / 2, 3 * np.pi])
expected_angles_1 = np.array([2 * np.pi, np.pi / 2, np.pi / 2, np.pi])

angular_bins_2 = np.array([-np.pi / 2, np.pi / 2, 3 * np.pi / 2])
angles_2 = np.array([-2 * np.pi, 0, 2 * np.pi])
expected_angles_2 = np.array([0, 0, 0])

test_angles = [(angular_bins_1, angles_1, expected_angles_1), (angular_bins_2, angles_2, expected_angles_2)]


@pytest.mark.parametrize("angular_bins,angles,expected_angles", test_angles)
def test_periodic_angular_conditions(angular_bins, angles, expected_angles):
    """Test the periodic_angular_conditions function.

    Args:
        angular_bins: An array with bin edges ranging over a total of 2 * np.pi.
        angles: An array with angles to tests.
        expected_angles: The expected periodic angles.
    """
    periodic_angles = periodic_angular_conditions(angles, angular_bins)
    np.testing.assert_array_equal(periodic_angles, expected_angles)
