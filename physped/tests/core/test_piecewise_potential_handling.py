import numpy as np
import pytest

from physped.core.piecewise_potential_handling import get_most_left_boundary, get_most_right_boundary

bins1 = np.arange(0, 4, 0.5)
bins2 = np.arange(-1, 3, 0.5)
test_data = [
    (bins1, 0, 0, 1),
    (bins1, 1.0, 2, 3),
    (bins1, (0.1, 0.4), 0, 1),
    (bins1, (0.1, 0.6), 0, 2),
    (bins1, (0.6, 3.6), 1, 1),
    (bins1, (1.1, 4.1), 2, 2),
    (bins1, (-0.1, 0.1), 6, 1),
    (bins1, (-0.6, 0.6), 5, 2),
    (bins2, 0, 2, 3),
    (bins2, 1.0, 4, 5),
    (bins2, (0.1, 0.4), 2, 3),
    (bins2, (0.1, 0.6), 2, 4),
    (bins2, (0.6, 3.6), 3, 3),
    (bins2, (1.1, 4.1), 4, 4),
    (bins2, (-0.1, 0.1), 1, 3),
    (bins2, (-0.6, 0.6), 0, 4),
]


@pytest.mark.parametrize("bins,values,expected_left,expected_right", test_data)
def test_boundary_functions(bins, values, expected_left, expected_right):
    assert get_most_left_boundary(values, bins) == expected_left
    assert get_most_right_boundary(values, bins) == expected_right
