import numpy as np
import pytest

# from physped.core.piecewise_potential_handling import get_most_left_boundary, get_most_right_boundary
from physped.core.functions_to_select_grid_piece import get_index_of_the_enclosing_bin

# bins1 = np.arange(0, 4, 0.5)
# bins2 = np.arange(-1, 3, 0.5)
# test_data = [
#     (bins1, 0, 0, 1),
#     (bins1, 1.0, 2, 3),
#     (bins1, (0.1, 0.4), 0, 1),
#     (bins1, (0.1, 0.6), 0, 2),
#     (bins1, (0.6, 3.6), 1, 1),
#     (bins1, (1.1, 4.1), 2, 2),
#     (bins1, (-0.1, 0.1), 6, 1),
#     (bins1, (-0.6, 0.6), 5, 2),
#     (bins2, 0, 2, 3),
#     (bins2, 1.0, 4, 5),
#     (bins2, (0.1, 0.4), 2, 3),
#     (bins2, (0.1, 0.6), 2, 4),
#     (bins2, (0.6, 3.6), 3, 3),
#     (bins2, (1.1, 4.1), 4, 4),
#     (bins2, (-0.1, 0.1), 1, 3),
#     (bins2, (-0.6, 0.6), 0, 4),
# ]


# @pytest.mark.parametrize("bins,values,expected_left,expected_right", test_data)
# def test_boundary_functions(bins, values, expected_left, expected_right):
#     assert get_most_left_boundary(values, bins) == expected_left
#     assert get_most_right_boundary(values, bins) == expected_right


bins1 = np.arange(0, 4, 0.5)
bins2 = np.arange(-1, 3, 0.5)
test_data = [
    (bins1, 0, 0),
    (bins1, 1.0, 2),
    (bins1, -100, np.nan),
    (bins1, 100, np.nan),
    (bins2, 0, 2),
    (bins2, 1.0, 4),
    (bins2, -1, 0),
    (bins2, -100, np.nan),
    (bins2, 100, np.nan),
]


@pytest.mark.parametrize("bins,selected_value,expected_index", test_data)
def test_point_selection(bins, selected_value, expected_index):
    assert np.allclose(
        get_index_of_the_enclosing_bin(selected_value, bins),
        expected_index,
        equal_nan=True,
    )


test_data = [
    (bins1, (0.1, 0.4), (0, 1)),
    # (bins1, (0.1, 0.6), (0, 2)),
    # (bins1, (0.6, 3.6), (1, 1)),
    # (bins1, (1.1, 4.1), (2, 2)),
    # (bins1, (-0.1, 0.1), (6, 1)),
    # (bins1, (-0.6, 0.6), (5, 2)),
    # (bins2, (0.1, 0.4), (2, 3)),
    # (bins2, (0.1, 0.6), (2, 4)),
    # (bins2, (0.6, 3.6), (3, 3)),
    # (bins2, (1.1, 4.1), (4, 4)),
    # (bins2, (-0.1, 0.1), (1, 3)),
    # (bins2, (-0.6, 0.6), (0, 4)),
]

# def test_range_selection(bins, selected_range, expected_indices):
