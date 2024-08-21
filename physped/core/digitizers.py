import numpy as np


def digitize_coordinates_to_lattice(coordinates: np.ndarray[float], lattice_bins: np.ndarray[float]) -> np.ndarray[np.int_]:
    """Digitizes the given coordinates to the specified lattice bins.

    Boundary conditions:
    - Coordinates outside the lattice return -1.
    Note: we can not return nan because the output is an array of integers.

    Args:
        coordinates: The coordinates in one dimension to be digitized.
        lattice_bins: The bin edges in one dimension defining the lattice cells.

    Returns:
        The array with integer lattice indices associated with the coordinates.
    """
    indices = np.digitize(coordinates, lattice_bins) - 1
    smallest_index = 0
    biggest_index = len(lattice_bins) - 2
    indices = np.where(indices < smallest_index, -1, indices)
    indices = np.where(indices > biggest_index, -1, indices)
    return indices
