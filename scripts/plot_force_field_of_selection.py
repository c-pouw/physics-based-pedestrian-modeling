import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

import physped as pp
from physped.visualization.plot_fields import (
    plot_force_field_of_selection,
)

log = logging.getLogger(__name__)


def main(name: str):
    # Read discrete grid
    selection = {
        "x": None,
        "y": None,
        "r": [0.9, 1.1],
        "theta": [(0.1 * np.pi), (0.2 * np.pi) - 0.01],
        "k": [0, 2],
    }
    params = pp.read_parameter_file(name)
    # params["force_field_plot"] = {"clip": 0, "scale": 800, "sparseness": 3}
    filepath = pp.create_filepath(params)
    grids = pp.read_discrete_grid_from_file(filepath)

    plot_force_field_of_selection(grids, params, selection)
    plt.savefig(f"figures/{name}_force_field_of_selection.pdf")


if __name__ == "__main__":
    input_name = sys.argv[1]
    main(input_name)
