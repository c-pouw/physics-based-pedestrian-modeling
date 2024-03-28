import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np

from physped.io.readers import read_piecewise_potential_from_file
from physped.visualization.plot_fields import plot_force_field_of_selection

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def main(cfg):
    # Read discrete grid
    selection = {
        "x": None,
        "y": None,
        "r": [0.9, 1.1],
        "theta": [(0.1 * np.pi), (0.2 * np.pi) - 0.01],
        "k": [0, 2],
    }
    # params["force_field_plot"] = {"clip": 0, "scale": 800, "sparseness": 3}
    folderpath = Path(cfg.params.folder_path)
    name = cfg.params.env_name
    grids = read_piecewise_potential_from_file(folderpath / "piecewise_potential.pickle")

    plot_force_field_of_selection(grids, cfg.params, selection)
    plt.savefig(f"figures/{name}_force_field_of_selection.pdf")


if __name__ == "__main__":
    main()
