import logging
from pathlib import Path

import hydra

from physped.core.functions_to_discretize_grid import cast_trajectories_to_discrete_grid
from physped.io.readers import read_grid_bins, read_preprocessed_trajectories
from physped.io.writers import save_discrete_grid
from physped.utils.functions import ensure_folder_exists

log = logging.getLogger(__name__)

# optional_filters = [pp.filter_trajectories_by_velocity]


@hydra.main(version_base=None, config_path="../conf")
def create_and_save_grid(cfg):
    """Create and save trajectories in grid."""
    # Read parameters and trajectories
    folderpath = Path(cfg.params.folder_path)
    trajectories = read_preprocessed_trajectories(folderpath)

    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    # Cast trajectories to grid
    grid_bins = read_grid_bins(cfg.params.grid_name)
    grids = cast_trajectories_to_discrete_grid(trajectories, grid_bins)

    # Save the grid
    ensure_folder_exists(folderpath=folderpath)
    save_discrete_grid(grids, folderpath)


if __name__ == "__main__":
    create_and_save_grid()
