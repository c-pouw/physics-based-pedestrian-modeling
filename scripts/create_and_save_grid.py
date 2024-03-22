import logging
from pathlib import Path

import hydra

from physped.io.writers import save_discrete_grid
from physped.io.readers import read_preprocessed_trajectories, read_grid_bins
from physped.core.functions_to_discretize_grid import trajectories_to_grid
from physped.utils.functions import create_folder_if_not_exists

log = logging.getLogger(__name__)

# optional_filters = [pp.filter_trajectories_by_velocity]


@hydra.main(version_base=None, config_path="../conf")
def create_and_save_grid(cfg):
    """Create and save trajectories in grid."""
    # Read parameters and trajectories
    # params = pp.read_parameter_file(name)
    folderpath = Path(cfg.params.folder_path)
    trajectories = read_preprocessed_trajectories(folderpath)

    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    # Cast trajectories to grid
    # grid_bins = pp.create_grid_bins(params["grid"])
    grid_bins = read_grid_bins(cfg.params.grid_name)
    grids = trajectories_to_grid(trajectories, grid_bins)

    # Save the grid
    # filepath = pp.create_filepath(params=params)
    create_folder_if_not_exists(folderpath=folderpath)
    # pp.save_parameters(parameters=cfg.params, folderpath=folderpath)  # TODO: Hydra automatically saves the config
    save_discrete_grid(grids, folderpath)


if __name__ == "__main__":
    create_and_save_grid()
