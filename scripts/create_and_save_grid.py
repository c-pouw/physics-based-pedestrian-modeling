# %%
import logging
from pathlib import Path

import hydra

import physped as pp

log = logging.getLogger(__name__)

# %%

# optional_filters = [pp.filter_trajectories_by_velocity]


@hydra.main(version_base=None, config_path="../conf")
def create_and_save_grid(cfg):
    """Create and save trajectories in grid."""
    # Read parameters and trajectories
    # params = pp.read_parameter_file(name)
    folderpath = Path(cfg.params.folder_path)
    trajectories = pp.read_preprocessed_trajectories(folderpath)

    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    # Cast trajectories to grid
    # grid_bins = pp.create_grid_bins(params["grid"])
    grid_bins = pp.read_grid_bins(cfg.params.grid_name)
    grids = pp.trajectories_to_grid(trajectories, grid_bins)

    # Save the grid
    # filepath = pp.create_filepath(params=params)
    pp.create_folder_if_not_exists(folderpath=folderpath)
    # pp.save_parameters(parameters=cfg.params, folderpath=folderpath)  # TODO: Hydra automatically saves the config
    pp.save_discrete_grid(grids, folderpath)


# %%

if __name__ == "__main__":
    create_and_save_grid()

# %%
