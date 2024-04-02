import logging
from pathlib import Path

import hydra

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.io.readers import read_grid_bins, read_trajectories_from_path
from physped.io.writers import save_piecewise_potential

# from physped.utils.functions import ensure_folder_exists

log = logging.getLogger(__name__)

# optional_filters = [pp.filter_trajectories_by_velocity]


@hydra.main(version_base=None, config_path="../conf")
def learn_piecewise_potential(cfg):
    """Create and save trajectories in grid."""
    try:
        preprocessed_trajectories = read_trajectories_from_path(
            Path.cwd().parent / "preprocessed_trajectories.csv"
        )
    except FileNotFoundError as e:
        log.error("Prepprocessed trajectories not found: %s", e)

    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    # Cast trajectories to grid
    grid_bins = read_grid_bins(cfg.params.grid_name)
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, grid_bins)

    # Save the grid
    save_piecewise_potential(piecewise_potential, Path.cwd().parent)


if __name__ == "__main__":
    learn_piecewise_potential()
