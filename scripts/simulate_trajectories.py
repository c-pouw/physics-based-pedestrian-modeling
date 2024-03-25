import logging
from pathlib import Path

import hydra

from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import read_discrete_grid_from_file

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def sim_and_save_trajs(cfg):
    """Simulates and saves trajectories based on the given configuration.

    :param cfg: The configuration object containing simulation parameters.
    :type cfg: object
    """
    folderpath = Path(cfg.params.folder_path)

    discrete_potential = read_discrete_grid_from_file(folderpath / "model.pickle")

    simulate_trajectories(discrete_potential, cfg.params)


if __name__ == "__main__":
    sim_and_save_trajs()
