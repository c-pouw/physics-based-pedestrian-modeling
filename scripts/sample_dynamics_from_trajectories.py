import logging
from pathlib import Path

import hydra
import numpy as np

from physped.core.pedestrian_initializer import (
    sample_dynamics_from_trajectories,
)
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import read_trajectories_from_file
from physped.utils.config_utils import (
    log_configuration,
    register_new_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../physped/conf", config_name="config"
)
def sample_and_save_dynamics_from_trajectories(config: dict):
    log_configuration(config)
    env_name = config.params.env_name
    n_trajs = config.params.simulation.ntrajs
    state = config.params.simulation.sample_state
    preprocessed_trajectories = read_trajectories_from_file(
        filepath=Path.cwd() / config.filename.preprocessed_trajectories
    )
    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )

    dynamics = sample_dynamics_from_trajectories(
        preprocessed_trajectories, n_trajs, state
    )
    folderpath = Path.cwd() / "initial_dynamics"
    folderpath.mkdir(parents=True, exist_ok=True)
    filename = f"{env_name}_state_{state}_dynamics.npy"
    np.save(folderpath / filename, dynamics)


def main():
    register_new_resolvers()
    sample_and_save_dynamics_from_trajectories()


if __name__ == "__main__":
    main()
