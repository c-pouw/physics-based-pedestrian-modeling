import logging
from pathlib import Path

import hydra

from physped.core.pedestrian_simulator import simulate_pedestrians
from physped.io.readers import read_piecewise_potential_from_file
from physped.io.writers import save_trajectories
from physped.utils.config_utils import (
    log_configuration,
    register_new_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../physped/conf", config_name="config"
)
def simulate_from_potential(config):
    log_configuration(config)
    filepath = Path.cwd() / "potentials" / config.filename.piecewise_potential
    piecewise_potential = read_piecewise_potential_from_file(filepath)
    simulated_trajectories = simulate_pedestrians(
        piecewise_potential,
        config,
    )
    save_trajectories(
        simulated_trajectories,
        Path.cwd() / "simulated_trajectories",
        config.filename.simulated_trajectories,
    )


def main():
    register_new_resolvers()
    simulate_from_potential()


if __name__ == "__main__":
    main()
