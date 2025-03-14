import logging
from pathlib import Path

import hydra

from physped.core.parametrize_potential import (
    learn_potential_from_trajectories,
)
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import read_preprocessed_trajectories_from_file
from physped.io.writers import save_piecewise_potential
from physped.utils.config_utils import (
    log_configuration,
    register_new_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../physped/conf", config_name="config"
)
def read_and_preprocess_data(config):
    log_configuration(config)
    preprocessed_trajectories = read_preprocessed_trajectories_from_file(
        config
    )

    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )

    log.info("LEARNING POTENTIAL")
    piecewise_potential = learn_potential_from_trajectories(
        preprocessed_trajectories, config
    )
    save_piecewise_potential(
        piecewise_potential,
        Path.cwd().parent,
        config.filename.piecewise_potential,
    )


if __name__ == "__main__":
    register_new_resolvers()
    read_and_preprocess_data()
