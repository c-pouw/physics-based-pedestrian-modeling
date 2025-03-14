import logging
from pathlib import Path

import hydra

from physped.io.readers import read_trajectories
from physped.io.writers import save_trajectories
from physped.preprocessing.trajectories import preprocess_trajectories
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
    trajectories = read_trajectories(config)
    preprocessed_trajectories = preprocess_trajectories(
        trajectories, config=config
    )
    save_trajectories(
        preprocessed_trajectories,
        folderpath=Path.cwd().parent,
        filename=config.filename.preprocessed_trajectories,
    )


if __name__ == "__main__":
    register_new_resolvers()
    read_and_preprocess_data()
