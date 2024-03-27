import logging
from pathlib import Path

import hydra

from physped.io.readers import trajectory_reader
from physped.preprocessing.trajectory_preprocessor import preprocess_trajectories
from physped.utils.functions import ensure_folder_exists

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def preprocess_and_save_trajectories(cfg):
    """Preprocess and save trajectories"""
    log.info("Starting script.")
    name = cfg.params.env_name
    folderpath = Path(cfg.params.folder_path)

    ensure_folder_exists(folderpath=folderpath)

    # if ... exists:
    # if cfg.params.get("force_trajectory_preprocessing", False):
    # if (folderpath / "preprocessed_trajectories.csv").exists():
    #     log.info("Preprocessed trajectories already exist. Skipping.")
    #     return

    trajectories = trajectory_reader[name]()
    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    preprocess_trajectories(trajectories, parameters=cfg.params)


if __name__ == "__main__":
    preprocess_and_save_trajectories()
