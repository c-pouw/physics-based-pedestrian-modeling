import logging
from pathlib import Path

import hydra

from physped.utils.functions import create_folder_if_not_exists
from physped.io.readers import trajectory_reader
from physped.preprocessing.trajectory_preprocessor import preprocess_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def preprocess_and_save_trajectories(cfg):
    """Preprocess and save trajectories"""
    log.info("Starting script.")
    name = cfg.params.env_name
    folderpath = Path(cfg.params.folder_path)

    create_folder_if_not_exists(folderpath=folderpath)

    # if ... exists:
    if (folderpath / "preprocessed_trajectories.csv").exists():
        log.info("Preprocessed trajectories already exist. Skipping.")
        return

    trajectories = trajectory_reader[name]()

    # for optional_filter in optional_filters:
    #     trajectories = optional_filter(trajectories)

    # Preprocess trajectories
    trajectories.rename(columns={"Rstep": "time", "Pid": "Pid", "t": "time"}, inplace=True)
    trajectories = preprocess_trajectories(trajectories, parameters=cfg.params)

    filepath = folderpath / "preprocessed_trajectories.csv"
    trajectories.to_csv(filepath, index=False)
    log.info(
        "Saved prepocessed trajectories to %s",
        filepath.relative_to(Path.cwd()),
    )


if __name__ == "__main__":
    preprocess_and_save_trajectories()
