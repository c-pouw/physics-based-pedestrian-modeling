import logging
from pathlib import Path

import pandas as pd
import hydra

from physped.io.readers import trajectory_reader, read_preprocessed_trajectories
from physped.visualization.histograms import (
    create_all_histograms,
    plot_multiple_histograms,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def plot_and_save_histograms(cfg):
    # Read parameters
    # params = pp.read_parameter_file(name)
    folderpath = Path(cfg.params.folder_path)
    name = cfg.params.env_name

    # Read raw and preprocess trajectories
    if name == "ehv_azure":
        datelist = pd.date_range(start="2023-10-01", end="2023-10-04", freq="1h")
        trajs = trajectory_reader[name](datelist[0])
    else:
        trajs = read_preprocessed_trajectories(folderpath)

    # Read simulated trajectories
    simtrajs = pd.read_csv(folderpath / "simulated_trajectories.csv")

    # Create histograms
    observables = ["xf", "yf", "rf", "thetaf"]
    histograms = create_all_histograms(trajs, simtrajs, observables)

    # Plot Histograms
    plot_multiple_histograms(observables, histograms, "PDF", cfg.params)


if __name__ == "__main__":
    plot_and_save_histograms()
