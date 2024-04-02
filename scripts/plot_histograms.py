# This script plots a comparison between the probability distributions of the original and simulated trajectories
import logging
from pathlib import Path

import hydra
import pandas as pd

from physped.io.readers import read_trajectories_from_path, trajectory_reader
from physped.visualization.histograms import create_all_histograms, plot_multiple_histograms

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def plot_and_save_histograms(cfg):
    name = cfg.params.env_name

    # Read raw and preprocess trajectories
    if name == "ehv_azure":
        datelist = pd.date_range(start="2023-10-01", end="2023-10-04", freq="1h")
        trajs = trajectory_reader[name](datelist[0])
    else:
        trajs = read_trajectories_from_path(Path.cwd().parent / "preprocessed_trajectories.csv")

    # Read simulated trajectories
    simtrajs = pd.read_csv(Path.cwd().parent / "simulated_trajectories.csv")

    # Create histograms
    observables = ["xf", "yf", "rf", "thetaf"]
    histograms = create_all_histograms(trajs, simtrajs, observables)

    # Plot Histograms
    plot_multiple_histograms(observables, histograms, "PDF", cfg.params)


if __name__ == "__main__":
    plot_and_save_histograms()
