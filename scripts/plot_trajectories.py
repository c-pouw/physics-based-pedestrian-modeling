import logging
from pathlib import Path

import hydra
import pandas as pd

from physped.io.readers import read_trajectories_from_path, trajectory_reader
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def plot_and_save_trajectories(cfg):
    # Read parameters
    # params = pp.read_parameter_file(name)
    name = cfg.params.env_name
    trajectory_type = cfg.params.trajectory_plot.trajectory_type
    # Read raw and preprocess trajectories
    if trajectory_type == "recorded":
        if name == "ehv_azure":
            datelist = pd.date_range(start="2023-10-01", end="2023-10-04", freq="1h")
            trajs = trajectory_reader[name](datelist[0])
        else:
            trajs = read_trajectories_from_path(Path.cwd().parent / "preprocessed_trajectories.csv")
            # trajs = pp.trajectory_reader[name]()
            # print(trajs.columns)

        # trajs.rename(columns={"Rstep": "time", "Pid": "Pid"}, inplace=True)
        # trajs = pp.preprocess_trajectories(trajs, params)
    elif trajectory_type == "simulated":
        # Read simulated trajectories
        trajs = pd.read_csv(Path.cwd().parent / "simulated_trajectories.csv")

    # Plot trajectories
    plot_trajectories(trajs, cfg.params, trajectory_type)


if __name__ == "__main__":
    plot_and_save_trajectories()
