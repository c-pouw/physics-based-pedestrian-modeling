# This script plots a comparison between the probability distributions of the original and simulated trajectories
import logging

import hydra

from physped.visualization.plot_discrete_grid import plot_discrete_grid

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def plot_grid(cfg):
    plot_discrete_grid(cfg)


if __name__ == "__main__":
    plot_grid()
