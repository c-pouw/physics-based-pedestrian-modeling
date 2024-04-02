# This script plots a comparison between the probability distributions of the original and simulated trajectories
import logging

import hydra

from physped.visualization.plot_discrete_grid import plot_discrete_grid

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def plot_grid(cfg):
    # name = cfg.params.env_name
    # selection = cfg.params.get("selection")

    # grid_selection = make_grid_selection(discrete_potential, selection)
    plot_discrete_grid(cfg.params)


if __name__ == "__main__":
    plot_grid()
