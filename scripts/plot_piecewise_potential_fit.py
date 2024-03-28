# This script plots a comparison between the probability distributions of the original and simulated trajectories
import logging

import hydra

from physped.visualization.plot_1d_gaussian_fits import plot_1d_gaussian_fits

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def plot_piecewise_potential_fit(cfg):
    plot_1d_gaussian_fits(cfg.params)


if __name__ == "__main__":
    plot_piecewise_potential_fit()
