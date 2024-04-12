# This script plots a comparison between the probability distributions of the original and simulated trajectories
import glob
import logging
import shutil
from pathlib import Path

import hydra
import matplotlib.pyplot as plt

from physped.visualization.plot_1d_gaussian_fits import learn_piece_of_potential_plot
from physped.visualization.plot_discrete_grid import plot_discrete_grid

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def plot_piecewise_potential_fit(cfg):
    plt.style.use(str(cfg.root_dir / cfg.params.plot_style))
    plot_discrete_grid(cfg)
    learn_piece_of_potential_plot(cfg)
    output_figures = glob.glob("*.pdf")
    for figure in output_figures:
        shutil.copyfile(figure, Path.cwd().parent / figure)


if __name__ == "__main__":
    plot_piecewise_potential_fit()
