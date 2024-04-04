import logging
import pprint
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd

from physped.core.functions_to_discretize_grid import (
    create_grid_bins_from_config,
    learn_potential_from_trajectories,
)
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import read_grid_bins, trajectory_reader
from physped.io.writers import save_piecewise_potential
from physped.preprocessing.trajectory_preprocessor import preprocess_trajectories
from physped.visualization.plot_discrete_grid import plot_discrete_grid
from physped.visualization.plot_histograms import create_all_histograms, plot_multiple_histograms
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    env_name = cfg.params.env_name
    log.debug("Configuration: \n%s", pprint.pformat(dict(cfg)))
    log.debug("Working directory %s", Path.cwd())
    log.debug("Project root %s", get_original_cwd())

    plt.style.use(Path(get_original_cwd()) / cfg.params.plot_style)

    trajectories = trajectory_reader[env_name]()
    preprocessed_trajectories = preprocess_trajectories(trajectories, config=cfg)

    # * Optional plotting of preprocessed trajectories
    if cfg.plot.preprocessed_trajectories:
        print("\n")
        log.info("---- Plot preprocessed trajectories ----")
        log.debug("Configuration 'plot.preprocessed_trajectories' is set to True.")
        plot_trajectories(preprocessed_trajectories, cfg, "recorded")
    else:
        log.warning("Configuration 'plot.preprocessed_trajectories' is set to False.")

    print("\n")
    if cfg.read.grid.from_file:
        log.debug("Configuration 'read.grid.from_file' is set to True.")
        log.info(" ---- Create grid bins from configuration file ----")
        log.warning(
            "Note: this is only needed for grids with nonuniform bin sizes "
            "otherwise it is adviced to create the bins from the configuration."
        )
        grid_bins = read_grid_bins(cfg.read.grid.filename)
    else:
        log.info(" ---- Create grid bins from configuration file ----")
        grid_bins = create_grid_bins_from_config(cfg)

    print("\n")
    log.info("---- Learn piecewise potential from trajectories ----")
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, grid_bins, cfg)
    if cfg.save.piecewise_potential:
        save_piecewise_potential(piecewise_potential, Path.cwd().parent)

    print("\n")
    simulated_trajectories = simulate_trajectories(piecewise_potential, cfg)

    # * Optional plotting of simulated trajectories
    if cfg.plot.simulated_trajectories:
        print("\n")
        log.info("---- Plot simulated trajectories ----")
        log.debug("Configuration 'plot.simulated_trajectories' is set to True.")
        plot_trajectories(simulated_trajectories, cfg, "simulated")
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    # * Optional plotting of probability distributions
    if cfg.plot.histograms:
        print("\n")
        log.info("---- Plot probability distribution comparison ----")
        log.debug("Configuration 'plot.histograms' is set to True.")
        observables = ["xf", "yf", "rf", "thetaf"]
        histograms = create_all_histograms(preprocessed_trajectories, simulated_trajectories, observables)
        plot_multiple_histograms(observables, histograms, "PDF", cfg)
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    # * Optional plotting of the grid
    if cfg.plot.grid:
        print("\n")
        log.info("---- Plot grid ----")
        log.debug("Configuration 'plot.grid' is set to True.")
        plot_discrete_grid(cfg)
    else:
        log.info("Configuration 'plot.grid' is set to False.")


if __name__ == "__main__":
    main()
