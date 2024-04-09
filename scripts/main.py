import glob
import logging
import pprint
import shutil
from pathlib import Path

import hydra
import matplotlib.pyplot as plt

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import trajectory_reader
from physped.io.writers import save_piecewise_potential
from physped.omegaconf_resolvers import register_new_resolvers
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.visualization.plot_discrete_grid import plot_discrete_grid
from physped.visualization.plot_histograms import create_all_histograms, plot_multiple_histograms
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    env_name = cfg.params.env_name
    log.debug("Configuration: \n%s", pprint.pformat(dict(cfg)))
    log.info("Working directory %s", Path.cwd())
    log.info("Project root %s", cfg.root_dir)

    plt.style.use(Path(cfg.root_dir) / cfg.params.plot_style)

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
    log.info("---- Learn piecewise potential from trajectories ----")
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, cfg)
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
        observables = ["xf", "yf", "uf", "vf"]
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

    output_figures = glob.glob("*.pdf")
    for figure in output_figures:
        shutil.copyfile(figure, Path.cwd().parent / figure)


if __name__ == "__main__":
    register_new_resolvers()
    main()
