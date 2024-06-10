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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    env_name = config.params.env_name
    log.debug("Configuration: \n%s", pprint.pformat(dict(config)))
    log.critical("Environment name: %s", env_name)
    log.info("Working directory %s", Path.cwd())
    log.info("Project root %s", config.root_dir)

    plt.style.use(Path(config.root_dir) / config.params.plot_style)

    log.info("READING TRAJECTORIES")
    trajectories = trajectory_reader[env_name](config)
    log.info("PREPROCESSING TRAJECTORIES")
    preprocessed_trajectories = preprocess_trajectories(trajectories, config=config)
    config.params.input_ntrajs = len(preprocessed_trajectories.Pid.unique())

    log.info("LEARNING POTENTIAL")
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, config)
    if config.save.piecewise_potential and not config.read.piecewise_potential:
        save_piecewise_potential(
            piecewise_potential,
            Path.cwd().parent,
            config.filename.piecewise_potential,
        )

    log.info("SIMULATING TRAJECTORIES")
    simulated_trajectories = simulate_trajectories(piecewise_potential, config)

    log.info("PLOTTING FIGURES")
    # * Optional plotting of preprocessed trajectories
    if config.plot.preprocessed_trajectories:
        log.info("Plot preprocessed trajectories.")
        log.debug("Configuration 'plot.preprocessed_trajectories' is set to True.")
        plot_trajectories(preprocessed_trajectories, config, "recorded")
    else:
        log.warning("Configuration 'plot.preprocessed_trajectories' is set to False.")

    # * Optional plotting of simulated trajectories
    if config.plot.simulated_trajectories:
        log.info("Plot simulated trajectories")
        log.debug("Configuration 'plot.simulated_trajectories' is set to True.")
        plot_trajectories(simulated_trajectories, config, "simulated")
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    # * Optional plotting of probability distributions
    if config.plot.histograms:
        log.info("Plot probability distribution comparison.")
        log.debug("Configuration 'plot.histograms' is set to True.")
        observables = ["xf", "yf", "uf", "vf"]
        config.params.simulation.ntrajs = len(simulated_trajectories.Pid.unique())
        histograms = create_all_histograms(preprocessed_trajectories, simulated_trajectories, config)
        plot_multiple_histograms(observables, histograms, "PDF", config)
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    # * Optional plotting of the grid
    if config.plot.grid:
        log.info("Plot the configuration of the grid.")
        log.debug("Configuration 'plot.grid' is set to True.")
        plot_discrete_grid(config)
    else:
        log.warning("Configuration 'plot.grid' is set to False.")

    output_figures = glob.glob("*.pdf")
    log.info("Pulling output figures to the parent directory for easy access.")
    for figure in output_figures:
        shutil.copyfile(figure, Path.cwd().parent / figure)


if __name__ == "__main__":
    register_new_resolvers()
    main()
