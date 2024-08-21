import glob
import logging
import shutil
from pathlib import Path
from pprint import pformat

import hydra
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from physped.core.lattice_selection import evaluate_selection_range
from physped.core.parametrize_potential import learn_potential_from_trajectories
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import trajectory_reader
from physped.io.writers import save_piecewise_potential
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.utils.config_utils import register_new_resolvers
from physped.visualization.plot_discrete_grid import plot_discrete_grid
from physped.visualization.plot_histograms import (
    compute_joint_kl_divergence,
    create_all_histograms,
    plot_multiple_histograms,
    save_joint_kl_divergence_to_file,
)
from physped.visualization.plot_potential_at_slow_index import plot_potential_at_slow_index
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def model(config):
    env_name = config.params.env_name
    log.debug("Configuration: \n%s", pformat(dict(config)))
    log.critical("Environment name: %s", env_name)
    log.info("Working directory %s", Path.cwd())
    log.info("Project root %s", config.root_dir)

    plt.style.use(Path(config.root_dir) / config.plot_style)

    log.info("READING TRAJECTORIES")
    trajectories = trajectory_reader[env_name](config)
    log.info("PREPROCESSING TRAJECTORIES")
    preprocessed_trajectories = preprocess_trajectories(trajectories, config=config)

    # TODO Check if input_ntrajs is still needed
    config.params.input_ntrajs = len(preprocessed_trajectories.Pid.unique())

    logging.info("MODELING PARAMETERS: \n%s", pformat(OmegaConf.to_container(config.params.model, resolve=True), depth=1))

    log.info("PROCESSING SLOW MODES")
    preprocessed_trajectories = compute_slow_dynamics(preprocessed_trajectories, config=config)

    log.info("LEARNING POTENTIAL")
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, config)
    if config.save.piecewise_potential and not config.read.piecewise_potential:
        save_piecewise_potential(
            piecewise_potential,
            Path.cwd().parent,
            config.filename.piecewise_potential,
        )

    log.info("SIMULATING TRAJECTORIES")
    simulated_trajectories = simulate_trajectories(piecewise_potential, config, preprocessed_trajectories)

    log.info("PLOTTING FIGURES")
    # * Optional plotting of preprocessed trajectories
    if config.plot.preprocessed_trajectories:
        log.info("Plot preprocessed trajectories.")
        log.debug("Configuration 'plot.preprocessed_trajectories' is set to True.")
        plot_trajectories(preprocessed_trajectories, config, "recorded")
        plot_trajectories(preprocessed_trajectories, config, "recorded", traj_type="s")
    else:
        log.warning("Configuration 'plot.preprocessed_trajectories' is set to False.")

    # * Optional plotting of simulated trajectories
    if config.plot.simulated_trajectories:
        log.info("Plot simulated trajectories")
        log.debug("Configuration 'plot.simulated_trajectories' is set to True.")
        config.params.trajectory_plot.plot_intended_path = False
        plot_trajectories(simulated_trajectories, config, "simulated")
        plot_trajectories(simulated_trajectories, config, "simulated", traj_type="s")
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    # * Optional plotting of probability distributions
    if config.plot.histograms:
        log.info("Plot probability distribution comparison.")
        log.debug("Configuration 'plot.histograms' is set to True.")
        observables = ["xf", "yf", "uf", "vf"]
        config.params.simulation.ntrajs = len(simulated_trajectories.Pid.unique())
        histograms = create_all_histograms(preprocessed_trajectories, simulated_trajectories, config)
        joint_kl_divergence = compute_joint_kl_divergence(piecewise_potential, simulated_trajectories)
        save_joint_kl_divergence_to_file(joint_kl_divergence, config)
        plot_multiple_histograms(observables, histograms, "PDF", config)
    else:
        log.warning("Configuration 'plot.simulated_trajectories' is set to False.")

    config = evaluate_selection_range(config)
    selection = config.params.selection.range
    slow_indices = (
        selection.x_indices[0],
        selection.y_indices[0],
        selection.r_indices[0],
        selection.theta_indices[0],
        selection.k_indices[0],
    )

    # * Optional plotting of the grid
    if config.plot.grid:
        log.info("Plot the configuration of the grid.")
        log.debug("Configuration 'plot.grid' is set to True.")
        plot_discrete_grid(config, slow_indices, preprocessed_trajectories)
    else:
        log.warning("Configuration 'plot.grid' is set to False.")

    # * Optional plotting of the potential
    if config.plot.potential_at_selection:
        log.info("Plot potential at selection.")
        log.debug("Configuration 'plot.potential_at_selection' is set to True.")
        plot_potential_at_slow_index(config, slow_indices, piecewise_potential)
    else:
        log.warning("Configuration 'plot.potential_at_selection' is set to False.")

    output_figures = glob.glob("*.pdf")
    log.info("Pulling output figures to the parent directory for easy access.")
    for figure in output_figures:
        shutil.copyfile(figure, Path.cwd().parent / figure)


def main():
    register_new_resolvers()
    model()


if __name__ == "__main__":
    main()
