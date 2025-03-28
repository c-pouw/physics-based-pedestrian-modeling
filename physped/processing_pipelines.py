import logging
from pathlib import Path

import numpy as np

from physped.core.lattice_selection import evaluate_selection_range
from physped.core.parametrize_potential import (
    learn_potential_from_trajectories,
)
from physped.core.pedestrian_initializer import (
    sample_dynamics_from_trajectories,
)
from physped.core.pedestrian_simulator import simulate_pedestrians
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import (
    read_piecewise_potential_from_file,
    read_trajectories,
    read_trajectories_from_file,
)
from physped.io.writers import save_piecewise_potential, save_trajectories
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.utils.config_utils import (
    log_configuration,
)
from physped.visualization.plot_discrete_grid import plot_discrete_grid
from physped.visualization.plot_histograms import (
    compute_joint_kl_divergence,
    create_all_histograms,
    plot_multiple_histograms,
    save_joint_kl_divergence_to_file,
)
from physped.visualization.plot_potential_at_slow_index import (
    plot_potential_at_slow_index,
)
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


def read_and_preprocess_data(config):
    log_configuration(config)
    trajectories = read_trajectories(config)
    preprocessed_trajectories = preprocess_trajectories(
        trajectories, config=config
    )
    save_trajectories(
        preprocessed_trajectories,
        folderpath=Path.cwd(),
        filename=config.filename.preprocessed_trajectories,
    )


def sample_and_save_dynamics_from_trajectories(config: dict):
    log_configuration(config)
    env_name = config.params.env_name
    n_trajs = config.params.simulation.ntrajs
    state = config.params.simulation.sample_state
    preprocessed_trajectories = read_trajectories_from_file(
        filepath=Path.cwd() / config.filename.preprocessed_trajectories
    )
    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )

    dynamics = sample_dynamics_from_trajectories(
        preprocessed_trajectories, n_trajs, state
    )
    folderpath = Path.cwd() / "initial_dynamics"
    folderpath.mkdir(parents=True, exist_ok=True)
    filename = f"{env_name}_state_{state}_dynamics.npy"
    np.save(folderpath / filename, dynamics)


def learn_potential_from_data(config):
    log_configuration(config)

    preprocessed_trajectories = read_trajectories_from_file(
        filepath=Path.cwd() / config.filename.preprocessed_trajectories
    )
    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )
    piecewise_potential = learn_potential_from_trajectories(
        preprocessed_trajectories, config
    )
    save_piecewise_potential(
        piecewise_potential,
        Path.cwd() / "potentials",
        config.filename.piecewise_potential,
    )


def simulate_from_potential(config):
    log_configuration(config)
    filepath = Path.cwd() / "potentials" / config.filename.piecewise_potential
    piecewise_potential = read_piecewise_potential_from_file(filepath)
    simulated_trajectories = simulate_pedestrians(
        piecewise_potential,
        config,
    )
    save_trajectories(
        simulated_trajectories,
        Path.cwd() / "simulated_trajectories",
        config.filename.simulated_trajectories,
    )


def plot_figures(config):
    log_configuration(config)
    preprocessed_trajectories = read_trajectories_from_file(
        filepath=Path.cwd() / config.filename.preprocessed_trajectories
    )
    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )
    config.params.input_ntrajs = len(preprocessed_trajectories.Pid.unique())
    piecewise_potential = read_piecewise_potential_from_file(
        filepath=Path.cwd()
        / "potentials"
        / config.filename.piecewise_potential
    )
    simulated_trajectories = read_trajectories_from_file(
        filepath=Path.cwd()
        / "simulated_trajectories"
        / config.filename.simulated_trajectories
    )

    if config.plot.preprocessed_trajectories:
        log.info("Plot preprocessed trajectories.")
        plot_trajectories(preprocessed_trajectories, config, "recorded")
        plot_trajectories(
            preprocessed_trajectories, config, "recorded", traj_type="s"
        )

    if config.plot.simulated_trajectories:
        log.info("Plot simulated trajectories")
        config.params.trajectory_plot.plot_intended_path = False
        plot_trajectories(simulated_trajectories, config, "simulated")
        plot_trajectories(
            simulated_trajectories, config, "simulated", traj_type="s"
        )

    if config.plot.histograms:
        log.info("Plot probability distribution comparison.")
        observables = ["xf", "yf", "uf", "vf"]
        config.params.simulation.ntrajs = len(
            simulated_trajectories.Pid.unique()
        )
        histograms = create_all_histograms(
            preprocessed_trajectories, simulated_trajectories, config
        )
        joint_kl_divergence = compute_joint_kl_divergence(
            piecewise_potential, simulated_trajectories
        )
        save_joint_kl_divergence_to_file(joint_kl_divergence, config)
        plot_multiple_histograms(observables, histograms, "PDF", config)

    config = evaluate_selection_range(config)
    selection = config.params.selection.range
    slow_indices = (
        selection.x_indices[0],
        selection.y_indices[0],
        selection.r_indices[0],
        selection.theta_indices[0],
        selection.k_indices[0],
    )

    if config.plot.grid:
        log.info("Plot the configuration of the grid.")
        plot_discrete_grid(config, slow_indices, preprocessed_trajectories)

    if config.plot.potential_at_selection:
        log.info("Plot potential at selection.")
        plot_potential_at_slow_index(config, slow_indices, piecewise_potential)
