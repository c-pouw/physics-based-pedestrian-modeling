import logging
from pathlib import Path

import hydra

from physped.core.lattice_selection import evaluate_selection_range
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import (
    read_piecewise_potential_from_file,
    read_trajectories_from_file,
)
from physped.utils.config_utils import (
    log_configuration,
    register_new_resolvers,
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


@hydra.main(
    version_base=None, config_path="../physped/conf", config_name="config"
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


def main():
    register_new_resolvers()
    plot_figures()


if __name__ == "__main__":
    main()
