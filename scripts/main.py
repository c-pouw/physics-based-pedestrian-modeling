import logging

import hydra

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import read_grid_bins, trajectory_reader
from physped.preprocessing.trajectory_preprocessor import preprocess_trajectories
from physped.visualization.histograms import create_all_histograms, plot_multiple_histograms
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def main(cfg):
    name = cfg.params.env_name

    log.info("---- Preprocess recorded trajectories ----")
    trajectories = trajectory_reader[name]()
    preprocessed_trajectories = preprocess_trajectories(trajectories, parameters=cfg.params)

    print("\n")
    log.info("---- Plot recorded trajectories ----")
    plot_trajectories(preprocessed_trajectories, cfg.params, "simulated")

    print("\n")
    log.info("---- Learn piecewise potetential from trajectories ----")
    grid_bins = read_grid_bins(cfg.params.grid_name)
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, grid_bins)

    print("\n")
    log.info("---- Simulate trajectories with piecewise potential ----")
    simulated_trajectories = simulate_trajectories(piecewise_potential, cfg.params)

    print("\n")
    log.info("---- Plot simulated trajectories ----")
    plot_trajectories(simulated_trajectories, cfg.params, "simulated")

    print("\n")
    log.info("---- Plot probability distribution comparison ----")
    observables = ["xf", "yf", "rf", "thetaf"]
    histograms = create_all_histograms(preprocessed_trajectories, simulated_trajectories, observables)
    plot_multiple_histograms(observables, histograms, "PDF", cfg.params)


if __name__ == "__main__":
    main()
