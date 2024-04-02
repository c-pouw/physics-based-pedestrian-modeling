import logging
import pprint
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import read_grid_bins, trajectory_reader
from physped.io.writers import save_piecewise_potential
from physped.preprocessing.trajectory_preprocessor import preprocess_trajectories
from physped.visualization.histograms import create_all_histograms, plot_multiple_histograms
from physped.visualization.plot_trajectories import plot_trajectories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    name = cfg.params.env_name
    log.debug("Configuration: \n%s", pprint.pformat(dict(cfg)))
    log.debug("Working directory %s", Path.cwd())
    log.debug("Project root %s", get_original_cwd())

    plt.style.use(Path(get_original_cwd()) / cfg.params.plot_style)
    log.info("---- Preprocess recorded trajectories ----")
    trajectories = trajectory_reader[name]()
    preprocessed_trajectories = preprocess_trajectories(trajectories, parameters=cfg.params)

    print("\n")
    log.info("---- Plot preprocessed trajectories ----")
    plot_trajectories(preprocessed_trajectories, cfg.params, "recorded")

    print("\n")
    log.info("---- Learn piecewise potetential from trajectories ----")
    grid_bins = read_grid_bins(cfg.params.grid_name)
    piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, grid_bins)
    save_piecewise_potential(piecewise_potential, Path.cwd().parent)

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
