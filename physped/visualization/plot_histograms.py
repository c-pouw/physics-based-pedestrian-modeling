import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import entropy

from physped.core.parametrize_potential import add_trajectories_to_histogram, digitize_trajectories_to_grid
from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)


def create_histogram(values: pd.Series, bins: np.ndarray) -> dict:
    """
    Create a histogram of the input values.

    Paramters:
    - values (pd.Series): A Pandas Series containing the values to bin.
    - bins (np.ndarray): An array of bin edges.

    Returns:
    - dict: A dictionary containing the bin edges, bin width, bin centers, counts, and PDF of the histogram.
    """
    counts, bin_edges = np.histogram(values, bins)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2
    PDF = counts / (counts.sum() * bin_width)
    return {
        "bin_edges": bin_edges,
        "bin_width": bin_width,
        "bin_centers": bin_centers,
        "counts": counts,
        "PDF": PDF,
    }


def create_all_histograms(
    recorded_paths: pd.DataFrame,
    simulated_paths: pd.DataFrame,
    config: dict,
):
    observables = config.params.histogram_plot.observables
    histograms = {}
    for traj_type, trajectories in zip(["recorded", "simulated"], [recorded_paths, simulated_paths]):
        histograms[traj_type] = {}
        for observable in observables:
            lims = config.params.histogram_plot[f"{observable}lims"]
            bins = np.linspace(lims[0], lims[1], 50)
            values = trajectories[observable]
            histograms[traj_type][observable] = create_histogram(values, bins)
    return histograms


def compute_joint_kl_divergence(
    piecewise_potential: PiecewisePotential,
    simulated_paths: pd.DataFrame,
) -> float:
    # We compare the probability distributions of the actual dynamics for measurements and simulations
    recorded_paths_histogram = piecewise_potential.histogram
    simulated_paths = digitize_trajectories_to_grid(simulated_paths, piecewise_potential.lattice)
    simulated_paths_histogram = np.zeros_like(recorded_paths_histogram)
    simulated_paths_histogram = add_trajectories_to_histogram(simulated_paths_histogram, simulated_paths, "fast_grid_indices")
    recorded_paths_histogram = np.where(recorded_paths_histogram == 0, np.nan, recorded_paths_histogram)
    simulated_paths_histogram = np.where(simulated_paths_histogram == 0, np.nan, simulated_paths_histogram)
    kl = entropy(recorded_paths_histogram, simulated_paths_histogram, nan_policy="omit", axis=(0, 1, 2, 3, 4))
    return kl


def compute_joint_kl_divergence_with_volume(
    piecewise_potential: PiecewisePotential,
    simulated_paths: pd.DataFrame,
) -> float:
    histogram_measurements = piecewise_potential.histogram
    cell_volume = piecewise_potential.lattice.cell_volume

    simulated_paths = digitize_trajectories_to_grid(simulated_paths, piecewise_potential.lattice)
    histogram_simulations = np.zeros_like(histogram_measurements)
    histogram_simulations = add_trajectories_to_histogram(histogram_simulations, simulated_paths, "fast_grid_indices")
    histogram_simulations = np.where(histogram_simulations == 0, np.nan, histogram_simulations)
    histogram_measurements = np.where(histogram_measurements == 0, np.nan, histogram_measurements)

    prob_dist_measurements = histogram_measurements / np.nansum(histogram_measurements)
    prob_dens_measurements = np.divide(prob_dist_measurements, cell_volume)

    prob_dist_simulations = histogram_simulations / np.nansum(histogram_simulations)
    prob_dens_simulations = np.divide(prob_dist_simulations, cell_volume)

    # Compute Kullback-Leibler divergence
    kl = np.nansum(np.multiply(prob_dist_measurements, np.log(np.divide(prob_dens_measurements, prob_dens_simulations))))
    return kl


def plot_histogram(
    ax: Axes,
    histograms: Dict[str, Any],
    observable: str,
    hist_type: str,
    config: dict,
) -> Axes:
    """
    Plot a histogram.

    Parameters:
    - ax (plt.Axes): The axes to plot the histogram on.
    - histograms (Dict[str, Any]): The histograms to plot.
    - observable (str): The observable to plot the histogram for.
    - hist_type (str): The type of histogram to plot.
    - kl_div (float): The KL divergence value for the histogram.

    Returns:
    - The axes object.
    """
    ntrajs = {
        "recorded": config.params.input_ntrajs,
        "simulated": config.params.simulation.ntrajs,
    }
    labels = {
        "recorded": f"Measurements ($N =$ {ntrajs['recorded']})",
        "simulated": f"Simulations ($N =$ {ntrajs['simulated']})",
    }
    histogram_plot_params = config.params.histogram_plot
    for trajectory_type in ["recorded", "simulated"]:
        label = labels[trajectory_type]
        ax.scatter(
            histograms[trajectory_type][observable]["bin_centers"],
            histograms[trajectory_type][observable][hist_type],
            ec=histogram_plot_params[trajectory_type]["edgecolor"],
            fc=histogram_plot_params[trajectory_type]["facecolor"],
            label=label,
            s=histogram_plot_params[trajectory_type]["markersize"],
        )
    ax.set_xlabel(histogram_plot_params[observable]["xlabel"])
    ax.set_ylabel(histogram_plot_params[observable]["ylabel"][hist_type])
    return ax


def save_joint_kl_divergence_to_file(joint_kl_divergence: float, config: dict) -> None:
    kl_divergence = {}
    params = config.params
    kl_divergence["env_name"] = params.env_name
    kl_divergence["ntrajs"] = params.simulation.ntrajs
    kl_divergence["tauu"] = params.model.tauu
    kl_divergence["dt"] = params.model.dt
    kl_divergence["noise"] = params.model.sigma
    kl_divergence["joint_kl_divergence"] = joint_kl_divergence

    with open(Path.cwd() / "kl_divergence.pkl", "wb") as f:
        pickle.dump(kl_divergence, f)


def plot_multiple_histograms(observables: List, histograms: dict, histogram_type: str, config: dict):
    """
    Plot histograms for all observables.

    Parameters:
    - ax (plt.Axes): The axes to plot the histogram on.
    - histograms (dict): The histograms to plot.
    - observable (str): The observable to plot the histogram for.
    - hist_type (str): The type of histogram to plot.
    - kl_div (float): The KL divergence value for the histogram.

    Returns:
    - The axes object.
    """
    params = config.params
    width_single_panel = 1.77
    height_single_panel = 1.18
    subplot_grid = params.histogram_plot.subplot_grid
    fig = plt.figure(figsize=(width_single_panel * subplot_grid[1], height_single_panel * subplot_grid[0]), layout="constrained")
    hist_plot_params = params.histogram_plot

    for plotid, observable in enumerate(observables):
        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], plotid + 1)

        ax = plot_histogram(
            ax,
            histograms,
            observable,
            hist_type=histogram_type,
            config=config,
        )
        xlims = hist_plot_params.get(f"{observable}lims", None)
        ax.set_xlim(xlims)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 2, ymax * 2)
        # props = {"facecolor": "white", "alpha": 1, "edgecolor": "black", "lw": 0.5, "pad": 1.6}
    #     ax.text(
    #         0.215,
    #         0.92,
    #         f"$D_{{\\! K\\! L}}={kldiv:.3f}$",
    #         transform=ax.transAxes,
    #         ha="center",
    #         va="center",
    #         fontsize=6,
    #         bbox=props,
    #     )

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=7, loc="center")
    filepath = Path.cwd() / f"histograms_{params.env_name}.pdf"
    # log.info("Saving histograms figure to %s.", filepath.relative_to(config.root_dir))
    plt.savefig(filepath)
