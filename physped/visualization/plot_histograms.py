import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.axes import Axes
from scipy.special import kl_div

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


def compute_KL_divergence(PDF1: np.ndarray, PDF2: np.ndarray, bin_width: np.ndarray) -> np.ndarray:
    """
    Compute KL divergence between two probability density functions.

    Parameters:
    - PDF1 (np.ndarray): The first probability density function.
    - PDF2 (np.ndarray): The second probability density function.
    - bin_width (float): The width of the bins used to compute the PDFs.

    Returns:
    - An array of KL divergence values.
    """
    kl = kl_div(PDF1 * bin_width, PDF2 * bin_width)
    return ma.masked_invalid(kl).compressed()


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
    fig = plt.figure(figsize=(3.54, 2.36), layout="constrained")
    sum_kl_div = 0
    hist_plot_params = params.histogram_plot

    for plotid, observable in enumerate(observables):
        ax = fig.add_subplot(2, 2, plotid + 1)

        kldiv = sum(
            compute_KL_divergence(
                histograms["recorded"][observable][histogram_type],
                histograms["simulated"][observable][histogram_type],
                histograms["recorded"][observable]["bin_width"],
            )
        )

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
        props = {"facecolor": "white", "alpha": 1, "edgecolor": "black", "lw": 0.5, "pad": 1.6}
        ax.text(
            0.215,
            0.92,
            f"$D_{{\\! K\\! L}}={kldiv:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            bbox=props,
        )
        sum_kl_div += kldiv

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=7, loc="center")
    filepath = Path.cwd() / f"histograms_{params.env_name}.pdf"
    log.info("Saving histograms figure to %s.", filepath.relative_to(config.root_dir))
    plt.savefig(filepath)
