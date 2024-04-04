import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from hydra.utils import get_original_cwd
from matplotlib.axes import Axes
from scipy.special import kl_div

log = logging.getLogger(__name__)

histogram_plot_params = {
    "xf": {
        "xlabel": r"$x_f\;$[m]",
        "ylabel": {"counts": "Count($x_f$)", "PDF": "P($x_f$)"},
    },
    "yf": {
        "xlabel": r"$y_f\;$[m]",
        "ylabel": {"counts": "Count($y_f$)", "PDF": "P($y_f$)"},
    },
    "uf": {
        "xlabel": r"$u_f\;$[m/s]",
        "ylabel": {"counts": "Count($u_f$)", "PDF": "P($u_f$)"},
    },
    "vf": {
        "xlabel": r"$v_f\;$[m/s]",
        "ylabel": {"counts": "Count($v_f$)", "PDF": "P($v_f$)"},
    },
    "rf": {
        "xlabel": r"$r_f\;$[m/s]",
        "ylabel": {"counts": "Count($r_f$)", "PDF": "P($r_f$)"},
    },
    "thetaf": {
        "xlabel": "$\\theta_f\\;$[m/s]",
        "ylabel": {"counts": "Count($\\theta_f$)", "PDF": "P($\\theta_f$)"},
    },
    "raw": {
        "edgecolor": "C3",
        "facecolor": "C3",
        "marker": "o",
        "markersize": 8,
        "label": "Recordings",
    },
    "sim": {
        "edgecolor": "k",
        "facecolor": None,
        "marker": "o",
        "markersize": 4,
        "label": "Simulations",
    },
}


def create_automatic_bins(values: pd.Series) -> np.ndarray:
    """
    Create bins for the trajectories.

    Parameters:
    - values (pd.Series): A Pandas Series containing the values to bin.

    Returns:
    - np.ndarray: An array of bin edges.
    """
    # Nbins = int(np.sqrt(len(values)))
    Nbins = 50
    return np.linspace(values.min(), values.max(), Nbins)


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
    trajs: pd.DataFrame,
    simtrajs: pd.DataFrame,
    observables: Optional[List[str]] = None,
):
    if observables is None:
        observables = ["xf", "yf", "uf", "vf", "rf", "thetaf"]
    histograms = {}
    bin_generator = create_automatic_bins
    for traj_type, trajectories in zip(["raw", "sim"], [trajs, simtrajs]):
        histograms[traj_type] = {}
        for observable in observables:
            values = trajectories[observable]
            if traj_type == "raw":
                if observable == "rf":
                    bins = np.linspace(0, 3, 50)
                # elif observable == "thetaf":
                #     bins = np.linspace(0, 2 * np.pi, 100)
                else:
                    bins = bin_generator(values)
            else:
                bins = histograms["raw"][observable]["bin_edges"]
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
    hist_plot_params = params.get("histogram_plot", {})

    for plotid, observable in enumerate(observables):
        ax = fig.add_subplot(2, 2, plotid + 1)
        # ax = fig.add_subplot(1, len(observables), plotid + 1)

        kldiv = sum(
            compute_KL_divergence(
                histograms["raw"][observable][histogram_type],
                histograms["sim"][observable][histogram_type],
                histograms["raw"][observable]["bin_width"],
            )
        )

        ax = plot_histogram(
            ax,
            histograms,
            observable,
            hist_type=histogram_type,
        )
        lims = hist_plot_params.get(f"{observable[0]}lims", None)
        ax.set_xlim(lims)

        sum_kl_div += kldiv

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=7, loc="center")
    # plt.suptitle(
    #     f"Parameters: $\qquad \\tau_x = {params['taux']} \qquad \\tau_u = {params['tauu']} \qquad "
    #     f"dt = {params['dt']} \qquad \sigma = {params['sigma']} \qquad \\sum{{D_{{KL}}}} = {sum_kl_div:.2f}$",
    #     fontsize=16,
    # )
    # fig.text(-0.02, 0.5, "PDF", rotation=90)
    filepath = Path.cwd() / f"histograms_{params.get('env_name', '')}.pdf"
    log.info("Saving histograms figure to %s.", filepath.relative_to(get_original_cwd()))
    plt.savefig(filepath)


def plot_histogram(
    ax: Axes,
    histograms: Dict[str, Any],
    observable: str,
    hist_type: str,
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
    for traj_type in ["raw", "sim"]:
        ax.scatter(
            histograms[traj_type][observable]["bin_centers"],
            histograms[traj_type][observable][hist_type],
            ec=histogram_plot_params[traj_type]["edgecolor"],
            fc=histogram_plot_params[traj_type]["facecolor"],
            label=histogram_plot_params[traj_type]["label"],
            s=histogram_plot_params[traj_type]["markersize"],
        )
    # ax.set_title(f"$D_{{KL}}={kl_div:.2f}$")
    ax.set_xlabel(histogram_plot_params[observable]["xlabel"])
    ax.set_ylabel(histogram_plot_params[observable]["ylabel"][hist_type])
    return ax
