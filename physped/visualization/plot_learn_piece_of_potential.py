# import glob
import logging

# import shutil
from pathlib import Path

# import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from physped.core.lattice_selection import evaluate_selection_range
from physped.core.parametrize_potential import digitize_trajectories_to_grid
from physped.core.piecewise_potential import PiecewisePotential

# from physped.utils.config_utils import register_new_resolvers
# from physped.visualization.plot_discrete_grid import plot_discrete_grid

log = logging.getLogger(__name__)


def learn_piece_of_potential_plot(config: dict, preprocessed_trajectories: pd.DataFrame, piecewise_potential: PiecewisePotential):
    plot_params = config.params.learn_piece_of_potential_plot
    config = evaluate_selection_range(config)
    # filepath = Path.cwd().parent / config.filename.piecewise_potential
    # piecewise_potential = read_piecewise_potential_from_file(filepath)
    # piecewise_potential = read_piecewise_potential_from_file(Path.cwd().parent / "piecewise_potential.pickle")
    # selection = config.params.get("selection")
    # log.info("Selection: %s", selection)

    # selection_with_bins = \
    # [[selection[d][0], piecewise_potential.bins[d]] for d in piecewise_potential.dimensions]
    # grid_selection_by_indices = \
    # [get_most_left_boundary(v, b) for v, b in selection_with_bins]
    # log.info("Grid selection by indices: %s", grid_selection_by_indices)

    # filepath = Path.cwd().parent / config.filename.preprocessed_trajectories
    # preprocessed_trajectories = pd.read_csv(filepath)
    # trajs = pd.read_csv(Path.cwd().parent / "preprocessed_trajectories.csv")
    trajs = digitize_trajectories_to_grid(preprocessed_trajectories, piecewise_potential.lattice)

    parametrization = piecewise_potential.parametrization[
        config.params.selection.range.x_indices[0],
        config.params.selection.range.y_indices[0],
        config.params.selection.range.r_indices[0],
        config.params.selection.range.theta_indices[0],
        config.params.selection.range.k_indices[0],
        :,
        :,
    ]

    # fit_params = piecewise_potential.fit_params[
    #     grid_selection_by_indices[0],
    #     grid_selection_by_indices[1],
    #     grid_selection_by_indices[2],
    #     grid_selection_by_indices[3],
    #     grid_selection_by_indices[4],
    #     :,
    # ]
    log.info("Fit parameters: %s", parametrization)
    if np.nansum(parametrization) == 0.0:
        log.error("No data for this selection. Exiting.")
        return

    grid_selection_by_indices = [
        config.params.selection.range.x_indices[0],
        config.params.selection.range.y_indices[0],
        config.params.selection.range.r_indices[0],
        config.params.selection.range.theta_indices[0],
        config.params.selection.range.k_indices[0],
    ]
    points_inside_grid_cell = trajs[trajs.slow_grid_indices == tuple(grid_selection_by_indices)]

    fig = plt.figure(layout="constrained")
    fit_dimensions = piecewise_potential.dist_approximation.fit_dimensions
    for axis in fit_dimensions:
        fit_dimension_index = fit_dimensions.index(axis)
        ax = plt.subplot(2, 2, fit_dimension_index + 1)

        fit_param_index = fit_dimension_index * 2
        # ! This will break if the parametrization is not mu, sigma
        mu, variance = parametrization[fit_param_index : (fit_param_index + 2)]
        sigma = np.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = norm.pdf(x, mu, sigma)
        (gauss_line,) = ax.plot(x, y, c="C3", zorder=10, lw=1.5)

        nbins = 50
        hist_bins = np.linspace(plot_params.xlimits[axis][0], plot_params.xlimits[axis][1], nbins)
        dbin = hist_bins[1] - hist_bins[0]
        middle_bins = (hist_bins[1:] + hist_bins[:-1]) / 2

        hist_conditioned, _ = np.histogram(points_inside_grid_cell[f"{axis}f"], bins=hist_bins)
        hist_unconditioned, _ = np.histogram(trajs[f"{axis}f"], bins=hist_bins)

        # plt.plot(middle_bins, hist_conditioned / (np.sum(hist_unconditioned) * dbin), c="C0", zorder=5, lw=1.5)
        (pdf_line,) = ax.plot(middle_bins, hist_unconditioned / (np.sum(hist_unconditioned) * dbin), c="C1", zorder=5, lw=1.5)
        fast_hist = plt.hist(
            points_inside_grid_cell[f"{axis}f"],
            bins=hist_bins,
            density=True,
            alpha=1,
            ec="k",
            fc="#77AADD",
        )
        # pdf = plt.hist(
        #     trajs[f"{axis}f"], bins=hist_bins, alpha=0.5, ec="k", fc="#88CCEE"
        # )
        fast_hist_patches = fast_hist[2][0]

        ax.set_xlim(plot_params.xlimits[axis])
        ax.set_xlabel(plot_params.xlabel[axis])
        ax.set_ylabel(plot_params.ylabel[axis])

    lines = [fast_hist_patches, gauss_line, pdf_line]
    labels = [
        "Conditioned $\\mathbb{P}(\\vec{x}, \\vec{u} \\,|\\, \\vec{x}_s, \\vec{u}_s)$",
        "Fit of $\\mathbb{P}(\\vec{x}, \\vec{u} \\,|\\, \\vec{x}_s, \\vec{u}_s)$",
        "Unconditioned $\\mathbb{P}(\\vec{x}, \\vec{u})$",
    ]
    plt.figlegend(lines, labels, loc="center", bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure, ncol=2)

    filepath = Path.cwd() / "gaussian_fits_1d.pdf"
    plt.savefig(filepath, bbox_inches="tight")
    log.info("Saved plot to %s", filepath.relative_to(config.root_dir))


# @hydra.main(version_base=None, config_path="../../conf", config_name="config")
# def plot_piecewise_potential_fit(cfg):
#     plt.style.use(str(cfg.root_dir / cfg.plot_style))
#     plot_discrete_grid(cfg)
#     learn_piece_of_potential_plot(cfg)
#     output_figures = glob.glob("*.pdf")
#     for figure in output_figures:
#         shutil.copyfile(figure, Path.cwd().parent / figure)


# if __name__ == "__main__":
#     register_new_resolvers()
#     plot_piecewise_potential_fit()
