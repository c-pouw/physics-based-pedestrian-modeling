from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from physped.core.functions_to_discretize_grid import digitize_trajectories_to_grid
from physped.core.piecewise_potential_handling import get_most_left_boundary
from physped.io.readers import read_piecewise_potential_from_file, read_trajectories_from_path


def plot_1d_gaussian_fits(params):
    folderpath = Path(params.folder_path)
    piecewise_potential = read_piecewise_potential_from_file(folderpath / "piecewise_potential.pickle")
    selection = params.get("selection")
    selection = [[selection[d][0], piecewise_potential.bins[d]] for d in piecewise_potential.dimensions]

    grid_selection_by_indices = [get_most_left_boundary(v, b) for v, b in selection]

    trajs = read_trajectories_from_path(folderpath / "preprocessed_trajectories.csv")
    trajs = digitize_trajectories_to_grid(piecewise_potential.bins, trajs)

    fit_params = piecewise_potential.fit_params[
        grid_selection_by_indices[0],
        grid_selection_by_indices[1],
        grid_selection_by_indices[2],
        grid_selection_by_indices[3],
        grid_selection_by_indices[4, :],
    ]
    points_inside_grid_cell = trajs[trajs.slow_grid_indices == tuple(grid_selection_by_indices)]

    plt.figure(layout="constrained")
    for axis in piecewise_potential.dimensions[:4]:
        ax = plt.subplot(2, 2, piecewise_potential.dimensions.index(axis) + 1)
        axis_index = piecewise_potential.dimensions.index(axis)
        mu = fit_params[axis_index * 2]
        variance = fit_params[axis_index * 2 + 1]
        sigma = np.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = norm.pdf(x, mu, sigma)
        (gauss_line,) = ax.plot(x, y, "r", zorder=10)

        axis_to_obs = {"x": "xf", "y": "yf", "r": "uf", "theta": "vf"}
        fast_hist = plt.hist(points_inside_grid_cell[f"{axis}f"], bins=30, density=True, alpha=0.5)
        fast_hist_patches = fast_hist[2][0]

        if axis in ["x"]:
            ax.set_xlim(-0.5, 0.5)
        elif axis in ["y"]:
            ax.set_xlim(-0.5, 0.5)
        elif axis in ["theta"]:
            ax.set_xlim(-0.5, 0.5)
        elif axis in ["r"]:
            ax.set_xlim(0.5, 1.5)

        ax.set_xlabel(f"${axis_to_obs[axis][0]}$")

    lines = [fast_hist_patches, gauss_line]
    labels = ["$P(F)$", "Gaussian fit of $P(F)$"]
    plt.figlegend(lines, labels, loc="upper left", bbox_to_anchor=(0.05, 1.2), ncol=3)
    plt.savefig("figures/gaussian_fits_1d.pdf", bbox_inches="tight")
