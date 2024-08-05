import logging
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from physped.core.parametrize_potential import calculate_position_based_emperic_potential, extract_submatrix
from physped.core.piecewise_potential import PiecewisePotential
from physped.visualization.plot_utils import (
    apply_polar_plot_style,
    apply_xy_plot_style,
    highlight_velocity_selection,
    plot_polar_velocity_grid,
)

log = logging.getLogger(__name__)


def plot_potential_at_slow_index(config: DictConfig, slow_indices: List, piecewise_potential: PiecewisePotential):
    params = config.params
    traj_plot_params = params.trajectory_plot
    fig = plt.figure(layout="constrained")
    fig.set_size_inches(traj_plot_params.figsize)
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=traj_plot_params.width_ratios, wspace=0.1, hspace=0.1, figure=fig)

    ax = fig.add_subplot(spec[0])

    plot_params = config.params.force_field_plot
    cmap = "YlOrRd"
    xbin_middle = (config.params.grid.bins.x[1:] + config.params.grid.bins.x[:-1]) / 2
    ybin_middle = (config.params.grid.bins.y[1:] + config.params.grid.bins.y[:-1]) / 2
    X, Y = np.meshgrid(xbin_middle, ybin_middle, indexing="ij")

    slicing_indices = [
        [0, len(config.params.grid.bins.x) - 1],
        [0, len(config.params.grid.bins.y) - 1],
        [slow_indices[2], slow_indices[2] + 1],
        [slow_indices[3], slow_indices[3] + 1],
        [slow_indices[4], slow_indices[4] + 1],
    ]
    slow_subhistogram = extract_submatrix(piecewise_potential.histogram_slow, slicing_indices)
    position_based_emperic_potential = calculate_position_based_emperic_potential(slow_subhistogram, config)
    # matrix_to_plot = get_position_based_emperic_potential_from_state(config, slicing_indices, piecewise_potential)
    # X_indx = get_index_of_state(state, piecewise_potential)

    subparameterrization = extract_submatrix(piecewise_potential.parametrization, slicing_indices)
    center_x = subparameterrization[:, :, 0, 0, 0, 0, 0]
    center_y = subparameterrization[:, :, 0, 0, 0, 1, 0]
    curvature_x = subparameterrization[:, :, 0, 0, 0, 0, 1]
    curvature_y = subparameterrization[:, :, 0, 0, 0, 1, 1]

    # center_u = sliced_fit_parameters[:, :, 0, 0, 0, 4]
    # center_v = sliced_fit_parameters[:, :, 0, 0, 0, 6]
    # sliced_curvature_x = get_slice_of_multidimensional_matrix(piecewise_potential.curvature_x, slices)
    # sliced_curvature_y = get_slice_of_multidimensional_matrix(piecewise_potential.curvature_y, slices)

    curvature_scaling = 1
    curv_x = (curvature_x * (X - center_x)) / curvature_scaling
    curv_y = (curvature_y * (Y - center_y)) / curvature_scaling

    # fig, ax = plt.subplots()

    scale = plot_params.scale
    sparseness = plot_params.sparseness
    minimum_threshold = 1

    # sliced_histogram = extract_submatrix(piecewise_potential.histogram_slow, slicing_indices)
    plot_curv_x = np.where(slow_subhistogram[:, :, 0, 0, 0] < minimum_threshold, np.nan, curv_x)
    plot_curv_y = np.where(slow_subhistogram[:, :, 0, 0, 0] < minimum_threshold, np.nan, curv_y)

    ax.pcolormesh(X, Y, position_based_emperic_potential, cmap=cmap, shading="auto")  # , norm=norm)
    # ax = plot_colorbar(ax, cs)

    ax.quiver(
        X[::sparseness, ::sparseness],
        Y[::sparseness, ::sparseness],
        -plot_curv_x[::sparseness, ::sparseness],
        -plot_curv_y[::sparseness, ::sparseness],
        scale=scale,
        pivot="mid",
        width=0.0015,
        #     labelpos="E",
        #     label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
    )

    # ax = plot_quiverkey(ax, q)
    ax = apply_xy_plot_style(ax, params)
    # ax.set_aspect("equal")
    # ax.set_xlim(config.params.default_xlims)
    # ax.set_ylim(config.params.default_ylims)

    ax2 = fig.add_subplot(spec[1], polar=True)
    ax2 = apply_polar_plot_style(ax2, params)
    ax2 = plot_polar_velocity_grid(ax2, params.grid)
    # ax2 = plot_polar_labels(ax2, params.grid)
    # if plot_params.plot_trajs:
    # ax2 = plot_velocity_trajectories_in_polar_coordinates(ax2, plot_trajs, alpha=plot_params.alpha, traj_type="f")
    # ax2.set_ylim(params.grid.bins.r[0], params.grid.bins.r[-1])
    ax2.grid(False)
    # ax2.set_title(plot_params.title.velocity, y=1)

    # if plot_params.highlight_selection:
    # ax1 = highlight_position_selection(ax1, params)
    ax2 = highlight_velocity_selection(ax2, params)

    filepath = Path.cwd() / "potential_plot_at_slow_index.pdf"
    plt.savefig(filepath, bbox_inches="tight")
    # log.info("Saving plot of the grid to %s.", filepath.relative_to(config.root_dir))

    # plot_trajectories_on_field = False
    # if plot_trajectories_on_field:
    # ax.plot(traj.xf, traj.yf, ms=10, zorder=20, c = 'C0', lw = 0.5)
    # ax.plot(traj['xs'], traj['ys'], ms=10, zorder=20, linestyle = 'dashed', c = 'C1', lw = 0.5)


# %%
