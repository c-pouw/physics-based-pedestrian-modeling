import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from physped.core.functions_to_select_grid_piece import evaluate_selection_range
from physped.io.readers import read_trajectories_from_path
from physped.visualization.plot_trajectories import (
    plot_position_trajectories_in_cartesian_coordinates,
    plot_velocity_trajectories_in_polar_coordinates,
)
from physped.visualization.plot_utils import (
    apply_polar_plot_style,
    apply_xy_plot_style,
    highlight_position_selection,
    highlight_velocity_selection,
    plot_cartesian_spatial_grid,
    plot_polar_labels,
    plot_polar_velocity_grid,
)

log = logging.getLogger(__name__)


def plot_discrete_grid(config: dict):
    params = config.params
    config = evaluate_selection_range(config)
    plot_params = config.params.grid_plot
    if plot_params.plot_trajs:
        filepath = Path.cwd().parent / config.filename.preprocessed_trajectories
        preprocessed_trajectories = read_trajectories_from_path(filepath)
        pids_to_plot = preprocessed_trajectories.Pid.drop_duplicates().sample(plot_params.N_trajs)
        plot_trajs = preprocessed_trajectories[preprocessed_trajectories.Pid.isin(pids_to_plot)]

    fig = plt.figure(layout="constrained")
    spec = mpl.gridspec.GridSpec(
        ncols=2, nrows=1, width_ratios=plot_params.subplot_width_ratio, wspace=0.1, hspace=0.1, figure=fig
    )

    # * Subplot left: spatial grid
    ax1 = fig.add_subplot(spec[0])
    ax1 = apply_xy_plot_style(ax1, params)
    ax1 = plot_cartesian_spatial_grid(ax1, params.grid)
    if plot_params.plot_trajs:
        ax1 = plot_position_trajectories_in_cartesian_coordinates(ax1, plot_trajs, alpha=plot_params.alpha, traj_type="s")
    ax1.set_xlabel(plot_params.position.xlabel)
    ax1.set_ylabel(plot_params.position.ylabel)
    ax1.set_xlim(params.grid.bins.x[0], params.grid.bins.x[-1])
    ax1.set_ylim(params.grid.bins.y[0], params.grid.bins.y[-1])
    ax1.set_aspect("equal")
    ax1.grid(False)
    ax1.set_title(plot_params.title.position, y=1)

    if plot_params.get("customyticklabels", False):
        ax1.set_yticks(plot_params.customyticklabels)

    # * Subplot right: velocity grid
    ax2 = fig.add_subplot(spec[1], polar=True)
    ax2 = apply_polar_plot_style(ax2, params)
    ax2 = plot_polar_velocity_grid(ax2, params.grid)
    ax2 = plot_polar_labels(ax2, params.grid)
    if plot_params.plot_trajs:
        ax2 = plot_velocity_trajectories_in_polar_coordinates(ax2, plot_trajs, alpha=plot_params.alpha, traj_type="s")
    ax2.set_ylim(params.grid.bins.r[0], params.grid.bins.r[-2])
    ax2.grid(False)
    ax2.set_title(plot_params.title.velocity, y=1)

    if plot_params.highlight_selection:
        ax1 = highlight_position_selection(ax1, params)
        ax2 = highlight_velocity_selection(ax2, params)

    fig.suptitle(plot_params.title.figure, y=0.9)
    filepath = Path.cwd() / (params.grid.name + ".pdf")
    plt.savefig(filepath, bbox_inches="tight")
    log.info("Saving plot of the grid to %s.", filepath.relative_to(config.root_dir))
