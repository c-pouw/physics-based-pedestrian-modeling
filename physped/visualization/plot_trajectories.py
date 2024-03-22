"""Plot trajectories of particles in the metaforum dataset."""

from typing import Tuple
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from physped.io.readers import read_discrete_grid_from_file
from physped.core.functions_to_discretize_grid import (
    make_grid_selection,
    grid_bounds,
    return_grid_ids,
    create_grid_bins,
)

plt.style.use(
    "/home/pouw/workspace/crowd-tracking/2020-XX-Pouw-Corbetta-pathintegral-codes/physped/visualization/science.mplstyle"
)
log = logging.getLogger(__name__)


def plot_position_trajectories_in_cartesian_coordinates(ax: plt.Axes, df: pd.DataFrame) -> plt.Axes:
    """Plot the trajectories of particles in the metaforum dataset."""
    for ped_id in df.Pid.unique():
        dfp = df[df["Pid"] == ped_id]
        ax.scatter(
            dfp["xf"].iloc[0],
            dfp["yf"].iloc[0],
            fc="none",
            ec=f"C{int(ped_id%len(plt.rcParams['axes.prop_cycle'].by_key()['color']))}",
            zorder=10,
        )
        ax.plot(dfp["xf"], dfp["yf"], lw=0.9, alpha=0.8, zorder=10)
    return ax


def apply_xy_plot_style(ax: plt.Axes, params: dict) -> plt.Axes:
    ax.set_aspect("equal")
    ax.set_xlabel("$x\; [\mathrm{m}]$")
    ax.set_ylabel("$y\; [\mathrm{m}]$")

    ax.set_xlim(params.trajectory_plot.xlims)
    ax.set_ylim(params.trajectory_plot.ylims)
    return ax


def plot_velocity_trajectories_in_polar_coordinates(
    ax,
    df,
    # val, traj_type, N=100
) -> plt.Axes:
    """Plot the trajectories of particles in the metaforum dataset."""
    for ped_id in df.Pid.unique():
        dfp = df[df["Pid"] == ped_id]
        ax.plot(
            dfp["thetaf"],
            dfp["rf"],
            lw=0.9,
            alpha=0.8,
            zorder=0,
            c=f"C{int(ped_id%len(plt.rcParams['axes.prop_cycle'].by_key()['color']))}",
        )

    return ax


def plot_polar_grid(ax, r_grid, theta_grid):
    r_range = np.linspace(r_grid[1], r_grid[-1], 100)
    theta_range = np.linspace(0, 2 * np.pi, 100)
    linestyle = "dashed"
    for r in r_grid:
        if r == 0:
            continue
        ax.plot(theta_range, np.ones(100) * r, color="k", linestyle=linestyle, lw=0.6)
        ax.text(
            np.pi / 2,
            r + 0.2,
            f"{r}",
            ha="center",
            va="center",
            # bbox = dict(
            #     facecolor='white', alpha=0.5,
            #     edgecolor='none', boxstyle='round')
        )

    for thid, th in enumerate(theta_grid[:-1]):
        ax.plot(np.ones(100) * th, r_range, color="k", linestyle=linestyle, lw=0.6)
        ax.text(th, r_grid[-1] * 1.35, f"{th/np.pi:.1f}$\\pi$", ha="center", va="center")
    ax.set_ylim(0, r_grid[-1])
    return ax


def apply_polar_plot_style(ax, params):
    ax.set_aspect("equal")

    polar_grid_type = params["trajectory_plot"].get("polar_grid_type", "standard")

    r_grid = [0, 0.4, 1.1, 1.8]
    theta_grid = np.arange(-np.pi, np.pi + 0.01, np.pi / 3)

    if polar_grid_type == "custom":
        grid_bins = create_grid_bins(params["grid"])
        r_grid = params["grid"]["r"]
        theta_grid = grid_bins["theta"]

    ax.set_yticks([])
    ax.set_xticks([])
    ax = plot_polar_grid(ax, r_grid, theta_grid)

    return ax


def _create_grid_box_limits(bins, lim_idx, obs):
    ### Should be obsolete
    if not obs == "theta":
        lims = [bins[x] for x in lim_idx]
    else:  # The angle is periodic
        lims = list(lim_idx)

        if lims[0] < 0:
            lims[0] = lims[0] * -1
            lims[0] = -1 * bins[lims[0]] - 2 * np.pi
        else:
            lims[0] = bins[lims[0]]

        if lims[1] >= len(bins):
            lims[1] -= len(bins) - 1
            lims[1] = bins[lims[1]] + 2 * np.pi
        else:
            lims[1] = bins[lims[1]]

    return lims


def _create_grid_box_limits(slices: list, dimensions: Tuple, bins: dict, obs: list):
    ### Should be obsolete
    xidx, yidx = (np.where(np.array(dimensions) == obs[i])[0][0] for i in range(2))
    xlims = [bins[obs[0]][x] for x in slices[xidx]]
    ylims = list(slices[yidx])

    if ylims[0] < 0:
        ylims[0] = ylims[0] * -1
        ylims[0] = -1 * bins[obs[1]][ylims[0]] - 2 * np.pi
    else:
        ylims[0] = bins[obs[1]][ylims[0]]

    if ylims[1] >= len(bins[obs[1]]):
        ylims[1] -= len(bins[obs[1]]) - 1
        ylims[1] = bins[obs[1]][ylims[1]] + 2 * np.pi
    else:
        ylims[1] = bins[obs[1]][ylims[1]]

    return xlims, ylims


def highlight_grid_box(ax, limits, c="k"):
    """Highlight the selected grid box."""
    xlims, ylims = limits

    yrange = np.linspace(ylims[0], ylims[1], 100)
    colors = {
        "k": (0, 0, 0, 1),
        "r": (1, 0, 0, 1),
        "g": (0, 1, 0, 1),
        "b": (0, 0, 1, 1),
    }
    args = {
        "fc": (1, 1, 1, 0.6),
        "ec": colors[c],
        "zorder": 10,
        "lw": 1.5,
        "label": "$S$",
    }
    ax.fill_between(yrange, xlims[0], xlims[1], **args)
    return ax


def plot_station_background(ax, params):
    """Plot the background image of the station."""
    config = params.background
    img = mpimg.imread(params.background.imgpath)
    ax.imshow(
        img,
        cmap="gray",
        origin="upper",
        extent=(
            config["xmin"] / 1000,
            config["xmax"] / 1000,
            config["ymin"] / 1000,
            config["ymax"] / 1000,
        ),
        alpha=1,
    )

    ax.set_xlim(params.trajectory_plot.xlims)
    ax.set_ylim(params.trajectory_plot.ylims)
    return ax


def plot_trajectories(trajs: pd.DataFrame, params: dict, trajectory_type: str = None):
    traj_plot_params = params.get("trajectory_plot", {})
    name = params.get("env_name")
    folderpath = Path(params.folder_path)

    plot_title = traj_plot_params.get("title", "")
    N_trajs_to_plot = traj_plot_params.get("N_trajs", 10)
    sampled_trajectories = trajs.Pid.drop_duplicates().sample(N_trajs_to_plot)
    plot_trajs = trajs[trajs["Pid"].isin(sampled_trajectories)]

    fig = plt.figure(layout="constrained")

    width_ratios = traj_plot_params.get("width_ratios", [2, 1])
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=width_ratios, wspace=0.1, hspace=0.1, figure=fig)

    ax = fig.add_subplot(spec[0])
    ax = apply_xy_plot_style(ax, params)
    ax = plot_position_trajectories_in_cartesian_coordinates(ax, plot_trajs)
    ax.set_title("Positions $\\vec{x}$ [m]", y=1)
    plot_walls = traj_plot_params.get("plot_walls", False)
    yfillbetween = [10, -10]
    if plot_walls:
        ywalls = traj_plot_params.get("ywalls", [])
        for i, ywall in enumerate(ywalls):
            ax.axhline(ywall, color="k", ls=(0, (3, 1, 1, 1, 1, 1)), lw=1, zorder=30)
            ax.fill_between(
                [-4, 4],
                ywall,
                yfillbetween[i],
                color="k",
                alpha=0.3,
                zorder=30,
                hatch="//",
            )
            ax.text(
                1.05,
                ywall,
                "$y_{wall}$",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
            )
    plot_intended_path = traj_plot_params.get("plot_intended_path", False)
    if plot_intended_path:
        yps = traj_plot_params.get("yps", [])
        for yp in yps:
            ax.axhline(yp, color="k", ls="dashed", lw=2, zorder=30)
            ax.text(
                1.05,
                yp,
                f"$y_{{p}}$",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
            )
    if name == "single_paths":
        fig.set_size_inches(3.54, 2.36)
    elif name == "parallel_paths":
        fig.set_size_inches(3.54, 5)
    elif name == "station_paths":
        ax = plot_station_background(ax, params)

    if trajectory_type:
        # plot_title = f"{trajectory_type.capitalize()} {plot_title.lower()}"
        trajectory_type = f"{trajectory_type}_"

    plot_limits = []
    grids = read_discrete_grid_from_file(folderpath / "model.pickle")
    plot_potential_cross_section = traj_plot_params.get("plot_potential_cross_section", False)
    if plot_potential_cross_section and "potential_convolution" in params:
        for axis in ["x", "y"]:
            potential_convolution_params = params.get("potential_convolution", {})
            value = potential_convolution_params[axis]
            bins = grids.bins.get(axis)
            idx = return_grid_ids(bins, value)["grid_idx"]
            obs_limits = grid_bounds(bins, axis, idx)
            plot_limits.append(obs_limits)

        ax = highlight_grid_box(ax, plot_limits[::-1])

    ax = fig.add_subplot(spec[1], polar=True)
    ax = apply_polar_plot_style(ax, params)
    ax = plot_velocity_trajectories_in_polar_coordinates(ax, plot_trajs)
    ax.set_title("Velocities $\\vec{u}$ [m/s]", y=1.1)
    plot_selection = traj_plot_params.get("plot_selection", False)
    if plot_selection:
        selection = params.get("selection")
        grids = read_discrete_grid_from_file(folderpath / "model.pickle")
        grid_selection = make_grid_selection(grids, selection)
        plot_limits = [grid_selection[obs]["periodic_bounds"] for obs in ["r", "theta"]]

        ax = highlight_grid_box(ax, plot_limits)

    fig.suptitle(plot_title, y=0.83)
    save_figure = traj_plot_params.get("save_figure", False)
    if save_figure:
        filename = folderpath / f"{trajectory_type}trajectories_{params.get('env_name', '')}.pdf"
        log.info("Saving trajectories figure.")
        plt.savefig(filename)
