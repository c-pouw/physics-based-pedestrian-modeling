"""Plot trajectories of particles in the metaforum dataset."""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physped.core.functions_to_discretize_grid import (
    create_grid_bins,
    grid_bounds,
    make_grid_selection,
    return_grid_ids,
)
from physped.io.readers import read_piecewise_potential_from_file

log = logging.getLogger(__name__)


def plot_position_trajectories_in_cartesian_coordinates(ax: plt.Axes, df: pd.DataFrame) -> plt.Axes:
    """
    Plot the trajectories of pedestrians in cartesian coordinates.

    Parameters:
    - ax (plt.Axes): The matplotlib Axes object to plot on.
    - df (pd.DataFrame): The DataFrame containing the particle data.

    Returns:
    - ax (plt.Axes): The modified matplotlib Axes object.
    """
    for ped_id in df.Pid.unique():
        dfp = df[df["Pid"] == ped_id]
        # TODO : The color of the starting circle and the trajectory do not match
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
    """
    Apply XY plot style to the given Axes object.

    Parameters:
        ax (plt.Axes): The Axes object to apply the style to.
        params (dict): A dictionary containing the plot parameters.

    Returns:
        plt.Axes: The modified Axes object.

    """
    ax.set_aspect("equal")
    ax.set_xlabel("$x\\; [\\mathrm{m}]$")
    ax.set_ylabel("$y\\; [\\mathrm{m}]$")

    ax.set_xlim(params.trajectory_plot.xlims)
    ax.set_ylim(params.trajectory_plot.ylims)
    return ax


def plot_velocity_trajectories_in_polar_coordinates(ax: plt.Axes, df: pd.DataFrame) -> plt.Axes:
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


def plot_velocity_trajectories_in_cartesian_coordinates(ax: plt.Axes, df: pd.DataFrame) -> plt.Axes:
    """Plot the trajectories of particles in the metaforum dataset."""
    for ped_id in df.Pid.unique():
        dfp = df[df["Pid"] == ped_id]
        ax.plot(
            dfp["uf"],
            dfp["vf"],
            lw=0.9,
            alpha=0.8,
            zorder=0,
            c=f"C{int(ped_id%len(plt.rcParams['axes.prop_cycle'].by_key()['color']))}",
        )

    return ax


def plot_polar_grid(ax: plt.Axes, r_grid: np.ndarray, theta_grid: np.ndarray) -> plt.Axes:
    """
    Plot polar grid lines on a given axes object.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - r_grid (numpy.ndarray): Array of radial grid values.
    - theta_grid (numpy.ndarray): Array of angular grid values.

    Returns:
    - ax (matplotlib.axes.Axes): The modified axes object.
    """
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

    for _, th in enumerate(theta_grid[:-1]):
        ax.plot(np.ones(100) * th, r_range, color="k", linestyle=linestyle, lw=0.6)
        ax.text(th, r_grid[-1] * 1.35, f"{th/np.pi:.1f}$\\pi$", ha="center", va="center")
    ax.set_ylim(0, r_grid[-1])
    return ax


def plot_polar_grid_on_cartesian_plot(ax, r_grid, theta_grid):
    for radius in r_grid:
        circle = plt.Circle((0, 0), radius, color="k", linestyle="dashed", fill=False, lw=0.5, alpha=0.8)
        ax.add_patch(circle)
    for angle in theta_grid:
        x1 = np.cos(angle) * 0.4
        x2 = np.cos(angle) * 10
        y1 = np.sin(angle) * 0.4
        y2 = np.sin(angle) * 10
        ax.plot([x1, x2], [y1, y2], color="k", linestyle="dashed", lw=0.5, alpha=0.8)
    return ax


def apply_polar_plot_style(ax: plt.Axes, params: dict) -> plt.Axes:
    """
    Applies a polar plot style to the given axes object.

    Parameters:
    - ax: The axes object to apply the polar plot style to.
    - params: A dictionary containing parameters for customizing the plot style.

    Returns:
    - The modified axes object.

    """
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


def apply_cartesian_velocity_plot_style(ax: plt.Axes, params: dict) -> plt.Axes:
    """
    Applies a polar plot style to the given axes object.

    Parameters:
    - ax: The axes object to apply the polar plot style to.
    - params: A dictionary containing parameters for customizing the plot style.

    Returns:
    - The modified axes object.

    """
    ax.set_aspect("equal")
    ax.grid(False)
    log.warning("Under construction: Hardcoded grids and limits")
    r_grid = np.arange(0, 4, 0.4)
    theta_grid = np.linspace(-np.pi, np.pi + 0.01, 7)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)

    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("v [m/s]")
    ax = plot_polar_grid_on_cartesian_plot(ax, r_grid, theta_grid)
    return ax


def highlight_grid_box(ax: plt.Axes, limits: Tuple, c: str = "k") -> plt.Axes:
    """
    Highlight the selected grid box.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        limits (Tuple): The limits of the grid box as a tuple (xlims, ylims).
        c (str): The color of the highlight box. Default is "k" (black).

    Returns:
        plt.Axes: The modified matplotlib Axes object.

    """
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


def plot_station_background(ax: plt.Axes, params: dict) -> plt.Axes:
    """
    Plot the background image of the station.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        params (dict): A dictionary containing the parameters for plotting.

    Returns:
        plt.Axes: The modified matplotlib Axes object.

    """
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
    """
    Plot trajectories of pedestrians.

    Args:
        trajs (pd.DataFrame): DataFrame containing the trajectories of pedestrians.
        params (dict): Dictionary containing the plot parameters.
        trajectory_type (str, optional): Type of trajectory. Defaults to None.

    Returns:
        None
    """
    traj_plot_params = params.get("trajectory_plot", {})
    name = params.get("env_name")
    folderpath = Path(params.folder_path)

    plot_title = traj_plot_params.get("title", "")
    num_trajectories_to_plot = traj_plot_params.get("N_trajs", 10)
    sampled_trajectories = trajs.Pid.drop_duplicates().sample(num_trajectories_to_plot)
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
    # TODO Move walls to separate function
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
    # TODO Move intended path to separate function
    if plot_intended_path:
        yps = traj_plot_params.get("yps", [])
        for yp in yps:
            ax.axhline(yp, color="k", ls="dashed", lw=2, zorder=30)
            ax.text(
                1.05,
                yp,
                "$y_p$",
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
    plot_potential_cross_section = traj_plot_params.get("plot_potential_cross_section", False)
    if plot_potential_cross_section and "potential_convolution" in params:
        for axis in ["x", "y"]:
            piecewise_potential = read_piecewise_potential_from_file(folderpath / "model.pickle")
            potential_convolution_params = params.get("potential_convolution", {})
            value = potential_convolution_params[axis]
            bins = piecewise_potential.bins.get(axis)
            idx = return_grid_ids(bins, value)["grid_idx"]
            obs_limits = grid_bounds(bins, axis, idx)
            plot_limits.append(obs_limits)

        ax = highlight_grid_box(ax, plot_limits[::-1])

    ax = fig.add_subplot(spec[1], polar=True)
    ax = apply_polar_plot_style(ax, params)
    ax = plot_velocity_trajectories_in_polar_coordinates(ax, plot_trajs)
    # ax = fig.add_subplot(spec[1])  # , polar=True)
    # ax = apply_cartesian_velocity_plot_style(ax, params)
    # ax = plot_velocity_trajectories_in_cartesian_coordinates(ax, plot_trajs)

    ax.set_title("Velocities $\\vec{u}$ [m/s]", y=1.1)
    plot_selection = traj_plot_params.get("plot_selection", False)
    if plot_selection:
        selection = params.get("selection")
        piecewise_potential = read_piecewise_potential_from_file(folderpath / "model.pickle")
        grid_selection = make_grid_selection(piecewise_potential, selection)
        plot_limits = [grid_selection[obs]["periodic_bounds"] for obs in ["r", "theta"]]

        ax = highlight_grid_box(ax, plot_limits)

    fig.suptitle(plot_title, y=0.83)
    save_figure = traj_plot_params.get("save_figure", False)
    if save_figure:
        filename = folderpath / f"{trajectory_type}trajectories_{params.get('env_name', '')}.pdf"
        log.info("Saving trajectories figure.")
        plt.savefig(filename)
