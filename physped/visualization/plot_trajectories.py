"""Plot trajectories of particles in the metaforum dataset."""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from hydra.utils import get_original_cwd

from physped.core.functions_to_discretize_grid import (
    get_boundary_coordinates_of_selection,
    make_grid_selection,
    return_grid_ids,
)
from physped.io.readers import read_piecewise_potential_from_file
from physped.visualization.plot_utils import (
    apply_cartesian_velocity_plot_style,
    apply_polar_plot_style,
    apply_xy_plot_style,
    highlight_grid_box,
    plot_station_background,
)

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


def plot_trajectories(trajs: pd.DataFrame, config: dict, trajectory_type: str = None):
    """
    Plot trajectories of pedestrians.

    Args:
        trajs (pd.DataFrame): DataFrame containing the trajectories of pedestrians.
        params (dict): Dictionary containing the plot parameters.
        trajectory_type (str, optional): Type of trajectory. Defaults to None.

    Returns:
        None
    """
    params = config.params
    traj_plot_params = params.get("trajectory_plot", {})
    name = params.get("env_name")

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
    # TODO retrieve size from config
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
            piecewise_potential = read_piecewise_potential_from_file(
                Path.cwd().parent / "piecewise_potential.pickle"
            )
            potential_convolution_params = params.get("potential_convolution", {})
            value = potential_convolution_params[axis]
            bins = piecewise_potential.bins.get(axis)
            idx = return_grid_ids(bins, value)["grid_idx"]
            obs_limits = get_boundary_coordinates_of_selection(bins, axis, idx)
            plot_limits.append(obs_limits)

        ax = highlight_grid_box(ax, plot_limits[::-1])

    match traj_plot_params.velocity_grid:
        case "polar":
            ax = fig.add_subplot(spec[1], polar=True)
            ax = apply_polar_plot_style(ax, params)
            ax = plot_velocity_trajectories_in_polar_coordinates(ax, plot_trajs)
        case "cartesian":
            ax = fig.add_subplot(spec[1])
            ax = apply_cartesian_velocity_plot_style(ax, params)
            ax = plot_velocity_trajectories_in_cartesian_coordinates(ax, plot_trajs)

    ax.set_title("Velocities $\\vec{u}$ [m/s]", y=1.1)
    plot_selection = traj_plot_params.get("plot_selection", False)
    if plot_selection:
        selection = params.get("selection")
        piecewise_potential = read_piecewise_potential_from_file(Path.cwd().parent / "piecewise_potentail.pickle")
        grid_selection = make_grid_selection(piecewise_potential, selection)
        plot_limits = [grid_selection[obs]["periodic_bounds"] for obs in ["r", "theta"]]

        ax = highlight_grid_box(ax, plot_limits)

    fig.suptitle(plot_title, y=0.83)
    filepath = Path.cwd() / f"{trajectory_type}trajectories_{params.get('env_name', '')}.pdf"
    log.info("Saving trajectory plot to %s.", filepath.relative_to(get_original_cwd()))
    plt.savefig(filepath)
