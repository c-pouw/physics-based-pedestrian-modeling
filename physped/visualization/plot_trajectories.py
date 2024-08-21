"""Plot trajectories of particles in the metaforum dataset."""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from physped.core.digitizers import digitize_coordinates_to_lattice
from physped.core.parametrize_potential import get_boundary_coordinates_of_selection, make_grid_selection
from physped.io.readers import read_piecewise_potential_from_file
from physped.visualization.plot_utils import (  # apply_cartesian_velocity_plot_style,
    apply_polar_plot_style,
    apply_xy_plot_style,
    highlight_grid_box,
    plot_cartesian_spatial_grid,
    plot_station_background,
)

log = logging.getLogger(__name__)

# Colorset from bokeh.palettes.TolRainbow20
trajectory_colorset = [
    "#72190E",
    "#A5170E",
    "#DC050C",
    "#E8601C",
    "#F1932D",
    "#F6C141",
    "#F7F056",
    "#CAE0AB",
    "#90C987",
    "#4EB265",
    "#7BAFDE",
    "#6195CF",
    "#437DBF",
    "#1965B0",
    "#882E72",
    "#994F88",
    "#AA6F9E",
    "#BA8DB4",
    "#CAACCB",
    "#D9CCE3",
]


def plot_position_trajectories_in_cartesian_coordinates(
    ax: plt.Axes, df: pd.DataFrame, alpha: float = 1.0, traj_type: str = "f"
) -> plt.Axes:
    """
    Plot the trajectories of pedestrians in cartesian coordinates.

    Parameters:
    - ax (plt.Axes): The matplotlib Axes object to plot on.
    - df (pd.DataFrame): The DataFrame containing the particle data.

    Returns:
    - ax (plt.Axes): The modified matplotlib Axes object.
    """
    xcol, ycol = f"x{traj_type}", f"y{traj_type}"
    for i, ped_id in enumerate(df.Pid.unique()):
        path = df[df["Pid"] == ped_id]
        color = trajectory_colorset[i % len(trajectory_colorset)]

        # * Plot the starting point of the trajectory
        ax.plot(
            path[xcol].iloc[0],
            path[ycol].iloc[0],
            marker="h",
            markersize=4,
            markeredgecolor=color,
            markerfacecolor="none",
            zorder=10,
            alpha=alpha,
        )

        ax.plot(path[xcol], path[ycol], color=color, lw=0.9, alpha=alpha, zorder=10)
    return ax


def plot_velocity_trajectories_in_polar_coordinates(
    ax: plt.Axes, df: pd.DataFrame, alpha: float = 1.0, traj_type: str = "f"
) -> plt.Axes:
    """Plot the trajectories of particles in the metaforum dataset."""
    thetacol, rcol = f"theta{traj_type}", f"r{traj_type}"
    for i, ped_id in enumerate(df.Pid.unique()):
        dfp = df[df["Pid"] == ped_id]
        ax.plot(
            dfp[thetacol],
            dfp[rcol],
            lw=0.9,
            alpha=alpha,
            zorder=0,
            color=trajectory_colorset[i % len(trajectory_colorset)],
        )

    return ax


def plot_velocity_trajectories_in_cartesian_coordinates(ax: plt.Axes, df: pd.DataFrame) -> plt.Axes:
    """Plot the trajectories of particles in the metaforum dataset."""
    for i, ped_id in enumerate(df.Pid.unique()):
        dfp = df[df["Pid"] == ped_id]
        ax.plot(
            dfp["uf"],
            dfp["vf"],
            lw=0.9,
            alpha=0.8,
            zorder=0,
            # c=f"C{int(ped_id%len(plt.rcParams['axes.prop_cycle'].by_key()['color']))}",
            color=trajectory_colorset[i % len(trajectory_colorset)],
        )

    return ax


def plot_walls_in_environment(ax: plt.Axes, traj_plot_params: dict) -> plt.Axes:
    yfillbetween = [10, -10]
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
    return ax


def intended_path_label_generator(label_count, i):
    if label_count:
        return "$y_{s}$"
    else:
        return f"$y_{{s_{{{i}}}}}$"


def plot_intended_path(ax: plt.Axes, traj_plot_params: dict) -> plt.Axes:
    yps = traj_plot_params.get("yps", [])
    # colors = ["C0", "C1", "C4"]
    colors = ["k", "k", "k"]

    for i, yp in enumerate(yps, start=1):
        label = intended_path_label_generator(len(yps), i)
        ax.axhline(yp, color="k", ls="dashed", lw=1.5, zorder=10)
        ax.text(1.05, yp, label, transform=ax.get_yaxis_transform(), va="center", ha="left", zorder=-10, c=colors[i - 1])
    return ax


def plot_trajectories(trajs: pd.DataFrame, config: dict, trajectory_type: str = None, traj_type="f"):
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
    traj_plot_params = params.trajectory_plot

    num_trajectories_to_plot = traj_plot_params.get("N_trajs", 10)
    num_trajectories_to_plot = min(num_trajectories_to_plot, trajs.Pid.nunique())
    sampled_pids = trajs.Pid.drop_duplicates().sample(num_trajectories_to_plot)
    plot_trajs = trajs[trajs["Pid"].isin(sampled_pids)]

    if traj_plot_params.truncate_trajectories:
        plot_trajs = plot_trajs[plot_trajs["k"] < traj_plot_params.truncated_trajectory_length]

    fig = plt.figure(layout="constrained")
    fig.set_size_inches(traj_plot_params.figsize)
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=traj_plot_params.width_ratios, wspace=0.1, hspace=0.1, figure=fig)

    ax = fig.add_subplot(spec[0])
    if traj_plot_params.plot_cartesian_grid:
        ax.grid(False)
        ax = plot_cartesian_spatial_grid(ax, params.grid, alpha=0.5)
    ax = apply_xy_plot_style(ax, params)
    ax = plot_position_trajectories_in_cartesian_coordinates(ax, plot_trajs, 1, traj_type)
    ax.set_title("Positions $\\vec{x}$ [m]", y=1)
    if traj_plot_params.plot_walls:
        ax = plot_walls_in_environment(ax, traj_plot_params)

    if traj_plot_params.plot_intended_path:
        ax = plot_intended_path(ax, traj_plot_params)

    if traj_plot_params.show_background:
        ax = plot_station_background(ax, config)

    if traj_plot_params.get("customyticklabels", False):
        ax.set_yticks(traj_plot_params.customyticklabels)

    plot_limits = []
    plot_potential_cross_section = traj_plot_params.plot_potential_cross_section
    if plot_potential_cross_section and "potential_convolution" in params:
        for axis in ["x", "y"]:
            piecewise_potential = read_piecewise_potential_from_file(Path.cwd().parent / "piecewise_potential.pickle")
            potential_convolution_params = params.get("potential_convolution", {})
            value = potential_convolution_params[axis]
            bins = piecewise_potential.lattice.bins.get(axis)
            idx = digitize_coordinates_to_lattice(value, bins)
            obs_limits = get_boundary_coordinates_of_selection(bins, axis, idx)
            plot_limits.append(obs_limits)

        ax = highlight_grid_box(ax, plot_limits[::-1])

    # match traj_plot_params.velocity_grid:
    # case "polar":
    ax = fig.add_subplot(spec[1], polar=True)
    ax = apply_polar_plot_style(ax, params)
    ax = plot_velocity_trajectories_in_polar_coordinates(ax, plot_trajs, 1, traj_type)
    # case "cartesian":
    # ax = fig.add_subplot(spec[1])
    # ax = apply_cartesian_velocity_plot_style(ax, params)
    # ax = plot_velocity_trajectories_in_cartesian_coordinates(ax, plot_trajs)

    ax.set_title("Velocities $\\vec{u}\\, [\\mathrm{ms^{-1}}]$", y=1.1)

    if traj_plot_params.plot_selection:
        selection = params.get("selection")
        piecewise_potential = read_piecewise_potential_from_file(Path.cwd().parent / "piecewise_potentail.pickle")
        grid_selection = make_grid_selection(piecewise_potential, selection)
        plot_limits = [grid_selection[obs]["periodic_bounds"] for obs in ["r", "theta"]]
        ax = highlight_grid_box(ax, plot_limits)

    if (traj_plot_params.text_box.show) and (trajectory_type == "simulated"):
        textstr = (
            # f"Model parameters\n"
            f"$\\Delta t=\\,${config.params.model.dt:.3f} s\n"
            f"$\\sigma=\\,${config.params.model.sigma} ms$^{{\\mathdefault{{-3/2}}}}$\n"
            # f"$\\tau_x=\\,${config.params.model.taux:.3f} s\n"
            f"$\\tau=\\,${config.params.model.tauu} s"
        )
        props = {"boxstyle": "round", "facecolor": "white", "alpha": 1, "edgecolor": "black", "lw": 0.5}
        plt.figtext(
            traj_plot_params.text_box.x,
            traj_plot_params.text_box.y,
            textstr,
            ha="center",
            va="center",
            fontsize=5,
            bbox=props,
        )
    traj_type_description = {
        "recorded": "measured",
        "simulated": "simulated",
    }
    if traj_plot_params.plot_title:
        title = f"Sample of {num_trajectories_to_plot} {traj_type_description[trajectory_type]}" f" {traj_plot_params.title}"
        fig.suptitle(title, x=0.5, y=traj_plot_params.y_title, ha="center", va="center")
    filepath = Path.cwd() / f"{trajectory_type}_trajectories_{traj_type}_{params.env_name}.pdf"
    # log.info("Saving trajectory plot to %s.", filepath.relative_to(config.root_dir))
    plt.savefig(filepath)
