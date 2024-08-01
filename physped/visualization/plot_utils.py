import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from physped.io.readers import read_background_image

log = logging.getLogger(__name__)


def apply_xy_plot_style(ax: plt.Axes, params: DictConfig) -> plt.Axes:
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


def apply_polar_plot_style(ax: plt.Axes, params: DictConfig) -> plt.Axes:
    """
    Applies a polar plot style to the given axes object.

    Parameters:
    - ax: The axes object to apply the polar plot style to.
    - params: A dictionary containing parameters for customizing the plot style.

    Returns:
    - The modified axes object.

    """
    ax.set_aspect("equal")

    ax.set_yticks([])
    ax.set_xticks([])
    ax = plot_polar_velocity_grid(ax, params.grid)
    ax = plot_polar_labels(ax, params.grid)
    ax.set_ylim(0, params.grid.bins.r[-1])
    return ax


# def plot_polar_grid(ax: plt.Axes, r_grid: np.ndarray, theta_grid: np.ndarray) -> plt.Axes:
#     """
#     Plot polar grid lines on a given axes object.

#     Parameters:
#     - ax (matplotlib.axes.Axes): The axes object to plot on.
#     - r_grid (numpy.ndarray): Array of radial grid values.
#     - theta_grid (numpy.ndarray): Array of angular grid values.

#     Returns:
#     - ax (matplotlib.axes.Axes): The modified axes object.
#     """
#     r_range = np.linspace(r_grid[1], r_grid[-1], 100)
#     theta_range = np.linspace(0, 2 * np.pi, 100)
#     linestyle = "dashed"
#     for r in r_grid:
#         if r == 0:
#             continue
#         ax.plot(theta_range, np.ones(100) * r, color="k", linestyle=linestyle, lw=0.6)
#         ax.text(
#             np.pi / 2,
#             r + 0.2,
#             f"{r}",
#             ha="center",
#             va="center",
#             # bbox = dict(
#             #     facecolor='white', alpha=0.5,
#             #     edgecolor='none', boxstyle='round')
#         )

#     for _, th in enumerate(theta_grid[:-1]):
#         ax.plot(np.ones(100) * th, r_range, color="k", linestyle=linestyle, lw=0.6)
#         ax.text(th, r_grid[-1] * 1.3, f"{th/np.pi:.1f}$\\pi$", ha="center", va="center")
#     ax.set_ylim(0, r_grid[-1])
#     return ax


# def plot_polar_grid_on_cartesian_plot(ax, r_grid, theta_grid):
#     for radius in r_grid:
#         circle = plt.Circle((0, 0), radius, color="k", linestyle="dashed", fill=False, lw=0.5, alpha=0.8)
#         ax.add_patch(circle)
#     for angle in theta_grid:
#         x1 = np.cos(angle) * 0.4
#         x2 = np.cos(angle) * 10
#         y1 = np.sin(angle) * 0.4
#         y2 = np.sin(angle) * 10
#         ax.plot([x1, x2], [y1, y2], color="k", linestyle="dashed", lw=0.5, alpha=0.8)
#     return ax


# def apply_cartesian_velocity_plot_style(ax: plt.Axes, params: dict) -> plt.Axes:
#     """
#     Applies a polar plot style to the given axes object.

#     Parameters:
#     - ax: The axes object to apply the polar plot style to.
#     - params: A dictionary containing parameters for customizing the plot style.

#     Returns:
#     - The modified axes object.

#     """
#     ax.set_aspect("equal")
#     rgrid = params.grid.bins.r
#     thetagrid = params.grid.bins.theta
#     ax.set_xlim(-rgrid[-1], rgrid[-1])
#     ax.set_ylim(-rgrid[-1], rgrid[-1])

#     ax.set_xlabel(params.trajectory_plot.axis_labels.polar.x)
#     ax.set_ylabel(params.trajectory_plot.axis_labels.polar.y)
#     if params.trajectory_plot.plot_polar_grid:
#         ax.grid(False)
#         ax = plot_polar_grid_on_cartesian_plot(ax, rgrid, thetagrid)
#     return ax


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


def plot_station_background(ax: plt.Axes, config: DictConfig) -> plt.Axes:
    """
    Plot the background image of the station.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        params (dict): A dictionary containing the parameters for plotting.

    Returns:
        plt.Axes: The modified matplotlib Axes object.

    """
    bg_source = config.params.background.bg_source
    image = read_background_image[bg_source](config)
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        extent=(
            config.params.background["xmin"] / 1000,
            config.params.background["xmax"] / 1000,
            config.params.background["ymin"] / 1000,
            config.params.background["ymax"] / 1000,
        ),
        alpha=config.params.background.alpha,
    )
    return ax


def plot_cartesian_spatial_grid(ax: plt.Axes, grid_params: DictConfig, alpha: float = 0.8) -> plt.Axes:
    xbins = grid_params.bins.x
    ybins = grid_params.bins.y
    linestyle = "dashed"
    color = "k"
    linewidth = 0.6
    ax.vlines(
        xbins,
        ymin=ybins[0],
        ymax=ybins[-1],
        color=color,
        linestyle=linestyle,
        lw=linewidth,
        alpha=alpha,
    )
    ax.hlines(
        ybins,
        xmin=xbins[0],
        xmax=xbins[-1],
        color=color,
        linestyle=linestyle,
        lw=linewidth,
        alpha=alpha,
    )
    return ax


def plot_polar_velocity_grid(ax: plt.Axes, grid_params: DictConfig) -> plt.Axes:
    rbins = grid_params.bins.r
    thetabins = grid_params.bins.theta
    linestyle = "dashed"
    alpha = 0.8
    color = "k"
    linewidth = 0.6
    if len(rbins) > 2:
        for r in rbins[:-1]:
            ax.plot(
                np.linspace(0, 2 * np.pi, 100),
                np.ones(100) * r,
                color=color,
                linestyle=linestyle,
                lw=linewidth,
                alpha=alpha,
            )
    if len(thetabins) > 2:
        for theta in thetabins:
            ax.plot(
                [theta, theta],
                [rbins[1], rbins[-1]],
                color=color,
                alpha=alpha,
                linestyle=linestyle,
                linewidth=linewidth,
            )
    return ax


def plot_polar_labels(ax: plt.Axes, grid_params: DictConfig) -> plt.Axes:
    rbins = grid_params.bins.r

    for r in rbins:
        if r == 0:
            continue
        textangle = 5 * np.pi / 8
        ax.text(
            textangle,
            r,
            f"{r:.1f}",
            ha="center",
            va="center",
            fontsize=5,
            rotation=convert_rad_to_deg(textangle - np.pi / 2),
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round", pad=0.1),
        )

    winddirections = {
        1: zip([np.pi / 2, 0, -np.pi / 2, np.pi], ["N", "E", "S", "W"], [5, 5, 5, 5]),
        4: zip([np.pi / 2, 0, -np.pi / 2, np.pi], ["N", "E", "S", "W"], [5, 5, 5, 5]),
        8: zip(
            [np.pi / 2, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4, np.pi, 3 * np.pi / 4],
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            [5, 3, 5, 3, 5, 3, 5, 3],
        ),
    }
    if grid_params.theta.segments in winddirections:
        for theta, winddirection, fontsize in winddirections[grid_params.theta.segments]:
            ax.text(
                theta,
                rbins[-1],
                winddirection,
                ha="center",
                va="center",
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=1, edgecolor="k", boxstyle="circle,pad=0.4", lw=0.4),
            )

    return ax


def convert_rad_to_deg(theta: float) -> float:
    return theta * 180 / np.pi


def highlight_position_selection(ax: plt.Axes, params: DictConfig) -> plt.Axes:
    x_bounds = params.selection.range.x_bounds
    y_bounds = params.selection.range.y_bounds
    xrange = np.linspace(x_bounds[0], x_bounds[1], 100)
    c = "r"
    colors = {
        "k": (0, 0, 0, 1),
        "r": (1, 0, 0, 1),
        "g": (0, 1, 0, 1),
        "b": (0, 0, 1, 1),
    }
    args = {
        "fc": (1, 1, 1, 0),
        "ec": colors[c],
        "zorder": 10,
        "lw": 1.5,
        "label": "$S$",
    }
    ax.fill_between(xrange, y_bounds[0], y_bounds[1], **args)
    return ax


def highlight_velocity_selection(ax: plt.Axes, params: DictConfig) -> plt.Axes:
    r_bounds = params.selection.range.r_bounds
    theta_bounds = params.selection.range.theta_bounds
    theta_range = np.linspace(theta_bounds[0], theta_bounds[1], 100)
    c = "r"
    colors = {
        "k": (0, 0, 0, 1),
        "r": (1, 0, 0, 1),
        "g": (0, 1, 0, 1),
        "b": (0, 0, 1, 1),
    }
    args = {
        "fc": (1, 1, 1, 0),
        "ec": colors[c],
        "zorder": 10,
        "lw": 1.5,
        "label": "$S$",
    }
    ax.fill_between(theta_range, r_bounds[0], r_bounds[1], **args)
    return ax
