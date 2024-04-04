import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import get_original_cwd

from physped.core.piecewise_potential_handling import get_the_boundaries_that_enclose_the_selected_values
from physped.visualization.plot_trajectories import apply_polar_plot_style, apply_xy_plot_style

log = logging.getLogger(__name__)


def plot_cartesian_spatial_grid(ax: plt.Axes, grid_params: dict) -> plt.Axes:
    xbins = np.arange(grid_params.x.min, grid_params.x.max, grid_params.x.step)
    ybins = np.arange(grid_params.y.min, grid_params.y.max, grid_params.y.step)
    linestyle = "dashed"
    alpha = 0.8
    color = "k"
    linewidth = 0.6
    for x in xbins:
        ax.axvline(
            x,
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            alpha=alpha,
        )
    for y in ybins:
        ax.axhline(
            y,
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            alpha=alpha,
        )
    return ax


def plot_polar_velocity_grid(ax: plt.Axes, grid_params: dict) -> plt.Axes:
    rbins = np.arange(grid_params.r.min, grid_params.r.max, grid_params.r.step)
    thetabins = np.linspace(-np.pi, np.pi + 0.01, grid_params.theta.chunks + 1)
    linestyle = "dashed"
    alpha = 0.8
    color = "k"
    linewidth = 0.6
    for r in rbins:
        ax.plot(
            np.linspace(0, 2 * np.pi, 100),
            np.ones(100) * r,
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            alpha=alpha,
        )
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


def plot_polar_labels(ax: plt.Axes, grid_params: dict) -> plt.Axes:
    rbins = np.arange(grid_params.r.min, grid_params.r.max, grid_params.r.step)
    thetabins = np.linspace(-np.pi, np.pi + 0.01, grid_params.theta.chunks + 1)
    for r in rbins[1:]:
        ax.text(
            np.pi / 2,
            r + 0.15,
            f"{r:.1f}",
            ha="center",
            va="center",
            # bbox = dict(
            #     facecolor='white', alpha=0.5,
            #     edgecolor='none', boxstyle='round')
        )
    for theta in thetabins[:-1]:
        ax.text(theta, rbins[-1] * 1.35, f"{theta/np.pi:.1f}$\\pi$", ha="center", va="center")
    return ax


def highlight_position_selection(ax: plt.Axes, params: dict) -> plt.Axes:
    xbins = np.arange(params.grid.x.min, params.grid.x.max, params.grid.x.step)
    x_bounds = get_the_boundaries_that_enclose_the_selected_values(params.selection.x, xbins)

    ybins = np.arange(params.grid.y.min, params.grid.y.max, params.grid.y.step)
    y_bounds = get_the_boundaries_that_enclose_the_selected_values(params.selection.y, ybins)
    xrange = np.linspace(x_bounds[0], x_bounds[1], 100)
    c = "r"
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
    ax.fill_between(xrange, y_bounds[0], y_bounds[1], **args)
    return ax


def highlight_velocity_selection(ax: plt.Axes, params: dict) -> plt.Axes:
    rbins = np.arange(params.grid.r.min, params.grid.r.max, params.grid.r.step)
    thetabins = np.linspace(-np.pi, np.pi + 0.01, params.grid.theta.chunks + 1)
    r_bounds = get_the_boundaries_that_enclose_the_selected_values(params.selection.r, rbins)
    theta_bounds = get_the_boundaries_that_enclose_the_selected_values(params.selection.theta, thetabins)
    theta_range = np.linspace(theta_bounds[0], theta_bounds[1], 100)
    c = "r"
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
    ax.fill_between(theta_range, r_bounds[0], r_bounds[1], **args)
    return ax


def plot_discrete_grid(config: dict):
    params = config.params
    fig = plt.figure(layout="constrained")
    width_ratios = [2, 1]
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=width_ratios, wspace=0.1, hspace=0.1, figure=fig)

    ax = fig.add_subplot(spec[0])
    ax = apply_xy_plot_style(ax, params)
    ax = plot_cartesian_spatial_grid(ax, params.grid)
    ax.set_xlabel("$x\\; [\\mathrm{m}]$")
    ax.set_ylabel("$y\\; [\\mathrm{m}]$")
    ax.set_xlim(params.grid.x.min, params.grid.x.max)
    ax.set_ylim(params.grid.y.min, params.grid.y.max)
    ax.set_aspect("equal")

    if params.grid_plot.highlight_selection:
        ax = highlight_position_selection(ax, params)
    ax.set_title("Spatial grid", y=1.1)

    ax = fig.add_subplot(spec[1], polar=True)
    ax = apply_polar_plot_style(ax, params)
    ax = plot_polar_velocity_grid(ax, params.grid)
    ax = plot_polar_labels(ax, params.grid)
    ax.set_ylim(params.grid.r.min, params.grid.r.max - params.grid.r.step)
    if params.grid_plot.highlight_selection:
        ax = highlight_velocity_selection(ax, params)
    ax.set_title("Velocity grid", y=1.1)
    filepath = Path.cwd() / params.grid.name
    plt.savefig(filepath, bbox_inches="tight")
    log.info("Saving trajectory plot to %s.", filepath.relative_to(get_original_cwd()))
