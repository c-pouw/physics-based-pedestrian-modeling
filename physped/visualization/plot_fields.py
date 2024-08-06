from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from physped.core.parametrize_potential import extract_submatrix, make_grid_selection
from physped.utils.functions import weighted_mean_of_matrix
from physped.visualization.plot_trajectories import (  # create_grid_box_limits,
    apply_polar_plot_style,
    apply_xy_plot_style,
    highlight_grid_box,
    plot_station_background,
)


def plot_quiver_force_vectors(ax: plt.Axes, fields: dict, scale: int, sparseness: int) -> plt.Axes:
    """Plot the force field."""
    # scale = params["force_field_plot"]["scale"]
    # sparseness = params["force_field_plot"]["sparseness"]
    q = ax.quiver(
        fields["X"][::sparseness],
        fields["Y"][::sparseness],
        fields["dfx"][::sparseness],
        fields["dfy"][::sparseness],
        scale=scale,
        pivot="mid",
        width=0.005,
    )
    ax.quiverkey(
        q,
        X=0.2,
        Y=1.1,
        U=50,
        label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
        labelpos="E",
    )
    # ax = plot_quiverkey(ax, q)
    ax.set_aspect("equal")
    return ax


def plot_quiverkey(ax: plt.Axes, q: mpl.quiver.Quiver) -> plt.Axes:
    """Plot the quiver key."""
    ax.quiverkey(
        q,
        X=0.2,
        Y=1.1,
        U=50,
        label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
        labelpos="E",
    )
    return ax


def plot_pcolormesh_offset_field(ax: plt.Axes, fields: dict, cmap="YlOrRd", bounds: tuple = (0, 0.36)) -> plt.Axes:
    """Plot the offset field."""
    bounds = bounds or infer_bounds_from_data(fields["offset"])
    norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
    cs = ax.pcolormesh(fields["X"], fields["Y"], fields["offset"], cmap=cmap, shading="auto", norm=norm)
    ax = plot_colorbar(ax, cs)
    return ax


def plot_contourf_offset_field(ax: plt.Axes, fields: dict, cmap="YlOrRd", bounds: tuple = (0, 0.36)) -> plt.Axes:
    """Plot the offset field."""
    bounds = bounds or infer_bounds_from_data(fields["offset"])
    levels = np.linspace(bounds[0], bounds[1], 20)
    cs = ax.contourf(
        fields["X"],
        fields["Y"],
        fields["offset"],
        cmap=cmap,
        corner_mask=False,
        extend="both",
        alpha=0.7,
        levels=levels,
    )
    ax = plot_colorbar(ax, cs)
    return ax


def infer_bounds_from_data(data: np.ndarray) -> tuple:
    """Infer bounds from data."""
    return (np.nanmin(data), np.nanmax(data))


def plot_colorbar(ax: plt.Axes, cs: mpl.contour.QuadContourSet, label: str) -> plt.Axes:
    """Plot the colorbar."""
    cbar = plt.colorbar(
        cs,
        ax=ax,
        shrink=0.5,
    )
    # label = "$\\Delta V = 0.023\\log{\\left[\\mathbb{P}" "(y_s\\mid x_s,\\vec u_s)\\right]}$"
    cbar.set_label(label)
    return ax


def calculate_offsets(hist, axis: Tuple = (2, 3, 4)):
    """Calculate offsets."""
    pos_hist = np.nansum(hist, axis=axis)
    pos_hist = np.where(pos_hist == 0, np.nan, pos_hist)
    offset_parameter = 0.023
    offset = offset_parameter * (-np.log(pos_hist) + np.log(np.nansum(pos_hist)))
    return offset


def clip_fields(fields, clip):
    """Clip fields."""
    # clip = params["force_field_plot"]["clip"]
    if clip == 0:
        clip = np.inf

    clip_min = -clip
    clip_max = clip
    par_fields = list(fields.keys())
    for field_name in par_fields:
        field = fields[field_name].copy()
        field[field == -np.inf] = np.nan
        field[field == np.inf] = np.nan
        fields[field_name] = np.clip(field, clip_min, clip_max)
    return fields


def create_force_fields(grids: dict, sliced_histogram: np.ndarray) -> dict:
    # TODO: Move this to one of the core files
    fields = {}
    pos_histogram = np.nansum(sliced_histogram, axis=(2, 3, 4))
    X, Y = np.meshgrid(grids.bin_centers["x"], grids.bin_centers["y"], indexing="ij")
    fields["X"] = X
    fields["Y"] = Y
    fields["offset"] = calculate_offsets(sliced_histogram)

    minimum_datapoints = np.nanmean(pos_histogram) * 0.5
    for fit_param in grids.fit_param_names:
        field = grids.selection[..., grids.fit_param_names.index(fit_param)]
        fields[fit_param] = weighted_mean_of_matrix(field, sliced_histogram)
        fields[fit_param] = np.where(pos_histogram > minimum_datapoints, fields[fit_param], np.nan)

    fields["dfx"] = -(fields["X"] - fields["xmu"]) / np.where(fields["xvar"] != 0, fields["xvar"], np.inf)
    fields["dfy"] = -(fields["Y"] - fields["ymu"]) / np.where(fields["yvar"] != 0, fields["yvar"], np.inf)
    return fields


def plot_force_field_of_selection(grids, params, selection):
    # Create parameter selection
    grid_selection = make_grid_selection(grids, selection)
    slices = [tuple(grid_selection[dim]["grid_ids"]) for dim in grids.dimensions]

    sliced_histogram = extract_submatrix(grids.histogram_slow, slices)

    grids.selection = extract_submatrix(grids.parametrization, slices)

    # Create force fields
    fields = create_force_fields(grids, sliced_histogram)

    # Plot force fields
    force_field_params = params.get("force_field_plot", {})
    width_ratios = force_field_params.get("width_ratios", [4, 1])
    # figsize = force_field_params.get(
    #     "figsize", (20, 8)
    # )  # TODO change to figsize of mplstyle
    # width_ratios = [4, 1]
    fig = plt.figure(layout="constrained")
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=width_ratios, wspace=0.1, hspace=0.1, figure=fig)
    plotid = 0
    ax = fig.add_subplot(spec[plotid])

    ax = plot_pcolormesh_offset_field(ax, fields)
    ax = apply_xy_plot_style(ax, params)
    clipped_fields = clip_fields(fields, params["force_field_plot"]["clip"])
    ax = plot_quiver_force_vectors(
        ax,
        clipped_fields,
        params["force_field_plot"]["scale"],
        params["force_field_plot"]["sparseness"],
    )

    if params.get("name") == "station_paths":
        ax = plot_station_background(ax, params)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")

    plotid += 1
    ax = fig.add_subplot(spec[plotid], polar=True)
    # limits = create_grid_box_limits(
    #     slices, grids.dimensions, grids.bins, obs=["r", "theta"]
    # )
    selection_limits = [grid_selection[obs]["periodic_bounds"] for obs in ["r", "theta"]]
    ax = highlight_grid_box(ax, selection_limits)
    ax = apply_polar_plot_style(ax, params)
    # xlims = selection["r"]
    # ylims = np.linspace(selection["theta"][0], selection["theta"][1], 100)
    # print(xlims, np.divide(np.array(selection["theta"]), np.pi))
    # ax.fill_between(ylims, xlims[0], xlims[1], color="r", alpha=0.2)
    return ax
