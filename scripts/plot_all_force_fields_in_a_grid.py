import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import physped as pp
from physped.core.functions_to_discretize_grid import make_grid_selection

from physped.visualization.plot_trajectories import (
    plot_station_background,
    apply_polar_plot_style,
    highlight_grid_box,
    create_grid_box_limits,
)

from physped.visualization.plot_fields import (
    plot_quiver_force_vectors,
    plot_pcolormesh_offset_field,
    clip_fields,
    calculate_offsets,
    create_force_fields,
)

log = logging.getLogger(__name__)


def main_theta_r(name: str):
    # name = "station_paths"
    params = pp.read_parameter_file(name)
    params["force_field_plot"] = {"clip": 0, "scale": 800, "sparseness": 3}
    filepath = pp.create_filepath(params)
    grids = pp.read_discrete_grid_from_file(filepath)

    idk = 1
    fig = plt.figure(figsize=(60, 16), layout="constrained")
    N_idr_vals = grids.fit_params.shape[2]
    N_idtheta_vals = grids.fit_params.shape[3]
    spec = gridspec.GridSpec(
        figure=fig,
        ncols=N_idtheta_vals * 2,
        nrows=N_idr_vals,
        width_ratios=[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        wspace=0.3,
        hspace=0.05,
    )
    X, Y = np.meshgrid(grids.bin_centers["x"], grids.bin_centers["y"], indexing="ij")

    for idr in range(N_idr_vals):
        for idtheta in range(N_idtheta_vals):
            if idr == 0 and idtheta > 0:
                continue
            fields = {}
            fields["X"] = X
            fields["Y"] = Y
            ## Force field
            hist = grids.histogram_slow[:, :, idr : idr + 1, idtheta : idtheta + 1, :]
            offsets = calculate_offsets(hist)
            fields["offset"] = offsets
            for param in ["xmu", "xvar", "ymu", "yvar"]:
                idparam = grids.fit_param_names.index(param)
                field = grids.fit_params[:, :, idr, idtheta, idk, idparam]
                fields[param] = field

            fields["dfx"] = -(fields["X"] - fields["xmu"]) / np.where(fields["xvar"] != 0, fields["xvar"], np.inf)
            fields["dfy"] = -(fields["Y"] - fields["ymu"]) / np.where(fields["yvar"] != 0, fields["yvar"], np.inf)

            plotid = (idr * 2 * N_idtheta_vals) + ((idtheta * 2))
            ax = fig.add_subplot(spec[plotid])
            field = grids.fit_params[:, :, idr, idtheta, idk, idparam]
            ax = plot_pcolormesh_offset_field(ax, fields)
            clipped_fields = clip_fields(fields, params["force_field_plot"]["clip"])
            ax = plot_quiver_force_vectors(
                ax,
                clipped_fields,
                params["force_field_plot"]["scale"],
                params["force_field_plot"]["sparseness"],
            )
            ax = plot_station_background(ax, params)

            ## Velocity selection
            selection = {
                "x": None,
                "y": None,
                "r": [grids.bins["r"][idr], grids.bins["r"][idr]],
                "theta": [grids.bins["theta"][idtheta], grids.bins["theta"][idtheta]],
                "k": [0, 2],
            }
            grid_selection = make_grid_selection(grids, selection)
            slices = [tuple(grid_selection[dim]["grid_ids"]) for dim in grids.dimensions]

            plotid += 1
            ax = fig.add_subplot(spec[plotid], polar=True)
            limits = create_grid_box_limits(slices, grids.dimensions, grids.bins, obs=["r", "theta"])
            if limits[0][0] < grids.bins["r"][1]:
                ax.plot(
                    np.linspace(-np.pi, np.pi, 10),
                    np.repeat(grids.bins["r"][1], 10),
                    "k-",
                    lw=2,
                )
            else:
                ax = highlight_grid_box(ax, limits)
            ax = apply_polar_plot_style(ax, params)

            rect = plt.Rectangle(
                ((idtheta * 1 / N_idtheta_vals), (N_idr_vals - 1 - idr) / N_idr_vals),
                1 / N_idtheta_vals,
                1 / N_idr_vals,
                fill=False,
                color="k",
                lw=2,
                zorder=1000,
                transform=fig.transFigure,
                figure=fig,
            )
            fig.patches.extend([rect])

    plt.savefig(f"figures/{name}_all_force_fields_in_a_grid.pdf")


def main_r_theta(name: str):
    # name = "station_paths"
    params = pp.read_parameter_file(name)
    # params["force_field_plot"] = {"clip": 0, "scale": 800, "sparseness": 3}
    filepath = pp.create_filepath(params)
    grids = pp.read_discrete_grid_from_file(filepath)

    idk = 1
    N_idr_vals = grids.fit_params.shape[2]
    N_idtheta_vals = grids.fit_params.shape[3]
    fig = plt.figure(figsize=(N_idr_vals * 2 * 8, N_idr_vals * 6), layout="constrained")

    width_ratio = [2, 1]
    spec = gridspec.GridSpec(
        figure=fig,
        ncols=N_idr_vals * 2,
        nrows=N_idtheta_vals,
        width_ratios=np.tile(width_ratio, N_idr_vals),
        wspace=0.3,
        hspace=0.05,
    )
    X, Y = np.meshgrid(grids.bin_centers["x"], grids.bin_centers["y"], indexing="ij")

    for idr in range(N_idr_vals):
        for idtheta in range(N_idtheta_vals):
            if idr == 0 and idtheta > 0:
                continue

            sliced_histogram = grids.histogram_slow[:, :, idr : idr + 1, idtheta : idtheta + 1, :]
            grids.selection = grids.fit_params[:, :, idr : idr + 1, idtheta : idtheta + 1, :, :]
            fields = create_force_fields(grids, sliced_histogram)

            plotid = (idtheta * 2 * N_idr_vals) + ((idr * 2))
            ax = fig.add_subplot(spec[plotid])
            ax = plot_pcolormesh_offset_field(ax, fields)
            clipped_fields = clip_fields(fields, params["force_field_plot"]["clip"])
            ax = plot_quiver_force_vectors(
                ax,
                clipped_fields,
                params["force_field_plot"]["scale"],
                params["force_field_plot"]["sparseness"],
            )
            ax = plot_station_background(ax, params)

            ## Velocity selection
            selection = {
                "x": None,
                "y": None,
                "r": [grids.bins["r"][idr], grids.bins["r"][idr]],
                "theta": [grids.bins["theta"][idtheta], grids.bins["theta"][idtheta]],
                "k": [0, 2],
            }
            grid_selection = make_grid_selection(grids, selection)
            slices = [tuple(grid_selection[dim]["grid_ids"]) for dim in grids.dimensions]
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            plotid += 1
            ax = fig.add_subplot(spec[plotid], polar=True)
            limits = create_grid_box_limits(slices, grids.dimensions, grids.bins, obs=["r", "theta"])
            if limits[0][0] < grids.bins["r"][1]:
                ax.plot(
                    np.linspace(-np.pi, np.pi, 10),
                    np.repeat(grids.bins["r"][1], 10),
                    "k-",
                    lw=2,
                )
            else:
                ax = highlight_grid_box(ax, limits)
            ax = apply_polar_plot_style(ax, params)

            rect = plt.Rectangle(
                (
                    (idr * 1 / N_idr_vals),
                    (N_idtheta_vals - 1 - idtheta) / N_idtheta_vals,
                ),
                1 / N_idr_vals,
                1 / N_idtheta_vals,
                fill=False,
                color="k",
                lw=2,
                zorder=1000,
                transform=fig.transFigure,
                figure=fig,
            )
            fig.patches.extend([rect])

    plt.savefig(f"figures/{name}_all_force_fields_in_a_grid.pdf")


if __name__ == "__main__":
    name = sys.argv[1]
    main_r_theta(name)
