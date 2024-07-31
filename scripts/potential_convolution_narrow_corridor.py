# %%
import logging
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.core.functions_to_select_grid_piece import (
    evaluate_selection_point,
    evaluate_selection_range,
    get_index_of_the_enclosing_bin,
)
from physped.io.readers import trajectory_reader
from physped.omegaconf_resolvers import register_new_resolvers
from physped.preprocessing.trajectories import preprocess_trajectories, process_slow_modes
from physped.visualization.plot_discrete_grid import plot_discrete_grid
from physped.visualization.plot_potential_at_slow_index import plot_potential_at_slow_index
from physped.visualization.plot_trajectories import plot_trajectories

plt.style.use(Path.cwd().parent / "physped/conf/science.mplstyle")

# %%

env_name = "single_paths"
with initialize(version_base=None, config_path="../physped/conf", job_name="test_app"):
    config = compose(
        config_name="config",
        return_hydra_config=True,
        overrides=[
            f"params={env_name}",
            "params.data_source=local",
            #    "params.grid.spatial_cell_size=0.1"
        ],
    )
    print(config)
register_new_resolvers(replace=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.info("MODELING PARAMETERS: \n%s", pformat(OmegaConf.to_container(config.params.model, resolve=True), depth=1))
logging.info("GRID PARAMETERS: \n%s", pformat(OmegaConf.to_container(config.params.grid, resolve=True), depth=2))


# %%

config = evaluate_selection_range(config)
config = evaluate_selection_point(config)
logging.info(pformat(dict(config.params.selection.range)))

# %%

trajectories = trajectory_reader[env_name](config)
preprocessed_trajectories = preprocess_trajectories(trajectories, config=config)
preprocessed_trajectories = process_slow_modes(preprocessed_trajectories, config)
piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, config)

# %%


plot_trajectories(preprocessed_trajectories, config, "recorded")
# plot_trajectories(simulated_trajectories, config, "simulated")

# %%
# config.params.selection.range.x = [55] * 2
# config.params.selection.range.y = [-5] * 2
# config.params.selection.range.r = [1.3] * 2

config.params.selection.range.theta = [np.pi * 0] * 2
# config.params.force_field_plot.scale = 20
config = evaluate_selection_range(config)
selection = config.params.selection.range
slow_indices = (
    selection.x_indices[0],
    selection.y_indices[0],
    selection.r_indices[0],
    selection.theta_indices[0],
    selection.k_indices[0],
)
plot_discrete_grid(config, slow_indices, preprocessed_trajectories)
plot_potential_at_slow_index(config, slow_indices, piecewise_potential)

# y_index = 3
# offset = piecewise_potential.position_based_offset[bin_index[0], y_index]
# print(offset)
# X_dashed = np.linspace(ybins[y_index] - dy / 2, ybins[y_index + 1] + dy / 2, 100)
# print(X_dashed)
# Vy_dashed = calculate_potential(
#     piecewise_potential.curvature_y[*bin_index], piecewise_potential.center_y[*bin_index], offset, X_dashed
# )
# print(Vy_dashed)
# print(piecewise_potential.curvature_y[:,:,1,0,1])
# %%


def make_line_variable_width(x, y, color, ax):
    middle_index = len(x) // 2
    middle_x = x[middle_index]
    d_to_center = np.abs(x - middle_x)
    width_in_center = 3
    # normalize the width
    d_to_center = (d_to_center / d_to_center.max()) * width_in_center
    line_widths = width_in_center - d_to_center

    ax.scatter(x, y, s=line_widths, color=color, alpha=line_widths / np.max(line_widths), zorder=20)
    return ax


def calculate_potential(curvature, center, offset, value):
    return curvature * (value - center) ** 2 + offset


# Analytical parabolic potential
yrange = np.arange(-0.6, 0.6, 0.01)
px_to_mm = {"x": 3.9, "y": 4.1}
beta = 1.1
pot0 = 0.045
# A = 0.3
y_cent = 0.01
parabolic_potential = beta * (yrange - y_cent) ** 2 + pot0

# Get index for a point on the grid
# point = [0.45, -10, 0.6, 0, 3]
point = [0, -10, 1.3, 0, 3]
bin_index = []
for dim, value in zip(config.params.grid.bins, point):
    bin_index.append(get_index_of_the_enclosing_bin(value, config.params.grid.bins[dim]))
# bin_index[3] = 0

cmap = ["C2", "C1", "C3", "C0"] * 100
fig, ax = plt.subplots(figsize=(3.54, 1.5))
lw = 2

ybins = config.params.grid.bins.y
dy = ybins[1] - ybins[0]
middle_bins = ybins + dy / 2
for y_index in range(len(ybins) - 1):
    bin_index[1] = y_index
    # xmu, xvar, ymu, yvar, umu, uvar, vmu, vvar = piecewise_potential.fit_params[*bin_index, :]
    # if np.sum(piecewise_potential.fit_params[*bin_index, :]) == 0:
    #     continue

    offset = piecewise_potential.position_based_offset[bin_index[0], y_index]
    length_outside_bin = dy
    X_dashed = np.linspace(ybins[y_index] - length_outside_bin, ybins[y_index + 1] + length_outside_bin, 100)
    Vy_dashed = calculate_potential(
        piecewise_potential.curvature_y[*bin_index], piecewise_potential.center_y[*bin_index], offset, X_dashed
    )
    color = cmap[y_index]

    Vy_mid = calculate_potential(
        piecewise_potential.curvature_y[*bin_index],
        piecewise_potential.center_y[*bin_index],
        offset,
        middle_bins[y_index],
    )
    # ax.plot(middle_bins[y_index], Vy_mid, color="w", marker="|", ms=3, zorder=20)
    ax = make_line_variable_width(X_dashed, Vy_dashed, color=color, ax=ax)
    # ax.add_collection(lc)
    # ax.plot(X_dashed, Vy_dashed, alpha=0.4, linestyle="dashed", color=color, lw=lw)

    # X_solid = np.linspace(ybins[y_index], ybins[y_index + 1], 100)
    # Vy_solid = calculate_potential(
    #     piecewise_potential.curvature_y[*bin_index], piecewise_potential.center_y[*bin_index], offset, X_solid
    # )
    # ax.plot(X_solid, Vy_solid, color=color, lw=lw)

ax.set_xlim(config.params.default_ylims)
ax.grid(False)
# ax.set_xticks(ybins[::2], minor = True)
ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4], minor=True)
ax.set_xticks([-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])
# ax.xaxis.set_minor_locator(ybins[::2])

y_walls = config.params.trajectory_plot.ywalls
# Plot grid
ax.vlines(ybins, 0, 1, lw=0.4, color="k", linestyle="dashed", alpha=0.6)
ax.hlines(np.linspace(0, 1, 6), y_walls[0], y_walls[1], lw=0.4, color="k", linestyle="dashed", alpha=0.6)

# Plot walls
ax.vlines(y_walls, 0, 2, "k")
for ywall in y_walls:
    if ywall < 0:
        fillbetweenx = [10 * ywall, ywall]
    elif ywall > 0:
        fillbetweenx = [ywall, 10 * ywall]
    ax.fill_between(
        fillbetweenx,
        2,
        0,
        color="k",
        alpha=0.3,
        zorder=30,
        hatch="//",
    )

plt.ylim(0, 0.6)
plt.ylabel("$U(y\\,|\\,\\Phi) + O(\\Phi)$")
plt.ylabel("$U(y\\,|\\vec{x}_s, \\vec{u}_s) + O(\\vec{x}_s, \\vec{u}_s)$")
plt.xlabel("y [m]")
plt.plot(
    yrange,
    parabolic_potential,
    "k--",
    lw=1,
    zorder=90,
    label="Analytic potential \n$U(y) = \\beta y^2$ (Eq.~(5))",
)
# plt.plot(
#     yrange,
#     parabolic_potential,
#     "k--",
#     lw=1.5,
#     zorder=20,
#     alpha=0.3,
#     # label="Analytic potential \n$V(y) = \\beta y^2$ (Eq.~(5))",
# )
plt.legend(loc="upper center")
plt.savefig("../figures/potential_convolution_narrow_corridor.pdf")

# %%

fig, ax = plt.subplots()
ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
coefficients = piecewise_potential.curvature_y[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4]]
ybottom, ytop = -0.2, 0.2
ax.plot(middle_bins[:-1], dymu, ".-", label="Mean $\\mu_y - y_s$")
# ax.plot(middle_bins[:-1], np.divide(coefficients, 10), ".-", label="Curvature $\\beta_y/10$")
ax.set_xlim(config.params.default_ylims)
ax.grid(False)
ax.set_xticks(ybins[::2])
# ax.set_xticks([-0.6, -0.3,-0.1, 0.1, 0.3, 0.6])

y_walls = config.params.trajectory_plot.ywalls
# Plot grid
ax.vlines(ybins, ybottom, ytop, lw=0.4, color="k", linestyle="dashed", alpha=0.6)
ax.hlines(np.linspace(ybottom, ytop, 6), y_walls[0], y_walls[1], lw=0.4, color="k", linestyle="dashed", alpha=0.6)

# Plot walls
ax.vlines(y_walls, ytop, ybottom, "k")
for ywall in y_walls:
    if ywall < 0:
        fillbetweenx = [10 * ywall, ywall]
    elif ywall > 0:
        fillbetweenx = [ywall, 10 * ywall]
    ax.fill_between(
        fillbetweenx,
        ytop,
        ybottom,
        color="k",
        alpha=0.3,
        zorder=30,
        hatch="//",
    )

plt.legend(loc="lower center")
plt.savefig("../figures/potential_mean_narrow_corridor.pdf")

# %%
fig, ax = plt.subplots()
ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
coefficients = piecewise_potential.curvature_y[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4]]
ybottom, ytop = 0, 15
# ax.plot(middle_bins[:-1], dymu, ".-", label="Mean $\\mu_y$")
ax.plot(middle_bins[:-1], coefficients, ".-", label="Curvature $\\beta_y$")
ax.set_xlim(config.params.default_ylims)
ax.grid(False)
ax.set_xticks(ybins[::2])

y_walls = config.params.trajectory_plot.ywalls
# Plot grid
ax.vlines(ybins, ybottom, ytop, lw=0.4, color="k", linestyle="dashed", alpha=0.6)
ax.hlines(np.linspace(ybottom, ytop, 6), y_walls[0], y_walls[1], lw=0.4, color="k", linestyle="dashed", alpha=0.6)

# Plot walls
ax.vlines(y_walls, ytop, ybottom, "k")
for ywall in y_walls:
    if ywall < 0:
        fillbetweenx = [10 * ywall, ywall]
    elif ywall > 0:
        fillbetweenx = [ywall, 10 * ywall]
    ax.fill_between(
        fillbetweenx,
        ytop,
        ybottom,
        color="k",
        alpha=0.3,
        zorder=30,
        hatch="//",
    )

plt.legend(loc="lower center")
plt.savefig("../figures/potential_curvature_narrow_corridor.pdf")

# %%
fig, ax = plt.subplots()
ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
# dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
coefficients = piecewise_potential.curvature_y[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4]]
middle_bins = ybins + dy / 2
middle_bins = (middle_bins + 0.6) / 1.2
ybottom, ytop = 0, 15
# ax.plot(middle_bins[:-1], dymu, ".-", label="Mean $\\mu_y$")
ax.plot(middle_bins[:-1], coefficients, ".-", label="Curvature $\\beta_y$")
ax.set_xlim(-0.1, 1.1)
ax.grid(False)
# ax.set_xticks(ybins[::2])

# y_walls = cfg.params.trajectory_plot.ywalls
y_walls = [0, 1]
# y_walls = (np.array(y_walls) + 0.6)/1.2
# # Plot grid
ax.vlines(np.arange(0, 1.1, 0.1), ybottom, ytop, lw=0.4, color="k", linestyle="dashed", alpha=0.6)
ax.hlines(np.linspace(ybottom, ytop, 6), y_walls[0], y_walls[1], lw=0.4, color="k", linestyle="dashed", alpha=0.6)

# Plot walls
ax.vlines(y_walls, ytop, ybottom, "k")
for ywall in y_walls:
    if ywall < 0:
        fillbetweenx = [10 * ywall, ywall]
    elif ywall > 0:
        fillbetweenx = [ywall, 10 * ywall]
    else:
        fillbetweenx = [-5, ywall]
    ax.fill_between(
        fillbetweenx,
        ytop,
        ybottom,
        color="k",
        alpha=0.3,
        zorder=30,
        hatch="//",
    )

plt.xlabel("$y/w_{cor}$")

plt.legend(loc="lower center")
plt.savefig("../figures/potential_curvature_narrow_corridor_relative.pdf")

# # %%

# fig, ax = plt.subplots()
# # bounds = bounds or infer_bounds_from_data(fields["offset"])
# # norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
# xbin_middle = cfg.params.grid.bins.x + (cfg.params.grid.bins.x[1] - cfg.params.grid.bins.x[0]) / 2
# ybin_middle = cfg.params.grid.bins.y + (cfg.params.grid.bins.y[1] - cfg.params.grid.bins.y[0]) / 2
# X, Y = np.meshgrid(xbin_middle, ybin_middle, indexing="ij")
# cmap = "YlOrRd"
# cs = ax.pcolormesh(X, Y, piecewise_potential.position_based_offset, cmap=cmap, shading="auto")  # , norm=norm)
# # ax = plot_colorbar(ax, cs)

# # %%
# from physped.core.functions_to_discretize_grid import (
#     calculate_position_based_emperic_potential,
#     get_slice_of_multidimensional_matrix,
# )

# slices = [cfg.params.selection.range[f"{dim}_indices"] for dim in piecewise_potential.dimensions]
# slices = [[x, y + 1] for x, y in slices]
# print(slices)
# sliced_histogram = get_slice_of_multidimensional_matrix(piecewise_potential.histogram_slow, slices)
# print(sliced_histogram.shape)
# position_based_emperic_potential = calculate_position_based_emperic_potential(sliced_histogram)

# # %%

# fig, ax = plt.subplots()
# xbin_middle = (cfg.params.grid.bins.x[1:] + cfg.params.grid.bins.x[:-1]) / 2
# ybin_middle = (cfg.params.grid.bins.y[1:] + cfg.params.grid.bins.y[:-1]) / 2
# X, Y = np.meshgrid(xbin_middle, ybin_middle, indexing="ij")
# cmap = "YlOrRd"
# cs = ax.pcolormesh(X, Y, position_based_emperic_potential, cmap=cmap, shading="auto")  # , norm=norm)
# # ax = plot_colorbar(ax, cs)

# # %%

# sliced_fit_parameters = get_slice_of_multidimensional_matrix(piecewise_potential.fit_params, slices)
# # sliced_curvature = get_slice_of_multidimensional_matrix()

# # %%
# from physped.utils.functions import weighted_mean_of_matrix

# curv_x = weighted_mean_of_matrix(piecewise_potential.curvature_x, sliced_histogram)
# curv_y = weighted_mean_of_matrix(piecewise_potential.curvature_y, sliced_histogram)
# curv_u = weighted_mean_of_matrix(piecewise_potential.curvature_u, sliced_histogram)
# curv_v = weighted_mean_of_matrix(piecewise_potential.curvature_v, sliced_histogram)

# # %%

# fig, ax = plt.subplots()
# cs = ax.pcolormesh(X, Y, curv_y, cmap=cmap, shading="auto")  # , norm=norm)
# # ax = plot_colorbar(ax, cs)

# # %%

# fig, ax = plt.subplots()
# q = ax.quiver(
#     X,
#     Y,
#     curv_x,
#     curv_y,
#     scale=1000,
#     pivot="mid",
#     width=0.005,
# )
# # ax.quiverkey(
# #     q,
# #     X=0.2,
# #     Y=1.1,
# #     U=50,
# #     label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
# #     labelpos="E",
# # )
# # ax = plot_quiverkey(ax, q)
# ax.set_aspect("equal")
# ax.set_xlim(cfg.params.default_xlims)
# ax.set_ylim(cfg.params.default_ylims)

# # %%

# # field = grids.selection[..., grids.fit_param_names.index(fit_param)]
# fields[fit_param] = weighted_mean_of_matrix(field, sliced_histogram)
# fields[fit_param] = np.where(pos_histogram > minimum_datapoints, fields[fit_param], np.nan)


# # %%

# plt.imshow(piecewise_potential.curvature_y[:, :, bin_index[2], bin_index[3], bin_index[4]])


# # %%


# bin_counts = piecewise_potential.histogram_slow
# position_counts = np.nansum(bin_counts, axis=(2, 3, 4))
# position_counts = np.where(position_counts == 0, np.nan, position_counts)

# position_based_offsets = A * (-np.log(position_counts) + np.log(np.nansum(position_counts)))
# offset = position_based_offsets[bin_index[0], y_index]

# print(position_based_offsets)
# # %%

# plt.imshow(np.nansum(piecewise_potential.histogram, axis=(2, 3, 4)))

# # %%

# np.round(piecewise_potential.fit_params[:, :, bin_index[2], bin_index[3], bin_index[4], 0], 2)

# # %%

# %%
