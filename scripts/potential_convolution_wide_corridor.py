# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize

from physped.core.functions_to_discretize_grid import learn_potential_from_trajectories
from physped.core.functions_to_select_grid_piece import get_index_of_the_enclosing_bin
from physped.io.readers import trajectory_reader
from physped.omegaconf_resolvers import register_new_resolvers
from physped.preprocessing.trajectories import preprocess_trajectories, process_slow_modes

plt.style.use(Path.cwd() / "../physped/conf/science.mplstyle")

# %%

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

env_name = "parallel_paths"
with initialize(version_base=None, config_path="../physped/conf", job_name="test_app"):
    cfg = compose(
        config_name="config",
        return_hydra_config=True,
        overrides=[
            f"params={env_name}",
            "hydra.verbose=True",
            "read.preprocessed_trajectories=False",
            # "params.grid.spatial_cell_size=0.4"
        ],
    )
    print(cfg)
register_new_resolvers()

# %%

trajectories = trajectory_reader[env_name](cfg)
preprocessed_trajectories = preprocess_trajectories(trajectories, config=cfg)
preprocessed_trajectories = process_slow_modes(preprocessed_trajectories, cfg)
piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, cfg)

# %%


def calculate_potential(curvature, center, offset, value):
    return curvature * (value - center) ** 2 + offset


# Get index for a point on the grid
point = [-0.5, -10, 0.6, 0, 3]
point = [0, -10, 0.6, 0, 3]
bin_index = []
for dim, value in zip(cfg.params.grid.bins, point):
    bin_index.append(get_index_of_the_enclosing_bin(value, cfg.params.grid.bins[dim]))
# bin_index[3] = 3

fig, ax = plt.subplots(figsize=(3.54, 1.5))

cmap = ["C0", "C1", "C2", "C3"] * 100

ybins = cfg.params.grid.bins.y
dy = ybins[1] - ybins[0]
middle_bins = ybins + dy / 2
lw = 1

for y_index in range(len(ybins) - 1)[::1]:
    bin_index[1] = y_index
    xmu, xvar, ymu, yvar, umu, uvar, vmu, vvar = piecewise_potential.fit_params[
        bin_index[0], bin_index[1], bin_index[2], bin_index[3], bin_index[4], :
    ]

    bin_counts = piecewise_potential.histogram_slow
    position_counts = np.nansum(bin_counts, axis=(2, 3, 4))
    position_counts = np.where(position_counts == 0, np.nan, position_counts)
    position_based_offsets = 0.023 * (-np.log(position_counts) + np.log(np.nansum(position_counts)))
    offset = position_based_offsets[bin_index[0], y_index]

    color = cmap[y_index]

    X_dashed = np.linspace(ybins[y_index] - dy / 2, ybins[y_index + 1] + dy / 2, 100)
    Vy_dashed = calculate_potential(
        piecewise_potential.curvature_y[*bin_index], piecewise_potential.center_y[*bin_index], offset, X_dashed
    )

    Vy_mid = calculate_potential(
        piecewise_potential.curvature_y[*bin_index],
        piecewise_potential.center_y[*bin_index],
        offset,
        middle_bins[y_index],
    )
    ax.plot(middle_bins[y_index], Vy_mid, color="w", marker="|", ms=3, zorder=20)
    # ax.plot(X_dashed, Vy_dashed, alpha=0.4, linestyle="dashed", color=color, lw=lw)

    X_solid = np.linspace(ybins[y_index], ybins[y_index + 1], 100)
    Vy_solid = calculate_potential(
        piecewise_potential.curvature_y[*bin_index], piecewise_potential.center_y[*bin_index], offset, X_solid
    )
    ax.plot(X_solid, Vy_solid, color=color, lw=lw)

ax.grid(False)
ax.set_xlim(cfg.params.default_ylims)
ax.set_ylim(0, 1)

y_walls = cfg.params.trajectory_plot.ywalls
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

plt.ylabel("$U(y \\mid \\vec{x}_s,\\vec{u}_s) + O(\\vec{x}_s,\\vec{u}_s)$")  # A\ln[\\mathbb{P(\\cdot)}]$')
plt.xlabel("y [m]")
plt.savefig("../figures/potential_convolution_wide_corridor.pdf")

# %%

# fig, ax = plt.subplots(figsize=(3.54, 1.5))
# ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
# dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
# # coefficients = np.divide(coefficients,10)
# ybottom, ytop = -0.5, 2
# ax.plot(middle_bins[:-1], dymu, ".-", label="Mean $\\mu_y$")
# ax.plot(middle_bins[:-1], coefficients, ".-", label="Curvature $\\beta_y$")
# ax.set_xlim(cfg.params.default_ylims)
# ax.grid(False)
# # ax.set_xticks(ybins)

# y_walls = cfg.params.trajectory_plot.ywalls
# # Plot grid
# ax.vlines(ybins, ybottom, ytop, lw=0.4, color="k", linestyle="dashed", alpha=0.6)
# ax.hlines(np.linspace(ybottom, ytop, 6), y_walls[0], y_walls[1], lw=0.4, color="k", linestyle="dashed", alpha=0.6)

# # Plot walls
# ax.vlines(y_walls, ytop, ybottom, "k")
# for ywall in y_walls:
#     if ywall < 0:
#         fillbetweenx = [10 * ywall, ywall]
#     elif ywall > 0:
#         fillbetweenx = [ywall, 10 * ywall]
#     ax.fill_between(
#         fillbetweenx,
#         ytop,
#         ybottom,
#         color="k",
#         alpha=0.3,
#         zorder=30,
#         hatch="//",
#     )

# plt.legend(loc="upper center")
# plt.xlabel("y [m]")
# plt.ylim(ybottom, ytop)
# plt.savefig("../figures/potential_parameters_wide_corridor.pdf")

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
ax.set_xlim(cfg.params.default_ylims)
ax.grid(False)
ax.set_xticks(ybins[::2])

y_walls = cfg.params.trajectory_plot.ywalls
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
plt.savefig("../figures/potential_mean_wide_corridor.pdf")

# %%

fig, ax = plt.subplots()
ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
coefficients = piecewise_potential.curvature_y[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4]]
ybottom, ytop = 0, 18
# ax.plot(middle_bins[:-1], dymu, ".-", label="Mean $\\mu_y$")
ax.plot(middle_bins[:-1], coefficients, ".-", label="Curvature $\\beta_y$")
ax.set_xlim(cfg.params.default_ylims)
ax.grid(False)
ax.set_xticks(ybins[::2])

y_walls = cfg.params.trajectory_plot.ywalls
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
plt.savefig("../figures/potential_curvature_wide_corridor.pdf")

# %%
fig, ax = plt.subplots()
ymu = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 2]
# dymu = np.where(ymu == 0, np.nan, ymu - middle_bins[:-1])
# yvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 3]
# vvar = piecewise_potential.fit_params[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4], 7]
# coefficients = vvar / (2 * yvar)
coefficients = piecewise_potential.curvature_y[bin_index[0], :, bin_index[2], bin_index[3], bin_index[4]]
middle_bins = ybins + dy / 2
middle_bins = (middle_bins + 3.8) / 7.6
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
plt.savefig("../figures/potential_curvature_wide_corridor_relative.pdf")


# %%

fig, ax = plt.subplots()
# bounds = bounds or infer_bounds_from_data(fields["offset"])
# norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
xbin_middle = cfg.params.grid.bins.x + (cfg.params.grid.bins.x[1] - cfg.params.grid.bins.x[0]) / 2
ybin_middle = cfg.params.grid.bins.y + (cfg.params.grid.bins.y[1] - cfg.params.grid.bins.y[0]) / 2
X, Y = np.meshgrid(xbin_middle, ybin_middle, indexing="ij")
cmap = "YlOrRd"
cs = ax.pcolormesh(X, Y, piecewise_potential.position_based_offset, cmap=cmap, shading="auto")  # , norm=norm)

# %%
