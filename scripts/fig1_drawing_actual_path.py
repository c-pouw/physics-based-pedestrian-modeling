# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize
from matplotlib.patches import ConnectionPatch, Ellipse

from physped.io.readers import trajectory_reader
from physped.omegaconf_resolvers import register_new_resolvers
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.visualization.plot_utils import plot_station_background

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log = logging.getLogger(__name__)

plt.style.use(Path.cwd() / "../conf/science.mplstyle")

# %%

env_name = "station_paths"
with initialize(version_base=None, config_path="../conf", job_name="test_app"):
    cfg = compose(config_name="config", return_hydra_config=True, overrides=[f"params={env_name}"])
register_new_resolvers()

# %%

trajectories = trajectory_reader[env_name](cfg)
preprocessed_trajectories = preprocess_trajectories(trajectories, config=cfg)
# piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, cfg)

# %%

pd.set_option("display.max_columns", None)
preprocessed_trajectories.head()
# %%


def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x ** np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[: n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[: n + 1, : n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[: n + 1, n + 1 :] = xf_n / 2
    mat[n + 1 :, : n + 1] = xf_n.T
    mat[n + 1 :, n + 1 :] = 0
    vec[: n + 1] = yx_n
    vec[n + 1 :] = yf
    params = np.linalg.solve(mat, vec)
    return params[: n + 1]


def plot_frame(traj_to_plot):
    # xvals = traj_to_plot["xf"].to_list()
    # yvals = traj_to_plot["yf"].to_list()
    # xf = [xvals[0], xvals[-1]]
    # yf = [yvals[0], yvals[-1]]

    direct_path_x = [traj_to_plot["xf"].iloc[0], traj_to_plot["xf"].iloc[-1]]
    direct_path_y = [traj_to_plot["yf"].iloc[0], traj_to_plot["yf"].iloc[-1]]

    path1_color = "k"
    path2_color = "C1"
    path3_color = "C2"

    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    # fig, ax = plt.subplots()

    intended_path_x = traj_to_plot["xs"].tolist() + [direct_path_x[-1]]
    intended_path_y = traj_to_plot["ys"].tolist() + [direct_path_y[-1]]
    (path1,) = ax.plot(traj_to_plot["xf"], traj_to_plot["yf"], ".", c=path1_color, alpha=1, ms=1)
    (path2,) = ax.plot(intended_path_x, intended_path_y, ls="--", c=path2_color, alpha=1, lw=1)
    (path3,) = ax.plot(direct_path_x, direct_path_y, "-", c=path3_color, zorder=1)
    (path3,) = ax.plot(direct_path_x, direct_path_y, "--", c=path3_color, alpha=0.8, lw=1)

    ax.set_xlabel("$x\\; [\\mathrm{m}]$")
    ax.set_ylabel("$y\\; [\\mathrm{m}]$")
    ax.set_aspect("equal")
    if cfg.params.env_name == "station_paths":
        ax = plot_station_background(ax, cfg)

    ax.text(
        x=61.7,
        y=-3.1,
        s="Bench",
        ha="center",
        va="center",
        # fontsize = 6,
        # family='monospace',
        weight="bold",
    )
    bench_polygon = [[58.7, -2.4], [64.5, -2.4], [64.5, -3.6], [58.7, -3.6]]
    bench = plt.Polygon(bench_polygon, edgecolor="r", facecolor="white", lw=1)
    ax.add_patch(bench)

    # x1, x2, y1, y2 = 0, 0.2, 0, 0.2  # subregion of origanal image
    # inset_x = [x1, x2]
    # inset_y = [y1, y2]

    center1 = traj_to_plot.iloc[0][["xf", "yf"]].to_list()
    # ax.text(center1[0], center1[1], 'Origin', ha = 'center', )
    ax.text(
        center1[0] - 1.2,
        center1[1] + 1.5,
        "Origin",
        ha="center",
        bbox=dict(facecolor="white", edgecolor=None, boxstyle="round,pad=0.5"),
    )
    center2 = traj_to_plot.iloc[-1][["xf", "yf"]].to_list()
    ax.text(
        center2[0] + 1.2,
        center2[1] - 2,
        "Destination",
        ha="center",
        bbox=dict(facecolor="white", edgecolor=None, boxstyle="round,pad=0.5"),
    )
    for center in [center1, center2]:
        lw = 0.2
        circle = plt.Circle(
            (0 + center[0], 0.06 + center[1]),
            radius=0.2,
            color="white",
            ec="k",
            lw=lw,
            zorder=10,
        )
        ax.add_patch(circle)
        ellipse = Ellipse(
            xy=(0 + center[0], 0.05 + center[1]),
            width=0.4,
            height=0.8,
            edgecolor="k",
            fc="lightgray",
            lw=lw,
            zorder=5,
        )
        ax.add_patch(ellipse)

    inset = True
    if inset:
        x1 = 52
        x2 = x1 + 4
        y1 = -6.5
        y2 = y1 + 3
        # axin = ax.inset_axes([0.5, 0.05, 0.3, 0.35], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        axin = ax.inset_axes([0.05, 0.57, 0.3, 0.4], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

        (path1,) = axin.plot(traj_to_plot["xf"], traj_to_plot["yf"], ".", c=path1_color, alpha=1, ms=1)
        (path2,) = axin.plot(traj_to_plot["xs"], traj_to_plot["ys"], c=path2_color, ls="--", alpha=1, lw=1)
        (path3,) = axin.plot(direct_path_x, direct_path_y, "-", c=path3_color)

        rect = (x1, y1, x2 - x1, y2 - y1)

        inset_zoom_color = "white"
        ax.indicate_inset(rect, edgecolor=inset_zoom_color, alpha=1, lw=1, ls="--")
        cp1 = ConnectionPatch(
            xyA=(x1, y1),
            xyB=(0, 0),
            axesA=ax,
            axesB=axin,
            coordsA="data",
            coordsB="axes fraction",
            lw=1,
            ls=":",
            ec=inset_zoom_color,
            zorder=20,
        )
        cp2 = ConnectionPatch(
            xyA=(x2, y2),
            xyB=(1, 1),
            axesA=ax,
            axesB=axin,
            coordsA="data",
            coordsB="axes fraction",
            lw=1,
            ls=":",
            ec=inset_zoom_color,
            zorder=20,
        )

        ax.add_patch(cp1)
        ax.add_patch(cp2)

    ax.set_xlim(cfg.params.trajectory_plot.xlims)
    ax.set_ylim(-12, 5)
    # ax.set_ylim(cfg.params.trajectory_plot.ylims)

    lines = [path3, path2, path1]
    # labels = ["Direct path to destination", "Intended path perturbed by obstacles", "Actual path including noise"]
    # plt.figlegend(lines, labels, loc="upper left", bbox_to_anchor=(0.11, 1.07), ncol=1)
    labels = ["Direct path to destination", "Intended path around obstacles", "Actual path with fluctuations"]
    plt.figlegend(lines, labels, loc="upper left", bbox_to_anchor=(0.4, 0.39), ncol=1, fontsize=7.5)
    plt.savefig("../figures/fig1_drawing_intended_path.pdf", bbox_inches="tight")


number = 1983075
traj_to_plot = preprocessed_trajectories[preprocessed_trajectories.Pid == preprocessed_trajectories.iloc[number].Pid].copy()
plot_frame(traj_to_plot)

# %%
