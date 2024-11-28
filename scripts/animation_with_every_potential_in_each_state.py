# %%
# %load_ext autoreload
# %autoreload 2
# %%

import logging
from pathlib import Path
from pprint import pformat

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm

from physped.core.parametrize_potential import (
    calculate_position_based_emperic_potential,
    extract_submatrix,
    get_grid_indices,
    learn_potential_from_trajectories,
)
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.core.trajectory_simulator import simulate_trajectories
from physped.io.readers import trajectory_reader
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.utils.config_utils import register_new_resolvers
from physped.visualization.plot_trajectories import plot_station_background, plot_trajectories
from physped.visualization.plot_utils import apply_xy_plot_style

plt.style.use(Path.cwd().parent / "physped/conf/science.mplstyle")

# %%

env_name = "narrow_corridor"
# env_name = "wide_corridor"
# env_name = "curved_paths_synthetic"
# env_name = "station_paths"
# env_name = "asdz_pf34"
with initialize(version_base=None, config_path="../physped/conf", job_name="test_app"):
    config = compose(
        config_name="config",
        return_hydra_config=True,
        overrides=[
            f"params={env_name}",
            # "params.model.sigma=0.1",
            # "params.model.tauu=0.5",
            # # "params.simulation.step=0.01",
            # # "params.data_source=local",
            # "params.simulation.sample_state=-1",
            # "params.grid.x.min=-2.7",
            # "params.grid.y.min=-1.8",
            # "params.grid.theta.min_multiple_pi=-1.125",
            # "params.grid.theta.segments=8",
            # "params.grid.r.list=[0, 0.5, 1.0, 1.5, 2, 2.5]",
            # "params.grid.spatial_cell_size=0.2"
        ],
    )
    print(config)

register_new_resolvers(replace=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%

logging.info("MODELING PARAMETERS: \n%s", pformat(OmegaConf.to_container(config.params.model, resolve=True), depth=1))
logging.info("GRID PARAMETERS: \n%s", pformat(OmegaConf.to_container(config.params.grid, resolve=True), depth=2))

# %%

trajectories = trajectory_reader[env_name](config)
trajectories.head()

# %%

preprocessed_trajectories = preprocess_trajectories(trajectories, config=config)
preprocessed_trajectories = compute_slow_dynamics(preprocessed_trajectories, config=config)

# %%
# preprocessed_trajectories = preprocess_trajectories(trajectories, config=config)
# preprocessed_trajectories = process_slow_modes(preprocessed_trajectories, config)

# %%

piecewise_potential = learn_potential_from_trajectories(preprocessed_trajectories, config)

# %%

# config.params.simulation.sample_state = 0
config.params.simulation.ntrajs = 1
config.params.input_ntrajs = len(preprocessed_trajectories.Pid.unique())
simulated_trajectories = simulate_trajectories(piecewise_potential, config, preprocessed_trajectories)

# %%

# plot_trajectories(preprocessed_trajectories, config, "recorded")
# plot_trajectories(preprocessed_trajectories, config, "recorded", traj_type="s")
plot_trajectories(simulated_trajectories, config, "simulated")
plot_trajectories(simulated_trajectories, config, "simulated", traj_type="s")

# %%

traj_pid = simulated_trajectories[simulated_trajectories["Pid"] == 0].copy()
traj_pid.dropna(subset=["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"], inplace=True, how="all")
last_state = traj_pid.iloc[-1]
plot_trajectories(traj_pid, config, "simulated", traj_type="f")

# %%

print(len(traj_pid))
# plt.hist(traj_pid["vf"], bins=100)

plt.hist(traj_pid["rf"], bins=100)
plt.xlim(0, 2)

# %%

traj_pid["c"] = traj_pid.apply(lambda x: f"C{x.piece_id}", axis=1)

for piece_id, traj in traj_pid.groupby("piece_id"):
    plt.plot(traj["xf"], traj["yf"], color=f"C{piece_id}", marker=".")

# %%


def get_curvature_point(config, state):
    xf, yf, uf, vf, xs, ys, us, vs, rs, thetas = state[["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs", "rs", "thetas"]]
    k = 2
    X_vals = [xs, ys, rs, thetas, k]
    slow_state_index = get_grid_indices(piecewise_potential, X_vals)

    xmean, ymean, umean, vmean = piecewise_potential.parametrization[
        slow_state_index[0], slow_state_index[1], slow_state_index[2], slow_state_index[3], slow_state_index[4], :, 0
    ]
    beta_x, beta_y, beta_u, beta_v = piecewise_potential.parametrization[
        slow_state_index[0], slow_state_index[1], slow_state_index[2], slow_state_index[3], slow_state_index[4], :, 1
    ]

    # xmean, xvar, ymean, yvar, umean, uvar, vmean, vvar = piecewise_potential.fit_params[
    #     X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4], :
    # ]

    # # determine potential energy contributions
    # betax = piecewise_potential.curvature_x[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]]
    # betay = piecewise_potential.curvature_y[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]]
    # betau = piecewise_potential.curvature_u[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]]
    # betav = piecewise_potential.curvature_v[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]]

    V_x = beta_x * (xf - xmean)
    V_y = beta_y * (yf - ymean)
    V_u = beta_u * (uf - umean)
    V_v = beta_v * (vf - vmean)

    return [xmean, ymean, umean, vmean, beta_x, beta_y, beta_u, beta_v, -V_x, -V_y, -V_u, -V_v, -V_x - V_u, -V_y - V_v]


force = []
for k in tqdm(traj_pid["k"][::1]):
    state = traj_pid[traj_pid["k"] == k].iloc[0]
    force.append(get_curvature_point(config, state))

traj_pid.loc[:, ["xmean", "ymean", "umean", "vmean", "betax", "betay", "betau", "betav", "Vx", "Vy", "Vu", "Vv", "fx", "fy"]] = (
    force
)

# %%


def plot_potential(ax, state, potential_type):
    color_coding = {
        "x": "C0",
        "y": "C1",
        "u": "C2",
        "v": "C3",
    }
    beta_type = f"beta{potential_type}"
    mean_type = f"{potential_type}mean"
    position_type = f"{potential_type}s"
    beta = state[beta_type]
    mean = state[mean_type]
    pot_range = np.arange(mean - 1, mean + 1, 0.1)
    potential = beta * (pot_range - mean) ** 2
    ax.plot(pot_range, potential, c=color_coding[potential_type])
    pot_position = beta * (state[position_type] - mean) ** 2
    ax.plot(state[position_type], pot_position, "kx")
    return ax


traj_pid["k"] = traj_pid["k"].astype(int)

width_single_panel = 1.7
height_single_panel = 1.2
subplot_grid = [3, 4]
fig = plt.figure(figsize=(width_single_panel * subplot_grid[1], height_single_panel * subplot_grid[0]))

ymax = 2
spec = mpl.gridspec.GridSpec(ncols=4, nrows=3, wspace=0.5, hspace=0.5, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[0, 2])
ax4 = fig.add_subplot(spec[0, 3])
ax5 = fig.add_subplot(spec[1:, :2])
ax6 = fig.add_subplot(spec[1:, 2:])
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

cmap = "YlOrRd"
xbin_middle = (config.params.grid.bins.x[1:] + config.params.grid.bins.x[:-1]) / 2
ybin_middle = (config.params.grid.bins.y[1:] + config.params.grid.bins.y[:-1]) / 2
X, Y = np.meshgrid(xbin_middle, ybin_middle, indexing="ij")

n_frames = 50
plot_traj = traj_pid[traj_pid["k"] > (traj_pid["k"].max() - n_frames)].copy()
for _, state in tqdm(plot_traj.iterrows(), total=len(plot_traj), desc="Plotting potentials", ascii=True):
    timestep = state["k"]

    ax = axs[0]
    pot_type = "x"
    ax = plot_potential(ax, state, pot_type)
    ax.set_xlabel(pot_type)
    ax.set_xlim(config.params.grid.bins.x[0], config.params.grid.bins.x[-1])
    ax.set_ylim(0, ymax)

    ax = axs[1]
    pot_type = "y"
    ax = plot_potential(ax, state, pot_type)
    ax.set_xlabel(pot_type)
    ax.set_xlim(config.params.grid.bins.y[0], config.params.grid.bins.y[-1])
    ax.set_ylim(0, ymax)

    ax = axs[2]
    pot_type = "u"
    ax = plot_potential(ax, state, pot_type)
    ax.set_xlabel(pot_type)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, ymax)

    ax = axs[3]
    pot_type = "v"
    ax = plot_potential(ax, state, pot_type)
    ax.set_xlabel(pot_type)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, ymax)

    ax = axs[4]

    xf, yf, uf, vf, xs, ys, us, vs, rs, thetas = state[["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs", "rs", "thetas"]]
    k = 2
    X_vals = [xs, ys, rs, thetas, k]
    slow_indices = get_grid_indices(piecewise_potential, X_vals)
    # slow_indices = [0, 0, 0, 0, 1] # ! Slow indices zero becuase curved paths not discretized in velocity
    slicing_indices = [
        [0, len(config.params.grid.bins.x) - 1],
        [0, len(config.params.grid.bins.y) - 1],
        [slow_indices[2], slow_indices[2] + 1],
        [slow_indices[3], slow_indices[3] + 1],
        [slow_indices[4], slow_indices[4] + 1],
    ]
    slow_subhistogram = extract_submatrix(piecewise_potential.histogram_slow, slicing_indices)
    position_based_emperic_potential = calculate_position_based_emperic_potential(slow_subhistogram, config)
    # matrix_to_plot = get_position_based_emperic_potential_from_state(config, slicing_indices, piecewise_potential)

    subparametrization = extract_submatrix(piecewise_potential.parametrization, slicing_indices)
    center_x = subparametrization[:, :, 0, 0, 0, 0, 0]
    center_y = subparametrization[:, :, 0, 0, 0, 1, 0]
    curvature_x = subparametrization[:, :, 0, 0, 0, 0, 1]
    curvature_y = subparametrization[:, :, 0, 0, 0, 1, 1]

    # sliced_fit_parameters = get_slice_of_multidimensional_matrix(piecewise_potential.fit_params, slices)
    # center_x = sliced_fit_parameters[:, :, 0, 0, 0, 0]
    # center_y = sliced_fit_parameters[:, :, 0, 0, 0, 2]

    # sliced_curvature_x = get_slice_of_multidimensional_matrix(piecewise_potential.curvature_x, slices)
    # curvature_x = sliced_curvature_x[:, :, 0, 0, 0]
    # sliced_curvature_y = get_slice_of_multidimensional_matrix(piecewise_potential.curvature_y, slices)
    # curvature_y = sliced_curvature_y[:, :, 0, 0, 0]

    curv_x = curvature_x * (X - center_x)
    curv_y = curvature_y * (Y - center_y)

    ax.pcolormesh(X, Y, position_based_emperic_potential, cmap=cmap, shading="auto")  # , norm=norm)
    # ax = plot_colorbar(ax, cs)

    scale = 50  # plot_params.scale
    sparseness = 1  # plot_params.sparseness
    ax.quiver(
        X[::sparseness, ::sparseness],
        Y[::sparseness, ::sparseness],
        -curv_x[::sparseness, ::sparseness],
        -curv_y[::sparseness, ::sparseness],
        scale=scale,
        pivot="mid",
        width=0.0015,
        #     labelpos="E",
        #     label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
    )

    tail_window = 15
    dynamics_to_plot = "s"
    xpos, ypos = f"x{dynamics_to_plot}", f"y{dynamics_to_plot}"
    upos, vpos = f"u{dynamics_to_plot}", f"v{dynamics_to_plot}"
    c1 = traj_pid["k"] <= timestep
    c2 = traj_pid["k"] >= timestep - tail_window
    tail = traj_pid[c1 & c2].copy()
    ax.plot(tail[xpos], tail[ypos], "k.", ms=1, alpha=0.5)

    future_window = 5
    c1 = traj_pid["k"] >= timestep
    c2 = traj_pid["k"] <= timestep + future_window
    future = traj_pid[c1 & c2].copy()
    ax.plot(future[xpos], future[ypos], "b.", ms=1, alpha=0.5)
    ax.plot(state[xpos], state[ypos], "bX")

    traj_plot_params = config.params.trajectory_plot
    textstr = (
        f"$\\Delta t=\\,${config.params.model.dt:.3f} s\n"
        f"$\\sigma=\\,${config.params.model.sigma} ms$^{{\\mathdefault{{-3/2}}}}$\n"
        f"$\\tau=\\,${config.params.model.tauu} s\n"
        f"$k=\\,${int(timestep)}"
    )
    props = {"boxstyle": "round", "facecolor": "white", "alpha": 1, "edgecolor": "black", "lw": 0.5}
    plt.figtext(
        0.5,
        0.6,
        textstr,
        ha="center",
        va="center",
        fontsize=5,
        bbox=props,
    )

    ax = apply_xy_plot_style(ax, config.params)
    ax.set_title(config.params.trajectory_plot.title, y=0.9, bbox=props)

    if config.params.trajectory_plot.show_background:
        ax = plot_station_background(ax, config)

    ax = axs[5]
    # ax.grid(False)

    ax.plot(tail[upos], tail[vpos], "k.", ms=1, alpha=0.5)
    ax.plot(future[upos], future[vpos], "b.", ms=1, alpha=0.5)
    ax.plot(state[upos], state[vpos], "bx", ms=5, zorder=-20)

    scale = 5
    ax.quiver([state[upos]], [state[vpos]], [state.Vx], [0], color="C0", scale=scale, zorder=10, width=0.01)
    ax.quiver([state[upos]], [state[vpos]], [0], [state.Vy], color="C1", scale=scale, zorder=10, width=0.01)
    ax.quiver([state[upos]], [state[vpos]], [state.Vu], [0], color="C2", scale=scale, zorder=0, width=0.015)
    ax.quiver([state[upos]], [state[vpos]], [0], [state.Vv], color="C3", scale=scale, zorder=0, width=0.015)
    ax.quiver([state[upos]], [state[vpos]], [state.fx], [state.fy], color="k", scale=scale)
    vel_lim = 1.5
    ax.set_ylim(-vel_lim, vel_lim)
    ax.set_xlim(-2, 2)
    ax.set_ylabel("v [m/s]")
    ax.set_xlabel("u [m/s]")
    ax.set_aspect("equal")

    filename = Path.cwd() / "frames_with_potential_in_each_state/frames" / f"potential_at_slow_index_{int(timestep):05}.png"
    plt.savefig(filename)
    for ax in axs:
        ax.cla()
plt.clf()

# %%
