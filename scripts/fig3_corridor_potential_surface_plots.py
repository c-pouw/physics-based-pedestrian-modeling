# %%
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(
    "/home/pouw/workspace/crowd-tracking/2020-XX-Pouw-Corbetta-pathintegral-codes/physped/visualization/science.mplstyle"
)

# %%

# Create 2x2 sub plots
gs = gridspec.GridSpec(1, 2, hspace=0, wspace=0)

fig = pl.figure()
ax = pl.subplot(gs[0, 0], projection="3d")  # row 0, col 0
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(2, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-0.6, 0.6)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-4)
ax.xaxis.labelpad = -7

# y
ax.set_ylabel("x [m]")
ax.set_ylim(-1, 1)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([-1, 0, 1], rotation=0, verticalalignment="baseline", horizontalalignment="center")

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])
ax.tick_params(axis="z", which="major", pad=5)

# ------------------------------------------------------------
ax = pl.subplot(gs[0, 1], projection="3d")  # row 0, col 1
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(3, 6, 1))

X = np.arange(-2, 2, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = 1.5 * (X) ** 2 + 0.3 * (Y**2 - 1.3**2) ** 2
Z1 = np.where(Z1 > 1, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

# x
ax.set_xlim(-1.1, 1.1)
ax.set_xlabel("v [$\\mathrm{m\\,s}^{-1}$]")
ax.set_xticks([-1, 0, 1])
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-5)
ax.xaxis.labelpad = -6

# y
ax.set_ylabel("u [$\\mathrm{m\\,s}^{-1}$]")
ax.set_ylim(-2.5, 2.5)
ax.set_yticks([-2.5, -1.3, 0, 1.3, 2.5])
# ax.set_yticklabels()
# ax.tick_params(axis='y', which='major', pad=-2)
ax.set_yticklabels([-2.5, "$-u_s$", 0, "$u_s$", 2.5], rotation=0, verticalalignment="baseline", horizontalalignment="center")
# ax.yaxis.labelpad=-2

# z
ax.set_zlim(0, 0.8)
ax.set_zticks([1.1])
ax.set_zticklabels(["$\\phi$(u,v)"])
# ax.tick_params(axis='z', which='major', pad=0)

plt.savefig("../figures/potentials_3d.pdf")


# %%

# fig, ax = plt.subplots(figsize = (5, 5), subplot_kw={"projection": "3d"})
fig = plt.figure(constrained_layout=True)
ax = plt.axes(projection="3d")
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
# ax.dist = 20

ax.set_box_aspect(aspect=(8, 4, 1))


X = np.arange(-1, 1, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-2, 2, steps)
zz = np.repeat(0.1, steps)
y0s = [-1.8, -0.3, 2.8]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([3, 1, 0.5, 1]))
# ax.set_aspect(aspect = 'equalxy')
# ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")])

# ax = plt.axes(projection='3d')
# ax.set_box_aspect(aspect = (8,2,2))

# x
ax.set_xlim(-4, 4)
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 15

# y
ax.set_ylabel("x [m]")
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -7

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])
# ax.tick_params(axis='z', which='major', pad=-1)


# plt.tight_layout()
plt.savefig("../figures/confinement_potential_wide_corridor.pdf")
plt.show()

# %%

fig = plt.figure(constrained_layout=True)
ax = plt.axes(projection="3d")
ax.minorticks_off()
ax.view_init(elev=23, azim=75)

ax.set_box_aspect(aspect=(2, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-0.6, 0.6)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = -5

# y
ax.set_ylabel("x [m]")
ax.set_ylim(-1, 1)
ax.set_yticks([-1, 0, 1])
# ax.tick_params(axis='y', which='major', pad=-3)
# ax.yaxis.labelpad=-7

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])
ax.tick_params(axis="z", which="major", pad=12)
# ax.zaxis.labelpad=3

ax.minorticks_off()


# plt.tight_layout()
plt.savefig("../figures/confinement_potential_narrow_corridor.pdf")
plt.show()

# %%

fig = plt.figure(constrained_layout=True)
ax = plt.axes(projection="3d")
ax.minorticks_off()
ax.view_init(elev=23, azim=75)

ax.set_box_aspect(aspect=(8, 4, 1))

X = np.arange(-2, 2, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = 0.5 * (Y) ** 2 + 0.5 * (X**2 - 1**2) ** 2
Z1 = np.where(Z1 > 0.8, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    # xx = []
    # for i in range(steps):
    #     xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    # ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms = 1)

# x
ax.set_xlim(-2, 2)
# ax.set_xticks([-0.5,0,0.5])
ax.set_xlabel("u [$\\mathrm{m\\,s}^{-1}$]")
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = 3

# y
ax.set_ylabel("v [m/s]")
ax.set_ylim(-1.5, 1.5)
ax.set_yticks([-1, 0, 1])
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -7

# z
ax.set_zlim(0, 0.8)
ax.set_zticks([0.8])
ax.set_zticklabels(["$\\phi$(x,y)"])
ax.tick_params(axis="z", which="major", pad=0)
# ax.zaxis.labelpad=3

ax.minorticks_off()

# plt.tight_layout()
plt.savefig("../figures/velocity_potential_corridor.pdf")
plt.show()

# %%

# fig, axs = plt.subplots(
#     2,
#     2,
#     constrained_layout=True,
#     gridspec_kw={"height_ratios": [1, 1], "width_ratios": [1, 1]},
# )

# # Remove the unused subplot
# fig.delaxes(axs[1, 1])

# ax1 = axs[0, 0].add_subplot(111, projection="3d")
# ax2 = axs[0, 1].add_subplot(111, projection="3d")
# ax3 = axs[1, 0].add_subplot(111, projection="3d")

# %%

# Create 2x2 sub plots
gs = gridspec.GridSpec(3, 2, hspace=0, wspace=0.5)

fig = pl.figure(figsize=(3.54, 3.54))
ax = pl.subplot(gs[0, 0], projection="3d")  # row 0, col 0
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(2, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-0.6, 0.6)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = -5

# y
ax.set_ylabel("x [m]")
ax.set_ylim(-1, 1)
ax.set_yticks([-1, 0, 1])

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])
ax.tick_params(axis="z", which="major", pad=12)

# ------------------------------------------------------------
ax = pl.subplot(gs[0, 1], projection="3d")  # row 0, col 1
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(4, 8, 1))

X = np.arange(-2, 2, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = 0.5 * (X) ** 2 + 0.5 * (Y**2 - 1**2) ** 2
Z1 = np.where(Z1 > 0.8, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

# x
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel("v [$\\mathrm{m\\,s}^{-1}$]")
ax.set_xticks([-1, 0, 1])
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = -3

# y
ax.set_ylabel("u [$\\mathrm{m\\,s}^{-1}$]")
ax.set_ylim(-2, 2)
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -4

# z
ax.set_zlim(0, 0.8)
ax.set_zticks([0.8])
ax.set_zticklabels(["$\\phi$(x,y)"])
ax.tick_params(axis="z", which="major", pad=0)

# ------------------------------------------------------------
ax = pl.subplot(gs[1:, :], projection="3d")  # row 1, span all columns
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(8, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-2, 2, steps)
zz = np.repeat(0.1, steps)
y0s = [-1.8, -0.3, 2.8]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-4, 4)
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 15

# y
ax.set_ylabel("x [m]")
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -7

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])

# plt.subplots_adjust(left=0, right=0, top=3, bottom=0)

# plt.tight_layout()
plt.savefig("../figures/potentials_3d.pdf")
# plt.show()


# ------------------------------------------------------------
# %%
fig = pl.figure()
ax = pl.subplot(projection="3d")  # row 1, span all columns
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(8, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-2, 2, steps)
zz = np.repeat(0.1, steps)
y0s = [-1.8, -0.3, 2.8]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-4, 4)
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 15

# y
ax.set_ylabel("x [m]")
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -7

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig("../figures/confinement_potential_wide_corridor.pdf")
# %%

# Create 2x2 sub plots
gs = gridspec.GridSpec(1, 4, hspace=0, wspace=0.5)

fig = pl.figure(figsize=(7, 3.54))
ax = pl.subplot(gs[0], projection="3d")  # row 0, col 0
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(2, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-0.6, 0.6)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = -5

# y
ax.set_ylabel("x [m]")
ax.set_ylim(-1, 1)
ax.set_yticks([-1, 0, 1])

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])
ax.tick_params(axis="z", which="major", pad=12)

# ------------------------------------------------------------
ax = pl.subplot(gs[1], projection="3d")  # row 0, col 1
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(4, 8, 1))

X = np.arange(-2, 2, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = 0.5 * (X) ** 2 + 0.5 * (Y**2 - 1**2) ** 2
Z1 = np.where(Z1 > 0.8, np.nan, Z1)

steps = 20
yy = np.linspace(-1, 1, steps)
zz = np.repeat(0.1, steps)
y0s = [0]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

# x
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel("v [$\\mathrm{m\\,s}^{-1}$]")
ax.set_xticks([-1, 0, 1])
ax.xaxis.labelpad = 20
ax.tick_params(axis="x", which="major", pad=-3)
ax.xaxis.labelpad = -3

# y
ax.set_ylabel("u [$\\mathrm{m\\,s}^{-1}$]")
ax.set_ylim(-2, 2)
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -4

# z
ax.set_zlim(0, 0.8)
ax.set_zticks([0.8])
ax.set_zticklabels(["$\\phi$(x,y)"])
ax.tick_params(axis="z", which="major", pad=0)

# ------------------------------------------------------------
ax = pl.subplot(gs[2:], projection="3d")  # row 1, span all columns
ax.minorticks_off()
ax.view_init(elev=23, azim=75)
ax.set_box_aspect(aspect=(8, 4, 1))

X = np.arange(-1, 1, 0.01)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z1 = (X) ** 2
Z1 = np.where(Z1 > 0.4, np.nan, Z1)

steps = 20
yy = np.linspace(-2, 2, steps)
zz = np.repeat(0.1, steps)
y0s = [-1.8, -0.3, 2.8]
for y0 in y0s:
    ax.plot_surface(X + y0, Y, Z1, cmap="Blues", linewidth=0, antialiased=False)

    xx = []
    for i in range(steps):
        xx.append(y0 + (np.random.rand() * 2 - 1) / 7)
    ax.plot(xx, yy, zz, ".-", zorder=10, c="k", lw=0.5, ms=1)

# x
ax.set_xlim(-4, 4)
ax.set_xlabel("y [m]")
ax.xaxis.labelpad = 15

# y
ax.set_ylabel("x [m]")
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis="y", which="major", pad=-3)
ax.yaxis.labelpad = -7

# z
ax.set_zlim(0, 0.5)
ax.set_zticks([0.5])
ax.set_zticklabels(["V(x,y)"])

# plt.subplots_adjust(left=0, right=0, top=3, bottom=0)

# plt.tight_layout()
plt.savefig("../figures/potentials_3d.pdf")
# plt.show()

# %%
