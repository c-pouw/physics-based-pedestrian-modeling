# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.style.use(Path.cwd() / "../conf/science.mplstyle")

x = np.linspace(-6, 6, 100)
y = 0.5 * x**2
df = pd.DataFrame({"x": x, "y": y})
plt.plot(df.x, df.y)

# %%


def fit_function_group(xi_group):
    return lambda x, a, b, c: a * (x - xi_group) ** 2 + b * (x - xi_group)


# Define a function to fit a set of points
def fit_set_of_points(group):
    if len(group) > 2:
        # Get the x and y values
        x = group["x"]
        y = group["y"]

        # Get the initial guess for xi (the mean of x)
        xi = np.mean(x)
        yi = np.mean(y)
        fit_function = fit_function_group(xi)

        # Get the initial guess for a and b (0)
        a_initial = 0
        b_initial = 0
        c_initial = 0

        # Fit the function to the data
        popt, _ = curve_fit(fit_function, x, y, p0=[a_initial, b_initial, c_initial])

        # Return the fitted parameters
        return {"xi": xi, "yi": yi, "a": popt[0], "b": popt[1], "c": popt[2]}


# %%

xbins1 = np.arange(-5.5, 7.5, 1)
df["xcuts1"] = pd.cut(df.x, bins=xbins1, labels=xbins1[:-1])
xbins2 = np.arange(-6, 7, 4)
df["xcuts2"] = pd.cut(df.x, bins=xbins2, labels=xbins2[:-1])

# Apply the function to each group
fits1 = df.groupby("xcuts1").apply(fit_set_of_points)
fits2 = df.groupby("xcuts2").apply(fit_set_of_points)

# %%
fig, axs = plt.subplots(3, 2, figsize=(3.54, 3), sharey=False, sharex=True)
ax = axs[0, 0]

ax.plot(df.x, df.y, "-", color="r", lw=2, alpha=0.3)
avals1 = []
bvals1 = []
xivals1 = []
for key, fit in fits1.items():
    xi = fit["xi"]
    yi = fit["yi"]
    a = fit["a"]
    b = fit["b"]
    c = fit["c"]
    avals1.append(a)
    bvals1.append(b)
    xivals1.append(xi)
    x = np.linspace(xi - 1, xi + 1, 100)
    y = fit_function_group(xi)(x, a, b, c)
    ax.plot(x, y + yi, "--")
ax.set_ylim(0, 15)
ax.set_xlim(-5, 5)
ax.grid(False)
for x in xbins1:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")
ax.set_ylabel("$V(x)$")
ax.set_title(f"\\#bins = {len(xbins1) - 1}; width = {int(xbins1[1] - xbins1[0])}")

ax = axs[1, 0]
ax.plot(xivals1, avals1, ".", label="a")
ax.set_ylim(0, 100)
ax.set_ylabel("$a$")
for x in xbins1:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")

ax = axs[2, 0]
ax.plot(xivals1, bvals1, ".", color="C1", label="b")
ax.set_ylabel("$b$")
ax.set_xlabel("$x$")
ax.set_ylim(-7, 7)
for x in xbins1:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")

ax = axs[0, 1]
ax.plot(df.x, df.y, "-", color="r", lw=2, alpha=0.3)
avals2 = []
bvals2 = []
xivals2 = []
for key, fit in fits2.items():
    xi = fit["xi"]
    yi = fit["yi"]
    a = fit["a"]
    b = fit["b"]
    c = fit["c"]
    avals2.append(a)
    bvals2.append(b)
    xivals2.append(xi)
    x = np.linspace(xi - 4, xi + 4, 100)
    y = fit_function_group(xi)(x, a, b, c)
    ax.plot(x, y + yi, "--")
ax.set_ylim(0, 15)
ax.set_xlim(-5, 5)
ax.grid(False)
for x in xbins2:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")

ax.set_ylabel("$V(x)$")
ax.set_title(f"\\#bins = {len(xbins2) - 1}; width = {xbins2[1] - xbins2[0]}")

ax = axs[1, 1]
ax.plot(xivals2, avals2, ".", label="a")
ax.set_ylim(0, 100)
for x in xbins2:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")

ax = axs[2, 1]
ax.plot(xivals2, bvals2, ".", color="C1", label="b")
ax.set_ylim(-7, 7)
ax.set_xlabel("$x$")
for x in xbins2:
    ax.plot([x, x], [-1, 1], color="r", alpha=0.8, ls="solid")

fig.suptitle("Piecewise approximation $V(x) = a(x-x_i)^2 + b(x-x_i)$", y=1)
plt.tight_layout()
plt.savefig("../figures/piecewise_approx_of_2nd_order_potential.pdf", bbox_inches="tight")
# %%
