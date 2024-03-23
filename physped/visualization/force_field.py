import numpy as np


def clip_fields(fields, clip=50):
    """Clip fields."""
    clip_min = -clip
    clip_max = clip
    par_fields = list(fields.keys())
    for field_name in par_fields:
        field = fields[field_name].copy()
        field[field == -np.inf] = np.nan
        field[field == np.inf] = np.nan
        fields[field_name] = np.clip(field, clip_min, clip_max)
    return fields


def plot_quiver(ax, params, fields):
    """Plot the force field."""
    clip = params["force_field_plot"]["clip"]
    scale = params["force_field_plot"]["scale"]
    sparseness = params["force_field_plot"]["sparseness"]
    if clip == 0:
        clip = np.inf

    clipped_fields = clip_fields(fields, clip)

    q = ax.quiver(
        clipped_fields["X"][::sparseness],
        clipped_fields["Y"][::sparseness],
        clipped_fields["dfx_clipped"][::sparseness],
        clipped_fields["dfy_clipped"][::sparseness],
        scale=scale,
        width=0.002,
    )

    ax.quiverkey(
        q,
        X=0.2,
        Y=1.1,
        U=50,
        label="Vectors: $f^{\\prime }(x)=-{\\frac {x-\\mu }{\\sigma ^{2}}}f(x)$",
        labelpos="E",
    )
    ax.set_aspect("equal")
    # if params["name"] == "station_paths":
    #        ax = plot_station_background(val, ax)
    #   ax.set_xlim(val.grid.xgrid[0], val.grid.xgrid[-1])
    #  ax.set_ylim(val.grid.ygrid[0], val.grid.ygrid[-1])
    return ax
