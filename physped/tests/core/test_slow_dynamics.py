import numpy as np
import pandas as pd

from physped.core.slow_dynamics import low_pass_filter_single_path
from physped.utils.config_utils import initialize_hydra_config

test_trajectories = {
    "xf": [1.0, 3.0, 4.0, 5.0, 6.0],
    "yf": [0, 1, 2, 3, 4],
    "Pid": [0, 0, 0, 0, 0],
    "time": [0, 1, 2, 3, 4],
}


test_config = initialize_hydra_config("tests")


def test_low_pass_filter_single_path():
    df = pd.DataFrame(test_trajectories)
    xf = df["xf"]
    xs = low_pass_filter_single_path(df["xf"], tau=2, dt=1)
    df["xs"] = xs
    assert xs[0] == xf[0]
    assert xs[1] == xf[0] == 0.5 * xs[0] + 0.5 * xf[0]
    for row in np.arange(1, len(xf) - 1):
        assert xs[row + 1] == 1 / 2 * (xs[row] + xf[row])
