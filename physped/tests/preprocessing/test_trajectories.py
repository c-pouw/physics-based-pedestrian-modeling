import numpy as np
import pandas as pd

from physped.preprocessing.trajectories import (
    add_trajectory_index,
    compute_velocity_from_positions,
    preprocess_trajectories,
    prune_short_trajectories,
    rename_columns,
    transform_fast_velocity_to_polar_coordinates,
)
from physped.utils.config_utils import initialize_hydra_config

test_trajectories = {
    "xpos": [0, 1, 2, 3, 4, 0, 1, 2, 3, 0],
    "ypos": [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
    "pid": [0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    "tstep": [0, 1, 2, 3, 4, 0, 1, 2, 3, 1],
}

test_config = initialize_hydra_config("tests")


def test_rename_columns():
    df = pd.DataFrame(test_trajectories)
    df = rename_columns(df, test_config)
    assert df.columns.tolist() == ["xf", "yf", "Pid", "time"]


def test_prune_short_trajectories():
    df = pd.DataFrame(test_trajectories)
    df = rename_columns(df, test_config)
    pruned_df = prune_short_trajectories(df, test_config)
    assert len(pruned_df) < len(df)


def test_add_trajectory_index():
    df = pd.DataFrame(test_trajectories)
    df = rename_columns(df, test_config)
    df = add_trajectory_index(df, test_config)
    assert "k" in df.columns
    assert df.k.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0]


def test_compute_velocity_from_positions():
    df = pd.DataFrame(test_trajectories)
    df = rename_columns(df, test_config)
    df = prune_short_trajectories(df, test_config)
    df = compute_velocity_from_positions(df, test_config)
    assert "uf" in df.columns
    assert "vf" in df.columns
    expected_uf = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    expected_vf = [1, 1, 1, 1, 1, -1, -1, -1, -1]
    np.testing.assert_allclose(df.uf.tolist(), expected_uf)
    np.testing.assert_allclose(df.vf.tolist(), expected_vf)


def test_transform_fast_velocity_to_polar_coordinates():
    df = pd.DataFrame(test_trajectories)
    df = rename_columns(df, test_config)
    df = prune_short_trajectories(df, test_config)
    df = compute_velocity_from_positions(df, test_config)
    df = transform_fast_velocity_to_polar_coordinates(df, test_config)
    assert "rf" in df.columns
    assert "thetaf" in df.columns
    np.testing.assert_allclose(df.rf.tolist(), np.repeat(np.sqrt(2), 9))
    np.testing.assert_allclose(df.thetaf.tolist(), np.append(np.repeat(np.pi / 4, 5), (np.repeat(-np.pi / 4, 4))))


def test_preprocess_trajectories():
    """Test the full preprocessing pipeline."""
    df = pd.DataFrame(test_trajectories)
    df = preprocess_trajectories(df, config=test_config)
    columns = ["xf", "yf", "Pid", "time", "traj_len", "k", "uf", "vf", "rf", "thetaf"]
    for col in columns:
        assert col in df.columns
