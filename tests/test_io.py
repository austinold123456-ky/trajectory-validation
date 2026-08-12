from pathlib import Path

import numpy as np
import pytest

from trajectory_validator import load_trajectory_npz, save_trajectory_npz


def test_load_trajectory_npz_returns_saved_arrays(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.npz"
    timestamps = np.array([0.0, 1.0])
    positions = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.savez(trajectory_path, timestamps=timestamps, positions=positions)

    loaded_timestamps, loaded_positions = load_trajectory_npz(trajectory_path)

    np.testing.assert_array_equal(loaded_timestamps, timestamps)
    np.testing.assert_array_equal(loaded_positions, positions)


def test_save_trajectory_npz_round_trips_arrays(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "saved_trajectory.npz"
    timestamps = np.array([0.0, 1.0])
    positions = np.array([[0.0, 1.0], [2.0, 3.0]])

    save_trajectory_npz(trajectory_path, timestamps, positions)
    loaded_timestamps, loaded_positions = load_trajectory_npz(trajectory_path)

    np.testing.assert_array_equal(loaded_timestamps, timestamps)
    np.testing.assert_array_equal(loaded_positions, positions)


def test_load_trajectory_npz_rejects_missing_positions(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "missing_positions.npz"
    np.savez(trajectory_path, timestamps=np.array([0.0]))

    with pytest.raises(ValueError, match="missing required 'positions' array"):
        load_trajectory_npz(trajectory_path)


def test_load_trajectory_npz_rejects_missing_timestamps(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "missing_timestamps.npz"
    np.savez(trajectory_path, positions=np.array([[0.0]]))

    with pytest.raises(ValueError, match="missing required 'timestamps' array"):
        load_trajectory_npz(trajectory_path)
