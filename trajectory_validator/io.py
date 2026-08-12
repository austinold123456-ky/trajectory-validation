"""Input and output helpers for trajectory data."""

from pathlib import Path

import numpy as np


def load_trajectory_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and positions arrays from an NPZ file."""
    with np.load(path, allow_pickle=False) as trajectory_file:
        if "timestamps" not in trajectory_file:
            raise ValueError("NPZ file is missing required 'timestamps' array")
        if "positions" not in trajectory_file:
            raise ValueError("NPZ file is missing required 'positions' array")

        return trajectory_file["timestamps"], trajectory_file["positions"]
