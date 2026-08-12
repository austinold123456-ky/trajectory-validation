"""Validation helpers for sampled joint trajectories."""

from dataclasses import dataclass

import numpy as np


class TrajectoryStructureError(ValueError):
    """Raised when trajectory inputs have incompatible structure."""


@dataclass(frozen=True)
class TrajectoryValidationReport:
    """Results from validating a joint trajectory."""

    is_valid: bool
    invalid_rows: np.ndarray
    non_finite_rows: np.ndarray
    out_of_limit_rows: np.ndarray
    maximum_motion_per_joint: np.ndarray


def validate_trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> TrajectoryValidationReport:
    """Validate a joint trajectory against its per-joint limits."""
    if timestamps.ndim != 1:
        raise TrajectoryStructureError("timestamps must be a 1-dimensional array")
    if positions.ndim != 2:
        raise TrajectoryStructureError("positions must be a 2-dimensional array")
    if timestamps.shape[0] != positions.shape[0]:
        raise TrajectoryStructureError("timestamps length must match positions.shape[0]")
    if positions.shape[0] == 0:
        raise TrajectoryStructureError("trajectory must contain at least one time sample")
    if lower_limits.ndim != 1:
        raise TrajectoryStructureError("lower_limits must be a 1-dimensional array")
    if upper_limits.ndim != 1:
        raise TrajectoryStructureError("upper_limits must be a 1-dimensional array")

    joint_count = positions.shape[1]
    if lower_limits.shape[0] != joint_count:
        raise TrajectoryStructureError("lower_limits length must match positions.shape[1]")
    if upper_limits.shape[0] != joint_count:
        raise TrajectoryStructureError("upper_limits length must match positions.shape[1]")
    if np.any(lower_limits > upper_limits):
        raise TrajectoryStructureError(
            "each lower limit must be less than or equal to its upper limit"
        )

    finite_positions = np.isfinite(positions)
    finite_timestamps = np.isfinite(timestamps)
    non_finite_rows = np.flatnonzero(~finite_timestamps | ~np.all(finite_positions, axis=1))

    finite_limit_violations = finite_positions & (
        (positions < lower_limits) | (positions > upper_limits)
    )
    out_of_limit_rows = np.flatnonzero(np.any(finite_limit_violations, axis=1))

    invalid_rows = np.union1d(non_finite_rows, out_of_limit_rows)
    if np.any(~finite_positions):
        maximum_motion_per_joint = np.full(joint_count, np.nan)
    else:
        maximum_motion_per_joint = np.max(np.abs(positions - positions[0]), axis=0)

    return TrajectoryValidationReport(
        is_valid=invalid_rows.size == 0,
        invalid_rows=invalid_rows,
        non_finite_rows=non_finite_rows,
        out_of_limit_rows=out_of_limit_rows,
        maximum_motion_per_joint=maximum_motion_per_joint,
    )
