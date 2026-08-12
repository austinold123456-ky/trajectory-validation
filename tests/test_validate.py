import numpy as np
import pytest

from trajectory_validator import (
    TrajectoryStructureError,
    TrajectoryValidationReport,
    validate_trajectory,
)


def test_valid_trajectory() -> None:
    report = validate_trajectory(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 1.0]]),
        np.array([-1.0, 0.0]),
        np.array([2.0, 2.0]),
    )

    assert isinstance(report, TrajectoryValidationReport)
    assert report.is_valid is True
    np.testing.assert_array_equal(report.invalid_rows, np.array([], dtype=int))
    np.testing.assert_array_equal(report.non_finite_rows, np.array([], dtype=int))
    np.testing.assert_array_equal(report.out_of_limit_rows, np.array([], dtype=int))


def test_rejects_mismatched_timestamp_and_position_lengths() -> None:
    with pytest.raises(TrajectoryStructureError, match="timestamps length"):
        validate_trajectory(
            np.array([0.0]),
            np.array([[0.0], [1.0]]),
            np.array([-1.0]),
            np.array([1.0]),
        )


def test_rejects_empty_trajectory() -> None:
    with pytest.raises(TrajectoryStructureError, match="trajectory must contain at least one time sample"):
        validate_trajectory(
            np.array([]),
            np.empty((0, 2)),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_reports_non_finite_positions(bad_value: float) -> None:
    report = validate_trajectory(
        np.array([0.0, 1.0]),
        np.array([[0.0, 0.0], [bad_value, 0.5]]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
    )

    assert report.is_valid is False
    np.testing.assert_array_equal(report.non_finite_rows, np.array([1]))
    np.testing.assert_array_equal(report.invalid_rows, np.array([1]))
    np.testing.assert_array_equal(report.out_of_limit_rows, np.array([], dtype=int))
    assert np.isnan(report.maximum_motion_per_joint).all()


def test_reports_non_finite_timestamp() -> None:
    report = validate_trajectory(
        np.array([0.0, np.nan]),
        np.array([[0.0], [0.5]]),
        np.array([-1.0]),
        np.array([1.0]),
    )

    assert report.is_valid is False
    np.testing.assert_array_equal(report.non_finite_rows, np.array([1]))
    np.testing.assert_array_equal(report.invalid_rows, np.array([1]))
    np.testing.assert_array_equal(report.maximum_motion_per_joint, np.array([0.5]))


def test_reports_values_outside_joint_limits() -> None:
    report = validate_trajectory(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 0.0], [1.5, 0.0], [0.0, -1.5]]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
    )

    assert report.is_valid is False
    np.testing.assert_array_equal(report.out_of_limit_rows, np.array([1, 2]))
    np.testing.assert_array_equal(report.invalid_rows, np.array([1, 2]))


def test_rejects_wrong_limit_array_length() -> None:
    with pytest.raises(TrajectoryStructureError, match="lower_limits length"):
        validate_trajectory(
            np.array([0.0]),
            np.array([[0.0, 0.0]]),
            np.array([-1.0]),
            np.array([1.0, 1.0]),
        )


def test_calculates_maximum_motion_per_joint() -> None:
    report = validate_trajectory(
        np.array([0.0, 1.0, 2.0]),
        np.array([[1.0, -2.0], [4.0, -1.0], [-1.0, -6.0]]),
        np.array([-10.0, -10.0]),
        np.array([10.0, 10.0]),
    )

    np.testing.assert_array_equal(report.maximum_motion_per_joint, np.array([3.0, 4.0]))


def test_does_not_modify_input_arrays() -> None:
    timestamps = np.array([0.0, 1.0])
    positions = np.array([[0.0, 0.0], [np.nan, 2.0]])
    lower_limits = np.array([-1.0, -1.0])
    upper_limits = np.array([1.0, 1.0])
    originals = [array.copy() for array in (timestamps, positions, lower_limits, upper_limits)]

    validate_trajectory(timestamps, positions, lower_limits, upper_limits)

    for actual, original in zip(
        (timestamps, positions, lower_limits, upper_limits), originals
    ):
        assert np.array_equal(actual, original, equal_nan=True)
