from pathlib import Path

import numpy as np
import pytest

from trajectory_validator import TrajectoryFileError, load_trajectory_csv


def write_csv(tmp_path: Path, name: str, contents: str) -> Path:
    path = tmp_path / name
    path.write_text(contents)
    return path


def assert_csv_error(path: Path, expected_message: str) -> None:
    with pytest.raises(TrajectoryFileError) as error:
        load_trajectory_csv(path)

    message = str(error.value)
    assert message.startswith(f"CSV trajectory file '{path}': ")
    assert expected_message in message


def test_load_trajectory_csv_returns_expected_arrays(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "trajectory.csv",
        "timestamp,joint_0,joint_1\n0.0,1.0,2.0\n0.5,3.0,4.0\n",
    )

    timestamps, positions = load_trajectory_csv(path)

    assert timestamps.shape == (2,)
    assert positions.shape == (2, 2)
    np.testing.assert_array_equal(timestamps, np.array([0.0, 0.5]))
    np.testing.assert_array_equal(positions, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_trajectory_csv_rejects_missing_file(tmp_path: Path) -> None:
    assert_csv_error(tmp_path / "missing.csv", "could not open file")


def test_load_trajectory_csv_rejects_invalid_header(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "invalid_header.csv", "time,joint_0\n0.0,1.0\n")

    assert_csv_error(path, "first header field must be timestamp")


def test_load_trajectory_csv_rejects_malformed_joint_column_sequence(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "invalid_joint_columns.csv", "timestamp,joint_0,joint_2\n0.0,1.0,2.0\n")

    assert_csv_error(path, "joint columns must be named joint_0, joint_1, and so on")


@pytest.mark.parametrize(
    ("contents", "expected_message"),
    [
        ("", "header must not be empty"),
        ("\n", "header must not be empty"),
    ],
    ids=["empty", "blank"],
)
def test_load_trajectory_csv_rejects_empty_or_blank_header(
    tmp_path: Path,
    contents: str,
    expected_message: str,
) -> None:
    path = write_csv(tmp_path, "empty_header.csv", contents)

    assert_csv_error(path, expected_message)


def test_load_trajectory_csv_rejects_header_only_file(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "header_only.csv", "timestamp,joint_0\n")

    assert_csv_error(path, "CSV must contain at least one data row")


def test_load_trajectory_csv_rejects_blank_data_row(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "blank_row.csv", "timestamp,joint_0\n\n")

    assert_csv_error(path, "row 2 does not have the same number of fields as the header")


def test_load_trajectory_csv_rejects_inconsistent_field_count(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "inconsistent_columns.csv",
        "timestamp,joint_0,joint_1\n0.0,1.0\n",
    )

    assert_csv_error(path, "row 2 does not have the same number of fields as the header")


@pytest.mark.parametrize(
    ("contents", "expected_message"),
    [
        ("timestamp,joint_0\n0.0,not-a-number\n", "is not numeric"),
        ("timestamp,joint_0\n0.1abc,0.0\n", "is not numeric"),
    ],
    ids=["non_numeric", "partially_numeric"],
)
def test_load_trajectory_csv_rejects_invalid_numeric_fields(
    tmp_path: Path,
    contents: str,
    expected_message: str,
) -> None:
    path = write_csv(tmp_path, "invalid_numeric.csv", contents)

    assert_csv_error(path, expected_message)


@pytest.mark.parametrize(
    "contents",
    [
        "timestamp,joint_0\n 0.0,1.0\n",
        "timestamp,joint_0\n0.0,1.0 \n",
    ],
    ids=["leading", "trailing"],
)
def test_load_trajectory_csv_rejects_field_whitespace(tmp_path: Path, contents: str) -> None:
    path = write_csv(tmp_path, "field_whitespace.csv", contents)

    assert_csv_error(path, "has leading or trailing whitespace")


def test_load_trajectory_csv_preserves_non_finite_values(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "non_finite.csv",
        "timestamp,joint_0,joint_1\nnan,inf,-inf\n",
    )

    timestamps, positions = load_trajectory_csv(path)

    assert timestamps.shape == (1,)
    assert positions.shape == (1, 2)
    assert np.isnan(timestamps[0])
    assert np.isposinf(positions[0, 0])
    assert np.isneginf(positions[0, 1])
