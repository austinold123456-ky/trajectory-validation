"""Input and output helpers for trajectory data."""

from pathlib import Path

import numpy as np


class TrajectoryFileError(ValueError):
    pass


def _csv_error(path: Path, message: str) -> TrajectoryFileError:
    return TrajectoryFileError(f"CSV trajectory file '{path}': {message}")


def _parse_csv_numeric_field(
    field: str,
    path: Path,
    row_number: int,
    column_number: int,
) -> float:
    if field != field.strip():
        raise _csv_error(
            path,
            f"row {row_number}, column {column_number} has leading or trailing whitespace",
        )

    try:
        return float(field)
    except ValueError as error:
        raise _csv_error(
            path,
            f"row {row_number}, column {column_number} is not numeric",
        ) from error


def load_trajectory_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and positions arrays from a trajectory CSV file."""
    trajectory_path = Path(path)

    try:
        with trajectory_path.open("r", newline=None) as trajectory_file:
            header_line = trajectory_file.readline()
            if not header_line:
                raise _csv_error(trajectory_path, "header must not be empty")

            header_line = header_line.rstrip("\n")
            if not header_line:
                raise _csv_error(trajectory_path, "header must not be empty")

            header = header_line.split(",")
            if header[0] != "timestamp":
                raise _csv_error(trajectory_path, "first header field must be timestamp")
            if len(header) < 2:
                raise _csv_error(
                    trajectory_path,
                    "header must contain at least one joint column",
                )
            for column_index, column_name in enumerate(header[1:]):
                expected_name = f"joint_{column_index}"
                if column_name != expected_name:
                    raise _csv_error(
                        trajectory_path,
                        "joint columns must be named joint_0, joint_1, and so on",
                    )

            timestamps: list[float] = []
            position_rows: list[list[float]] = []
            for row_number, data_line in enumerate(trajectory_file, start=2):
                fields = data_line.rstrip("\n").split(",")
                if len(fields) != len(header):
                    raise _csv_error(
                        trajectory_path,
                        f"row {row_number} does not have the same number of fields as the header",
                    )

                timestamps.append(
                    _parse_csv_numeric_field(fields[0], trajectory_path, row_number, 0)
                )
                position_rows.append(
                    [
                        _parse_csv_numeric_field(
                            field,
                            trajectory_path,
                            row_number,
                            column_number,
                        )
                        for column_number, field in enumerate(fields[1:], start=1)
                    ]
                )
    except OSError as error:
        raise _csv_error(trajectory_path, "could not open file") from error

    if not timestamps:
        raise _csv_error(trajectory_path, "CSV must contain at least one data row")

    return np.array(timestamps, dtype=float), np.array(position_rows, dtype=float)


def load_trajectory_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and positions arrays from an NPZ file."""
    with np.load(path, allow_pickle=False) as trajectory_file:
        if "timestamps" not in trajectory_file:
            raise ValueError("NPZ file is missing required 'timestamps' array")
        if "positions" not in trajectory_file:
            raise ValueError("NPZ file is missing required 'positions' array")

        return trajectory_file["timestamps"], trajectory_file["positions"]


def save_trajectory_npz(
    path: str | Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Save timestamps and positions arrays to an NPZ file."""
    np.savez(path, timestamps=timestamps, positions=positions)
