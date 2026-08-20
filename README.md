# Trajectory Validator

`validate_trajectory` validates sampled joint trajectories: array structure, finite joint positions and timestamps, strictly increasing timestamps, per-joint limits, and maximum motion from the initial position.

## Installation

Install the local package:

```bash
python -m pip install .
```

Install the optional test dependencies and run the test suite:

```bash
python -m pip install ".[test]"
python -m pytest -q
```

## CSV trajectory loading

```python
from trajectory_validator import load_trajectory_csv

timestamps, positions = load_trajectory_csv("trajectory.csv")
# timestamps.shape == (N,)
# positions.shape == (N, D)
```

CSV loading parses data only. Use `validate_trajectory` separately to check trajectory structure and semantics.

## CSV contract

- The header must be exactly `timestamp,joint_0,joint_1,...` with at least one joint column.
- Fields are unquoted, comma-separated numeric values; every data row must have the header's column count.
- The file must contain at least one data row, and data fields cannot have leading or trailing whitespace.
- Malformed or non-numeric fields raise `TrajectoryFileError`.
- `nan` and `inf` load successfully; `validate_trajectory` subsequently reports their rows as non-finite.

## Timestamp validation

`validate_trajectory` requires timestamps to be finite and strictly increasing. A duplicate or decreasing later timestamp makes the report invalid and appears in `non_increasing_timestamp_rows`; non-finite timestamps appear in `non_finite_rows`.

## Validation usage

```python
import numpy as np

from trajectory_validator import validate_trajectory

timestamps = np.array([0.0, 1.0])
positions = np.array([[0.0, 0.0], [0.5, -0.5]])
lower_limits = np.array([-1.0, -1.0])
upper_limits = np.array([1.0, 1.0])

report = validate_trajectory(timestamps, positions, lower_limits, upper_limits)
```
