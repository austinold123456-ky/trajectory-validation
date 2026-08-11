# Trajectory Validator

`validate_trajectory` validates sampled joint trajectories by checking array shapes, timestamp and position finiteness, per-joint limits, and the maximum motion from the initial position.

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q
```

## Usage

```python
import numpy as np

from trajectory_validator import validate_trajectory

timestamps = np.array([0.0, 1.0])
positions = np.array([[0.0, 0.0], [0.5, -0.5]])
lower_limits = np.array([-1.0, -1.0])
upper_limits = np.array([1.0, 1.0])

result = validate_trajectory(timestamps, positions, lower_limits, upper_limits)
```
