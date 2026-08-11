import numpy as np
from validate import validate_trajectory

report = validate_trajectory(
    timestamps=np.array([0.0, 0.1, 0.2]),
    positions=np.array([
        [0.0, 0.2],
        [0.4, 2.0],
        [np.nan, 0.5],
    ]),
    lower_limits=np.array([-1.0, -1.0]),
    upper_limits=np.array([1.0, 1.0]),
)

print(report)