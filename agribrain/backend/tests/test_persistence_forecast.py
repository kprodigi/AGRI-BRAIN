from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.persistence_forecast import persistence_forecast


def test_persistence_point_and_empirical_uncertainty() -> None:
    frame = pd.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 16.0]})
    result = persistence_forecast(frame, series_col="x", horizon=2)
    assert result["forecast"] == [16.0, 16.0]
    expected_std = float(np.std(np.diff(frame["x"].to_numpy()), ddof=0))
    assert result["std"] == pytest.approx(expected_std, abs=1e-6)
    assert result["ci_lower"][1] <= result["ci_lower"][0]
    assert result["ci_upper"][1] >= result["ci_upper"][0]


def test_persistence_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        persistence_forecast(
            pd.DataFrame({"x": [1.0, 2.0]}), series_col="x", horizon=0,
        )
