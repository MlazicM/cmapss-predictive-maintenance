import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTING_COLUMNS


@pytest.fixture
def synthetic_run_to_failure():
    """Three engines with known lifetimes and a linearly degrading sensor."""

    def _build(lifetimes=(40, 60, 100), n_features=2):
        frames = []
        for engine_id, lifetime in enumerate(lifetimes, start=1):
            cycles = np.arange(1, lifetime + 1)
            frame = pd.DataFrame({"engine_id": engine_id, "cycle": cycles})
            for setting in SETTING_COLUMNS:
                frame[setting] = 0.0
            for feature in range(n_features):
                frame[f"sensor_{feature + 1}"] = cycles * (feature + 1.0)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    return _build
