"""Central configuration for the CMAPSS pipeline.

Every notebook and module imports its constants from here so that a change to
the sensor list or the RUL cap propagates to the whole project instead of being
copy-pasted into four notebooks.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Resolved relative to the project root, so imports work from notebooks/ too.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Schema ----------------------------------------------------------------
# The CMAPSS .txt files are headerless and space separated.
INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensors that carry degradation signal on FD001 (see notebooks/01_eda.ipynb).
# The remaining ten are constant or near-constant and were dropped.
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_9", "sensor_11", "sensor_12", "sensor_14",
    "sensor_15", "sensor_20", "sensor_21",
]

# --- Modelling defaults ----------------------------------------------------
# Piecewise-linear RUL: degradation is not observable while the engine is
# healthy, so every label above the cap is treated as "as good as new".
RUL_CAP = 125
SEQUENCE_LENGTH = 30
RANDOM_SEED = 42

# FD002 and FD004 are recorded under six discrete operating regimes.
N_OPERATING_REGIMES = 6

# What each benchmark subset actually contains. The regime count decides how a
# subset must be normalised: on the six-regime subsets the operating condition
# moves the sensors far more than degradation does, so a single global scaler
# mostly encodes "which regime is this" and drowns the signal.
SUBSET_INFO = {
    "FD001": {"regimes": 1, "faults": 1, "train_engines": 100},
    "FD002": {"regimes": 6, "faults": 1, "train_engines": 260},
    "FD003": {"regimes": 1, "faults": 2, "train_engines": 100},
    "FD004": {"regimes": 6, "faults": 2, "train_engines": 249},
}
SUBSETS = list(SUBSET_INFO)
