from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Reproducibility
RANDOM_STATE = 42

# Classification horizons
FAILURE_HORIZONS = (10, 20, 30)

# NASA C-MAPSS column names
CMAPSS_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)