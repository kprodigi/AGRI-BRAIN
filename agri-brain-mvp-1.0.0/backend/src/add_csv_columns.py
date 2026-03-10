"""Add quality_preference and regulatory_temp_max columns to data_spinach.csv.

Run this script once to update the CSV with the new columns required
by the customer preferences and regulatory standards features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data_spinach.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

rng = np.random.default_rng(42)
n = len(df)

# quality_preference: default 0.85, normal variation std=0.03
df["quality_preference"] = np.clip(
    rng.normal(0.85, 0.03, size=n), 0.0, 1.0
).round(4)

# regulatory_temp_max: default 8.0, occasional 5.0 at random rows
reg_temp = np.full(n, 8.0)
# ~10% of rows get stricter 5.0 threshold
strict_mask = rng.random(n) < 0.10
reg_temp[strict_mask] = 5.0
df["regulatory_temp_max"] = reg_temp

df.to_csv(DATA_PATH, index=False)
print(f"Updated {DATA_PATH}: added quality_preference and regulatory_temp_max columns ({n} rows)")
