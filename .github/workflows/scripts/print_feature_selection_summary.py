#!/usr/bin/env python3
"""Print a one-line stage summary from selected_features.json (used by
.github/workflows/ml_feature_selection.yml's Summary step)."""
import json
from pathlib import Path

path = Path("ml_models/feature_selection/selected_features.json")
with open(path) as f:
    d = json.load(f)

line = (
    f"{d['stage0_count']} -> {d['stage1_corr_count']} (corr) -> "
    f"{d['stage2_boruta_count']} (boruta) -> {d['stage3_rfecv_count']} (rfecv)"
)
if d.get("stage4_ga_count"):
    line += f" -> {d['stage4_ga_count']} (GA)"
print(line)
print(f"Final: {d['final_count']} features")
