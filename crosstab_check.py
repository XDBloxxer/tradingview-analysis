"""
One-off check: does the 5min/daily_fallback ratio differ by label?
Run with the same env/credentials as validate_symbol_demeaning.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt

client = rt.get_supabase_client()
t1_df = rt.load_t1_data(client, lookback_days=365)

props = t1_df.groupby("label")["t1_data_source"].value_counts(normalize=True).unstack()
counts = t1_df.groupby("label")["t1_data_source"].value_counts().unstack()

print("=== Proportions (5min vs daily_fallback) by label ===")
print(props)
print()
print("=== Raw counts by label ===")
print(counts)
print()

# ── Plain-English verdict ────────────────────────────────────────────────
if "5min" in props.columns and 0 in props.index and 1 in props.index:
    p0 = props.loc[0, "5min"]
    p1 = props.loc[1, "5min"]
    gap = abs(p0 - p1)
    print(f"5min share: non-winners={p0:.1%}, winners={p1:.1%}, gap={gap:.1%}")
    print()
    if gap < 0.03:
        print("VERDICT: Gap is small (<3pp). t1_data_source is NOT meaningfully")
        print("skewed by label -- this is not your leak. Look elsewhere (e.g. the")
        print("daily_fallback feature computation itself, independent of label mix).")
    elif gap < 0.08:
        print("VERDICT: Moderate gap (3-8pp). Worth a closer look, but not yet")
        print("conclusive on its own -- could still be secondary to something else.")
    else:
        print("VERDICT: Large gap (>8pp). t1_data_source IS meaningfully skewed by")
        print("label -- this is a real leak candidate. Winners and non-winners are")
        print("systematically differing in which T-1 data-collection path they took,")
        print("which the model can partially learn as a label proxy.")
else:
    print("VERDICT: Could not compute a clean comparison (missing label or source "
          "category) -- inspect the raw tables above manually.")

