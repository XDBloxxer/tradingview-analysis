"""
migrate_model_pkl.py
--------------------
One-time migration script: re-saves best_model.pkl (and optionally
gain_regressor.pkl / scaler.pkl) so that _PriorCorrectedModel is stored
under its new canonical module path:

    src.ml_predictor.prior_corrected_model._PriorCorrectedModel

instead of the old:

    __main__._PriorCorrectedModel

Run once after deploying the fix:

    python migrate_model_pkl.py

The script is safe to re-run; it overwrites the pkl in-place only if the
migration is needed.
"""

import sys
import io
import joblib
import pickle
from pathlib import Path

# ── Ensure the project root is on sys.path ──────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Import the shared class so pickle can resolve it ────────────────────────
from src.ml_predictor.prior_corrected_model import _PriorCorrectedModel  # noqa: F401


# ── Unpickle helper that maps __main__._PriorCorrectedModel to the real one ─

class _Remapper(pickle.Unpickler):
    """Custom unpickler that re-routes the old __main__ reference."""

    def find_class(self, module, name):
        if name == "_PriorCorrectedModel":
            # Always use the canonical location regardless of saved module path
            from src.ml_predictor.prior_corrected_model import _PriorCorrectedModel
            return _PriorCorrectedModel
        return super().find_class(module, name)


def _load_with_remap(path: Path):
    """Load a joblib/pickle file, remapping _PriorCorrectedModel if needed."""
    # joblib files are typically a single pickle stream; try direct unpickle
    # first (fast path), then fall back to joblib for compressed formats.
    with open(path, "rb") as f:
        data = f.read()

    try:
        obj = _Remapper(io.BytesIO(data)).load()
        return obj
    except Exception:
        pass

    # Fallback: joblib may use multiple pickle streams (compression, memmaps).
    # Patch pickle._Unpickler temporarily.
    import pickle as _pickle
    _orig = _pickle.Unpickler

    class _PatchedUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_PriorCorrectedModel":
                from src.ml_predictor.prior_corrected_model import _PriorCorrectedModel
                return _PriorCorrectedModel
            return super().find_class(module, name)

    _pickle.Unpickler = _PatchedUnpickler
    try:
        obj = joblib.load(path)
    finally:
        _pickle.Unpickler = _orig

    return obj


def migrate(pkl_path: Path) -> bool:
    """
    Load the pkl, check if migration is needed, re-save if so.
    Returns True if the file was migrated.
    """
    if not pkl_path.exists():
        print(f"  SKIP  {pkl_path.name} — file not found (likely a Git LFS stub)")
        return False

    # Quick check: does the raw bytes contain the old __main__ reference?
    raw = pkl_path.read_bytes()
    old_ref = b"__main__\n_PriorCorrectedModel"
    if old_ref not in raw:
        print(f"  OK    {pkl_path.name} — no migration needed")
        return False

    print(f"  MIGRATING {pkl_path.name} …", end=" ", flush=True)
    obj = _load_with_remap(pkl_path)
    joblib.dump(obj, pkl_path, protocol=4)
    # Verify the old reference is gone
    new_raw = pkl_path.read_bytes()
    if old_ref in new_raw:
        print("FAILED (old reference still present)")
        return False
    print("done")
    return True


if __name__ == "__main__":
    model_dir = ROOT / "ml_models"
    targets = [
        model_dir / "best_model.pkl",
        model_dir / "gain_regressor.pkl",
        model_dir / "scaler.pkl",
    ]

    print("Migrating pkl files in", model_dir)
    migrated = 0
    for p in targets:
        if migrate(p):
            migrated += 1

    print(f"\n{migrated} file(s) migrated.")
    if migrated == 0:
        print("Nothing to do — either files are Git LFS stubs (download them first)")
        print("or they were already saved with the correct module path.")
