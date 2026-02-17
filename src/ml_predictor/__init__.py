"""
ML Predictor Module
Self-learning stock explosion prediction system
"""

from .explosion_predictor import ExplosionPredictor
from .ml_supabase_client import MLPredictionSupabaseClient

__all__ = [
    "ExplosionPredictor",
    "MLPredictionSupabaseClient",
]

# FeatureMapper (feature_mapper.py) is retained in the package for reference
# but is no longer part of the active pipeline — explosion_predictor.py handles
# feature alignment internally.  Import it directly if needed:
#   from src.ml_predictor.feature_mapper import FeatureMapper
