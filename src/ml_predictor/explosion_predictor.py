"""
Explosion Predictor - FIXED FOR JOBLIB
Handles missing features, multiple timepoints, target gain estimation
"""

import logging
import joblib  # ← FIXED: Changed from pickle to joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .feature_mapper import FeatureMapper


class ExplosionPredictor:
    """
    Smart explosion predictor with adaptive feature mapping
    Handles all available timepoints intelligently
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(model_dir)
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        self._load_model()
        
        # Initialize feature mapper
        self.feature_mapper = FeatureMapper()
    
    def _load_model(self):
        """Load trained model and scaler - FIXED for joblib"""
        try:
            model_path = self.model_dir / "best_model.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                raise FileNotFoundError(f"Model files not found in {self.model_dir}")
            
            # FIXED: Use joblib.load instead of pickle.load
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            self.logger.info("✓ Loaded model and scaler using joblib")
            
            # Try to load metadata (optional - not critical)
            try:
                metadata_path = self.model_dir / "model_metadata.json"
                if metadata_path.exists():
                    try:
                        import json
                        with open(metadata_path, 'r') as f:
                            self.metadata = json.load(f)
                            self.feature_names = self.metadata.get('features', [])
                        self.logger.info("✓ Loaded metadata from JSON")
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata JSON: {e}")
                        self.metadata = None
                
                # If no metadata, try to infer from scaler
                if not self.feature_names and hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_names = list(self.scaler.feature_names_in_)
                    self.logger.info("✓ Inferred feature names from scaler")
                
                if not self.feature_names:
                    # Fall back to generic feature count
                    n_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 97
                    self.feature_names = [f'feature_{i}' for i in range(n_features)]
                    self.logger.warning(f"Using generic feature names for {n_features} features")
                
                self.logger.info(f"✓ Ready with {len(self.feature_names)} features")
                
            except Exception as e:
                self.logger.error(f"Failed to setup features: {e}")
                raise
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_features_from_daily_winners(
        self,
        daily_winners_data: pd.DataFrame,
        timepoint: str = 'day_prior_close'
    ) -> pd.DataFrame:
        """
        Prepare features from Daily Winners data
        ADAPTIVE: Works with whatever indicators are available
        
        Args:
            daily_winners_data: DataFrame from winners_day_prior_close, etc.
            timepoint: Which timepoint this data represents
        
        Returns:
            DataFrame ready for prediction
        """
        
        self.logger.info(f"Preparing features from {timepoint} data")
        self.logger.info(f"Input data shape: {daily_winners_data.shape}")
        
        # Map available indicators to expected features
        features_df, mapping_report = self.feature_mapper.map_features(
            daily_winners_data,
            self.feature_names
        )
        
        # Add symbol for tracking
        if 'symbol' in daily_winners_data.columns:
            features_df.insert(0, 'symbol', daily_winners_data['symbol'].values)
        
        # Add metadata columns for reference
        metadata_cols = ['exchange', 'detection_date']
        for col in metadata_cols:
            if col in daily_winners_data.columns:
                features_df[col] = daily_winners_data[col].values
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        # Log feature coverage
        coverage_report = self.feature_mapper.get_feature_coverage_report(
            features_df, 
            self.feature_names
        )
        
        self.logger.info(f"Feature coverage: {coverage_report['coverage_pct']:.1f}%")
        self.logger.info(f"  - Features found: {coverage_report['features_found']}")
        self.logger.info(f"  - Features missing: {coverage_report['features_missing']}")
        
        if coverage_report['missing_features']:
            self.logger.warning(f"Missing {len(coverage_report['missing_features'])} features:")
            for missing in coverage_report['missing_features'][:5]:
                self.logger.warning(f"    - {missing}")
            if len(coverage_report['missing_features']) > 5:
                self.logger.warning(f"    ... and {len(coverage_report['missing_features']) - 5} more")
        
        return features_df
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make basic predictions
        
        Returns:
            DataFrame with symbol, explosion_probability, prediction, signal
        """
        
        # Extract feature columns only (remove metadata)
        metadata_cols = ['symbol', 'exchange', 'detection_date']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        # Ensure we have all expected features
        X = features_df[self.feature_names].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'explosion_probability': probabilities,
            'prediction': predictions
        })
        
        # Add symbol if available
        if 'symbol' in features_df.columns:
            result_df.insert(0, 'symbol', features_df['symbol'].values)
        
        # Add signal classification
        result_df['signal'] = result_df['explosion_probability'].apply(self._classify_signal)
        
        # Sort by probability
        result_df = result_df.sort_values('explosion_probability', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def predict_with_targets(
        self,
        features_df: pd.DataFrame,
        historical_gains_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Make predictions with target gain estimates
        
        Args:
            features_df: Feature DataFrame with current_price if available
            historical_gains_df: Historical actual gains for calibration
        
        Returns:
            DataFrame with predictions + target gains
        """
        
        # Get base predictions
        predictions = self.predict(features_df)
        
        # Estimate target gains based on historical data
        if historical_gains_df is not None and not historical_gains_df.empty:
            # Calculate average gain for each probability bucket
            gain_buckets = historical_gains_df.copy()
            gain_buckets['prob_bucket'] = pd.cut(
                gain_buckets['probability'],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            avg_gains_by_bucket = gain_buckets.groupby('prob_bucket')['actual_gain_pct'].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ])
            
            # Map predictions to gain estimates
            predictions['prob_bucket'] = pd.cut(
                predictions['explosion_probability'],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            predictions = predictions.merge(
                avg_gains_by_bucket,
                left_on='prob_bucket',
                right_index=True,
                how='left'
            )
            
            # Use median as target, std for range
            predictions['target_gain_pct'] = predictions['median']
            predictions['target_gain_low'] = predictions['median'] - predictions['std']
            predictions['target_gain_high'] = predictions['median'] + predictions['std']
            
            # Clean up
            predictions = predictions.drop(['prob_bucket', 'mean', 'median', 'std', 'min', 'max', 'count'], axis=1)
            
        else:
            # Use rule-based estimates
            predictions['target_gain_pct'] = predictions['explosion_probability'].apply(
                self._estimate_target_gain
            )
            predictions['target_gain_low'] = predictions['target_gain_pct'] * 0.5
            predictions['target_gain_high'] = predictions['target_gain_pct'] * 1.5
        
        # Fill NaN target gains with rule-based estimates
        mask = predictions['target_gain_pct'].isna()
        if mask.any():
            predictions.loc[mask, 'target_gain_pct'] = predictions.loc[mask, 'explosion_probability'].apply(
                self._estimate_target_gain
            )
            predictions.loc[mask, 'target_gain_low'] = predictions.loc[mask, 'target_gain_pct'] * 0.5
            predictions.loc[mask, 'target_gain_high'] = predictions.loc[mask, 'target_gain_pct'] * 1.5
        
        # Calculate target price if current price available
        if 'close' in features_df.columns:
            price_df = features_df[['symbol', 'close']].copy() if 'symbol' in features_df.columns else pd.DataFrame()
            
            if not price_df.empty:
                predictions = predictions.merge(price_df, on='symbol', how='left')
                predictions['current_price'] = predictions['close']
                predictions['target_price'] = predictions['close'] * (1 + predictions['target_gain_pct'] / 100)
                predictions['target_price_low'] = predictions['close'] * (1 + predictions['target_gain_low'] / 100)
                predictions['target_price_high'] = predictions['close'] * (1 + predictions['target_gain_high'] / 100)
                predictions = predictions.drop('close', axis=1)
        
        return predictions
    
    def _classify_signal(self, probability: float) -> str:
        """Classify prediction into signal categories"""
        if probability >= 0.90:
            return "STRONG BUY"
        elif probability >= 0.70:
            return "BUY"
        elif probability >= 0.50:
            return "HOLD"
        else:
            return "AVOID"
    
    def _estimate_target_gain(self, probability: float) -> float:
        """Rule-based target gain estimation"""
        if probability >= 0.95:
            return 30.0
        elif probability >= 0.90:
            return 25.0
        elif probability >= 0.80:
            return 20.0
        elif probability >= 0.70:
            return 15.0
        elif probability >= 0.60:
            return 10.0
        elif probability >= 0.50:
            return 7.0
        else:
            return 3.0
    
    def predict_multiple_timepoints(
        self,
        day_prior_close: pd.DataFrame = None,
        day_prior_open: pd.DataFrame = None,
        current_open: pd.DataFrame = None,
        current_close: pd.DataFrame = None,
        historical_gains: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Make predictions using ALL available timepoints intelligently
        
        Priority order:
        1. day_prior_close (T-1 4pm) - BEST for prediction, no leakage
        2. day_prior_open (T-1 9:30am) - Good, slight leakage possible
        3. current_open (T 9:30am) - Some leakage, but usable
        4. current_close (T 4pm) - Most leakage, avoid if possible
        
        Strategy: Use best available timepoint
        """
        
        # Determine which timepoint to use
        if day_prior_close is not None and not day_prior_close.empty:
            self.logger.info("Using day_prior_close (T-1 4pm) - BEST timepoint")
            features_df = self.prepare_features_from_daily_winners(
                day_prior_close, 
                'day_prior_close'
            )
        elif day_prior_open is not None and not day_prior_open.empty:
            self.logger.info("Using day_prior_open (T-1 9:30am)")
            features_df = self.prepare_features_from_daily_winners(
                day_prior_open,
                'day_prior_open'
            )
        elif current_open is not None and not current_open.empty:
            self.logger.warning("Using current_open (T 9:30am) - has some leakage")
            features_df = self.prepare_features_from_daily_winners(
                current_open,
                'current_open'
            )
        elif current_close is not None and not current_close.empty:
            self.logger.warning("Using current_close (T 4pm) - WARNING: significant leakage")
            features_df = self.prepare_features_from_daily_winners(
                current_close,
                'current_close'
            )
        else:
            raise ValueError("No data available for prediction")
        
        # Make predictions with target gains
        predictions = self.predict_with_targets(features_df, historical_gains)
        
        return predictions
