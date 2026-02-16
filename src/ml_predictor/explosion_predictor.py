"""
Explosion Predictor - HYBRID MODEL VERSION
Works with models that know BOTH:
- T-3, T-5, T-10 (flat features from CSV: Close, RSI_14, MACD)
- T-1 open/close (prefixed features from database: t1_open_rsi, t1_close_macd)
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class ExplosionPredictor:
    """
    Hybrid explosion predictor
    Works with models that expect BOTH old CSV features AND new T-1 split features
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(model_dir)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scaler"""
        try:
            model_path = self.model_dir / "best_model.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                raise FileNotFoundError(f"Model files not found in {self.model_dir}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            self.logger.info("✓ Loaded model and scaler")
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('features', [])
                    
                    self.logger.info(f"✓ Model expects {len(self.feature_names)} features")
                    
                    # Show what model knows
                    has_t1_features = any('t1_open' in f or 't1_close' in f for f in self.feature_names)
                    has_flat_features = any(f in ['Close', 'RSI_14', 'MACD_12_26_9'] for f in self.feature_names)
                    
                    if has_flat_features and has_t1_features:
                        self.logger.info("✓ Model type: HYBRID (knows T-3/T-5/T-10 + T-1 open/close)")
                    elif has_flat_features:
                        self.logger.info("✓ Model type: CSV-ONLY (knows T-3/T-5/T-10)")
                    elif has_t1_features:
                        self.logger.info("✓ Model type: DATABASE-ONLY (knows T-1 open/close)")
                    
            else:
                # Infer from scaler
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_names = list(self.scaler.feature_names_in_)
                else:
                    n_features = self.scaler.n_features_in_
                    self.feature_names = [f'feature_{i}' for i in range(n_features)]
                
                self.logger.warning(f"No metadata - inferred {len(self.feature_names)} features")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Input data should have flat features from T-3 (fetched from yfinance):
        - Close, High, Low, Open, Volume
        - RSI_14, RSI_7, RSI_21, RSI_28
        - MACD_12_26_9, MACDh_12_26_9, etc.
        
        This function maps them to whatever the model expects
        """
        
        self.logger.info(f"Preparing features for {len(data_df)} stocks")
        
        # Create feature DataFrame with all expected features
        feature_df_final = pd.DataFrame(index=data_df.index)
        
        # Preserve metadata
        for col in ['symbol', 'exchange']:
            if col in data_df.columns:
                feature_df_final[col] = data_df[col]
        
        # Add all expected features
        matched = 0
        missing = 0
        
        for feature in self.feature_names:
            if feature in data_df.columns:
                # Direct match
                feature_df_final[feature] = data_df[feature]
                matched += 1
            else:
                # Feature missing - use intelligent default
                feature_df_final[feature] = self._get_default_value(feature, data_df)
                missing += 1
        
        # Fill NaN
        for col in feature_df_final.columns:
            if col not in ['symbol', 'exchange']:
                feature_df_final[col] = feature_df_final[col].fillna(0)
        
        # Log coverage
        coverage = (matched / len(self.feature_names)) * 100 if self.feature_names else 0
        
        self.logger.info(f"Feature coverage: {coverage:.1f}% ({matched}/{len(self.feature_names)})")
        
        if missing > 0:
            self.logger.debug(f"Missing {missing} features - using defaults")
        
        if coverage < 50:
            self.logger.warning(f"⚠️  LOW feature coverage ({coverage:.1f}%) - predictions may be unreliable")
        
        return feature_df_final
    
    def _get_default_value(self, feature: str, data: pd.DataFrame) -> float:
        """Get intelligent default for missing feature"""
        
        feature_lower = feature.lower()
        
        # Strip timepoint prefix for analysis
        base_feature = feature_lower
        for prefix in ['t1_open_', 't1_close_']:
            if feature_lower.startswith(prefix):
                base_feature = feature_lower[len(prefix):]
                break
        
        # Normalized indicators (0-100)
        if any(ind in base_feature for ind in ['rsi', 'stoch', 'w.r', 'cci', 'willr']):
            return 50.0
        
        # Percentages
        if any(ind in base_feature for ind in ['change', 'pct', '%', 'ratio']):
            return 0.0
        
        # Booleans
        if any(word in base_feature for word in ['above', 'below', 'cross', 'flag']):
            return 0.0
        
        # Volume
        if 'volume' in base_feature:
            for col in data.columns:
                if 'volume' in col.lower() and col not in ['symbol', 'exchange']:
                    return data[col].median()
            return 100000.0
        
        # Price
        if any(word in base_feature for word in ['price', 'close', 'open', 'high', 'low']):
            for col in data.columns:
                if 'close' in col.lower() and col not in ['symbol', 'exchange']:
                    return data[col].median()
            return 50.0
        
        # Oscillators
        if any(ind in base_feature for ind in ['macd', 'ao', 'roc', 'mom']):
            return 0.0
        
        # Moving averages
        if any(ind in base_feature for ind in ['ema', 'sma', 'wma', 'vwap', 'hma']):
            for col in data.columns:
                if 'close' in col.lower():
                    return data[col].median()
            return 50.0
        
        # Volatility
        if any(ind in base_feature for ind in ['atr', 'volatility', 'hv', 'bb']):
            return 1.0
        
        # Default
        return 0.0
    
    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on data
        
        Args:
            data_df: Input data with flat features from T-3 (from yfinance)
        
        Returns:
            DataFrame with predictions
        """
        
        # Prepare features
        features_df = self.prepare_features(data_df)
        
        # Extract only feature columns for prediction
        metadata_cols = ['symbol', 'exchange']
        X = features_df[self.feature_names].copy()
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create result
        result_df = pd.DataFrame({
            'explosion_probability': probabilities,
            'prediction': predictions,
            'signal': pd.Series(probabilities).apply(self._classify_signal)
        })
        
        # Add metadata
        for col in metadata_cols:
            if col in features_df.columns:
                result_df.insert(0, col, features_df[col].values)
        
        # Sort by probability
        result_df = result_df.sort_values('explosion_probability', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def predict_with_targets(
        self,
        data_df: pd.DataFrame,
        historical_gains_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Make predictions with target gain estimates
        """
        
        # Get base predictions
        predictions = self.predict(data_df)
        
        # Prepare features to get current price
        features_df = self.prepare_features(data_df)
        
        # Estimate target gains
        if historical_gains_df is not None and not historical_gains_df.empty:
            # Use historical calibration
            gain_buckets = historical_gains_df.copy()
            gain_buckets['prob_bucket'] = pd.cut(
                gain_buckets['predicted_probability'],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            avg_gains = gain_buckets.groupby('prob_bucket')['actual_gain_pct'].agg([
                'mean', 'median', 'std'
            ])
            
            predictions['prob_bucket'] = pd.cut(
                predictions['explosion_probability'],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            predictions = predictions.merge(avg_gains, left_on='prob_bucket', right_index=True, how='left')
            
            predictions['target_gain_pct'] = predictions['median']
            predictions['target_gain_low'] = predictions['median'] - predictions['std']
            predictions['target_gain_high'] = predictions['median'] + predictions['std']
            
            predictions = predictions.drop(['prob_bucket', 'mean', 'median', 'std'], axis=1)
        else:
            # Use rule-based estimates
            predictions['target_gain_pct'] = predictions['explosion_probability'].apply(self._estimate_target_gain)
            predictions['target_gain_low'] = predictions['target_gain_pct'] * 0.5
            predictions['target_gain_high'] = predictions['target_gain_pct'] * 1.5
        
        # Fill NaN gains
        mask = predictions['target_gain_pct'].isna()
        if mask.any():
            predictions.loc[mask, 'target_gain_pct'] = predictions.loc[mask, 'explosion_probability'].apply(self._estimate_target_gain)
            predictions.loc[mask, 'target_gain_low'] = predictions.loc[mask, 'target_gain_pct'] * 0.5
            predictions.loc[mask, 'target_gain_high'] = predictions.loc[mask, 'target_gain_pct'] * 1.5
        
        # Add target prices - find close price
        if 'symbol' in predictions.columns:
            close_col = None
            for col in features_df.columns:
                if col == 'Close' or col == 'close':
                    close_col = col
                    break
            
            if close_col:
                price_df = features_df[['symbol', close_col]].copy()
                price_df.columns = ['symbol', 'close']
                predictions = predictions.merge(price_df, on='symbol', how='left')
                predictions['current_price'] = predictions['close']
                predictions['target_price'] = predictions['close'] * (1 + predictions['target_gain_pct'] / 100)
                predictions['target_price_low'] = predictions['close'] * (1 + predictions['target_gain_low'] / 100)
                predictions['target_price_high'] = predictions['close'] * (1 + predictions['target_gain_high'] / 100)
                predictions = predictions.drop('close', axis=1)
        
        return predictions
    
    def _classify_signal(self, probability: float) -> str:
        """Classify prediction into signal"""
        if probability >= 0.90:
            return "STRONG BUY"
        elif probability >= 0.70:
            return "BUY"
        elif probability >= 0.50:
            return "HOLD"
        else:
            return "AVOID"
    
    def _estimate_target_gain(self, probability: float) -> float:
        """Rule-based target gain"""
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
