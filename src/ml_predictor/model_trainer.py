"""
Model Trainer - MULTI-TIMEPOINT VERSION
Trains model on ALL timepoint data with prefixes preserved
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb


class ModelTrainer:
    """
    Trains multi-timepoint explosion prediction model
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def prepare_multi_timepoint_training_data(
        self,
        supabase_client,
        lookback_days: int = 90,
        include_non_winners: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Prepare MULTI-TIMEPOINT training data from Daily Winners system
        
        Strategy:
        1. Fetch ALL timepoints (day_prior_open, day_prior_close for winners AND non-winners)
        2. Create features WITH PREFIXES (t1_open_rsi, t1_close_macd, etc.)
        3. Model learns which timepoints are most predictive
        """
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info("="*80)
        self.logger.info("PREPARING MULTI-TIMEPOINT TRAINING DATA")
        self.logger.info("="*80)
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Include non-winners: {include_non_winners}")
        
        client = supabase_client.client
        
        # Fetch winners T-1 close
        self.logger.info("\nFetching winners T-1 close...")
        response = client.table("winners_day_prior_close")\
            .select("*")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        winners_t1_close = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        self.logger.info(f"  Loaded {len(winners_t1_close)} winner T-1 close records")
        
        # Fetch winners T-1 open
        self.logger.info("Fetching winners T-1 open...")
        response = client.table("winners_day_prior_open")\
            .select("*")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        winners_t1_open = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        self.logger.info(f"  Loaded {len(winners_t1_open)} winner T-1 open records")
        
        # Fetch non-winners if enabled
        non_winners_t1_close = pd.DataFrame()
        non_winners_t1_open = pd.DataFrame()
        
        if include_non_winners:
            self.logger.info("Fetching non-winners T-1 close...")
            try:
                response = client.table("non_winners_day_prior_close")\
                    .select("*")\
                    .gte("detection_date", start_date.isoformat())\
                    .lte("detection_date", end_date.isoformat())\
                    .execute()
                
                non_winners_t1_close = pd.DataFrame(response.data) if response.data else pd.DataFrame()
                self.logger.info(f"  Loaded {len(non_winners_t1_close)} non-winner T-1 close records")
            except Exception as e:
                self.logger.warning(f"  Could not load non-winners close: {e}")
            
            self.logger.info("Fetching non-winners T-1 open...")
            try:
                response = client.table("non_winners_day_prior_open")\
                    .select("*")\
                    .gte("detection_date", start_date.isoformat())\
                    .lte("detection_date", end_date.isoformat())\
                    .execute()
                
                non_winners_t1_open = pd.DataFrame(response.data) if response.data else pd.DataFrame()
                self.logger.info(f"  Loaded {len(non_winners_t1_open)} non-winner T-1 open records")
            except Exception as e:
                self.logger.warning(f"  Could not load non-winners open: {e}")
        
        # Get actual winners for labeling
        self.logger.info("\nFetching actual winners for labeling...")
        response = client.table("daily_winners")\
            .select("symbol,detection_date,change_pct")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        actual_winners_df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        self.logger.info(f"  Loaded {len(actual_winners_df)} actual winner records")
        
        # Create multi-timepoint training samples
        self.logger.info("\n" + "="*80)
        self.logger.info("CREATING MULTI-TIMEPOINT FEATURES")
        self.logger.info("="*80)
        
        training_samples = []
        
        # Merge T-1 close and T-1 open data by symbol + detection_date
        def merge_timepoints(t1_close_df, t1_open_df, is_winner_data=True):
            """Merge close and open timepoints"""
            
            merged_samples = []
            
            for _, close_row in t1_close_df.iterrows():
                symbol = close_row['symbol']
                detection_date = close_row['detection_date']
                
                # Find matching open row
                open_row = t1_open_df[
                    (t1_open_df['symbol'] == symbol) & 
                    (t1_open_df['detection_date'] == detection_date)
                ]
                
                # Check if winner
                is_winner = len(actual_winners_df[
                    (actual_winners_df['symbol'] == symbol) & 
                    (actual_winners_df['detection_date'] == detection_date)
                ]) > 0
                
                # Create sample with BOTH timepoints
                sample = {
                    'symbol': symbol,
                    'detection_date': detection_date,
                    'label': 1 if is_winner else 0
                }
                
                # Add T-1 close features with prefix
                for col in close_row.index:
                    if col not in ['symbol', 'detection_date', 'exchange', 'snapshot_type', 
                                   'snapshot_time', 'snapshot_date', 'id', 'created_at']:
                        sample[f't1_close_{col}'] = close_row[col]
                
                # Add T-1 open features with prefix (if available)
                if not open_row.empty:
                    open_data = open_row.iloc[0]
                    for col in open_data.index:
                        if col not in ['symbol', 'detection_date', 'exchange', 'snapshot_type',
                                       'snapshot_time', 'snapshot_date', 'id', 'created_at']:
                            sample[f't1_open_{col}'] = open_data[col]
                
                merged_samples.append(sample)
            
            return merged_samples
        
        # Merge winners
        if not winners_t1_close.empty:
            winner_samples = merge_timepoints(winners_t1_close, winners_t1_open, is_winner_data=True)
            training_samples.extend(winner_samples)
            self.logger.info(f"  Created {len(winner_samples)} winner samples")
        
        # Merge non-winners
        if include_non_winners and not non_winners_t1_close.empty:
            non_winner_samples = merge_timepoints(non_winners_t1_close, non_winners_t1_open, is_winner_data=False)
            training_samples.extend(non_winner_samples)
            self.logger.info(f"  Created {len(non_winner_samples)} non-winner samples")
        
        if not training_samples:
            self.logger.error("No training samples created!")
            return pd.DataFrame(), pd.Series(), {}
        
        # Convert to DataFrame
        df = pd.DataFrame(training_samples)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING DATA SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total samples: {len(df)}")
        self.logger.info(f"  Positives (winners): {df['label'].sum()}")
        self.logger.info(f"  Negatives (non-winners): {len(df) - df['label'].sum()}")
        self.logger.info(f"  Positive rate: {df['label'].mean()*100:.2f}%")
        
        # Separate features and labels
        exclude_cols = ['symbol', 'detection_date', 'label']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self.logger.info(f"Total features: {len(feature_cols)}")
        
        # Count features per timepoint
        t1_close_features = sum(1 for f in feature_cols if f.startswith('t1_close_'))
        t1_open_features = sum(1 for f in feature_cols if f.startswith('t1_open_'))
        
        self.logger.info(f"  T-1 close features: {t1_close_features}")
        self.logger.info(f"  T-1 open features: {t1_open_features}")
        
        X = df[feature_cols]
        y = df['label']
        
        metadata = {
            'n_samples': len(df),
            'n_positives': int(df['label'].sum()),
            'n_negatives': int(len(df) - df['label'].sum()),
            'positive_rate': float(df['label'].mean()),
            'date_range': f"{start_date} to {end_date}",
            'feature_count': len(feature_cols),
            'features': feature_cols,
            'timepoints': ['t1_close', 't1_open'],
            'is_multi_timepoint': True,
            'includes_negative_examples': include_non_winners
        }
        
        return X, y, metadata
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray = None,
        test_size: float = 0.2
    ) -> Dict:
        """Train XGBoost model"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING MULTI-TIMEPOINT XGBOOST MODEL")
        self.logger.info("="*80)
        
        X = X.fillna(0)
        
        # Time-series split
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        if sample_weights is not None:
            weights_train = sample_weights[:split_idx]
        else:
            weights_train = None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Test samples: {len(X_test)}")
        self.logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        fit_params = {
            'eval_set': [(X_train_scaled, y_train), (X_test_scaled, y_test)],
            'verbose': False
        }
        
        if weights_train is not None:
            fit_params['sample_weight'] = weights_train
        
        model.fit(X_train_scaled, y_train, **fit_params)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING RESULTS")
        self.logger.info("="*80)
        self.logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Train AUC: {train_auc:.4f}")
        self.logger.info(f"Test AUC: {test_auc:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        results = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        return results
    
    def save_model(self, model, scaler, metadata: Dict, version: str = None):
        """Save model with multi-timepoint metadata"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archive old model
        old_model_path = self.model_dir / "best_model.pkl"
        
        if old_model_path.exists():
            archive_model = self.archive_dir / f"best_model_{version}.pkl"
            old_model_path.rename(archive_model)
        
        # Save new model
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metadata with multi-timepoint flag
        metadata['model_version'] = version
        metadata['saved_with'] = 'joblib'
        metadata['model_type'] = 'multi_timepoint'
        
        metadata_path = self.model_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✓ Saved multi-timepoint model: {model_path}")
        self.logger.info(f"✓ Model expects {len(metadata.get('features', []))} features across {len(metadata.get('timepoints', []))} timepoints")
    
    def calculate_feature_importance(self, model, feature_names) -> pd.DataFrame:
        """Calculate feature importance"""
        
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        importance_path = self.model_dir / "feature_importance.csv"
        df.to_csv(importance_path, index=False)
        
        # Analyze by timepoint
        self.logger.info("\nFeature Importance by Timepoint:")
        for timepoint in ['t1_close_', 't1_open_']:
            tp_features = df[df['feature'].str.startswith(timepoint)]
            if not tp_features.empty:
                total_importance = tp_features['importance'].sum()
                self.logger.info(f"  {timepoint[:-1]}: {total_importance:.4f}")
        
        return df
