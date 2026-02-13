"""
Model Trainer - SELF-LEARNING SYSTEM
Retrains on actual outcomes, learns from ALL available data
"""

import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb


class ModelTrainer:
    """
    Trains and updates the explosion prediction model
    Uses ACTUAL market outcomes to improve predictions
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Archive directory for old models
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def prepare_training_data_from_daily_winners(
        self,
        supabase_client,
        lookback_days: int = 90,
        use_all_timepoints: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Prepare training data from Daily Winners system
        
        Strategy:
        1. Get all day_prior_close data (T-1 indicators) - PRIMARY SOURCE
        2. Optionally add day_prior_open, market_open for MORE DATA
        3. Label stocks that became winners next day as 1, others as 0
        4. Track actual gains for calibration
        
        This uses ALL available historical data, not just winners
        """
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        client = supabase_client.client
        
        # ===== STEP 1: Get T-1 close data (PRIMARY) =====
        self.logger.info("Loading day_prior_close data...")
        
        response = client.table("winners_day_prior_close")\
            .select("*")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        if not response.data:
            raise ValueError("No day_prior_close data available")
        
        day_prior_close_df = pd.DataFrame(response.data)
        self.logger.info(f"  Loaded {len(day_prior_close_df)} T-1 close records")
        
        # ===== STEP 2: Get actual winners for labeling =====
        self.logger.info("Loading actual winners for labeling...")
        
        response = client.table("daily_winners")\
            .select("symbol,detection_date,change_pct,price")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        if not response.data:
            raise ValueError("No winner data available")
        
        winners_df = pd.DataFrame(response.data)
        self.logger.info(f"  Loaded {len(winners_df)} actual winner records")
        
        # ===== STEP 3: Create training samples =====
        self.logger.info("Creating labeled training samples...")
        
        training_samples = []
        
        # Process day_prior_close (best data)
        for _, row in day_prior_close_df.iterrows():
            symbol = row['symbol']
            detection_date = row['detection_date']
            
            # Check if this stock became a winner
            winner_match = winners_df[
                (winners_df['symbol'] == symbol) & 
                (winners_df['detection_date'] == detection_date)
            ]
            
            is_winner = len(winner_match) > 0
            
            # Get actual gain if winner
            if is_winner:
                actual_gain = winner_match.iloc[0]['change_pct']
                actual_price = winner_match.iloc[0]['price']
            else:
                actual_gain = 0  # Didn't explode
                actual_price = row.get('close', 0)
            
            # Create training sample
            sample = {
                'symbol': symbol,
                'detection_date': detection_date,
                'label': 1 if is_winner else 0,
                'actual_gain_pct': actual_gain,
                'actual_price': actual_price,
                'timepoint': 'day_prior_close'
            }
            
            # Add all indicator features
            for col in row.index:
                if col not in ['symbol', 'detection_date', 'exchange', 'snapshot_type',
                              'snapshot_time', 'snapshot_date', 'id', 'created_at']:
                    sample[col] = row[col]
            
            training_samples.append(sample)
        
        # ===== OPTIONAL: Add more timepoints for additional data =====
        if use_all_timepoints:
            self.logger.info("Adding day_prior_open data for more training samples...")
            
            response = client.table("winners_day_prior_open")\
                .select("*")\
                .gte("detection_date", start_date.isoformat())\
                .lte("detection_date", end_date.isoformat())\
                .execute()
            
            if response.data:
                day_prior_open_df = pd.DataFrame(response.data)
                self.logger.info(f"  Loaded {len(day_prior_open_df)} T-1 open records")
                
                for _, row in day_prior_open_df.iterrows():
                    symbol = row['symbol']
                    detection_date = row['detection_date']
                    
                    # Skip if we already have this stock from day_prior_close
                    if any(s['symbol'] == symbol and s['detection_date'] == detection_date 
                          and s['timepoint'] == 'day_prior_close' for s in training_samples):
                        continue
                    
                    winner_match = winners_df[
                        (winners_df['symbol'] == symbol) & 
                        (winners_df['detection_date'] == detection_date)
                    ]
                    
                    is_winner = len(winner_match) > 0
                    
                    if is_winner:
                        actual_gain = winner_match.iloc[0]['change_pct']
                        actual_price = winner_match.iloc[0]['price']
                    else:
                        actual_gain = 0
                        actual_price = row.get('close', 0)
                    
                    sample = {
                        'symbol': symbol,
                        'detection_date': detection_date,
                        'label': 1 if is_winner else 0,
                        'actual_gain_pct': actual_gain,
                        'actual_price': actual_price,
                        'timepoint': 'day_prior_open'
                    }
                    
                    for col in row.index:
                        if col not in ['symbol', 'detection_date', 'exchange', 'snapshot_type',
                                      'snapshot_time', 'snapshot_date', 'id', 'created_at']:
                            sample[col] = row[col]
                    
                    training_samples.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_samples)
        
        self.logger.info(f"Created {len(df)} total training samples")
        self.logger.info(f"  - Positives (explosions): {df['label'].sum()}")
        self.logger.info(f"  - Negatives (no explosion): {len(df) - df['label'].sum()}")
        self.logger.info(f"  - Positive rate: {df['label'].mean()*100:.2f}%")
        
        # Separate features and labels
        exclude_cols = ['symbol', 'detection_date', 'label', 'actual_gain_pct', 
                       'actual_price', 'timepoint']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df['label']
        
        # Store metadata
        metadata = {
            'n_samples': len(df),
            'n_positives': int(df['label'].sum()),
            'n_negatives': int(len(df) - df['label'].sum()),
            'positive_rate': float(df['label'].mean()),
            'date_range': f"{start_date} to {end_date}",
            'feature_count': len(feature_cols),
            'features': feature_cols,
            'trained_at': datetime.now().isoformat(),
            'timepoints_used': list(df['timepoint'].unique())
        }
        
        return X, y, metadata
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_time_series_split: bool = True,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train XGBoost model with proper validation
        
        Args:
            X: Features
            y: Labels
            use_time_series_split: Use time-series aware split
            test_size: Test set size
        
        Returns:
            Dict with model, scaler, and metrics
        """
        
        self.logger.info("Training XGBoost model...")
        self.logger.info(f"  Training samples: {len(X)}")
        self.logger.info(f"  Features: {len(X.columns)}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        if use_time_series_split:
            # Time-series split (no random shuffling)
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            self.logger.info("Using time-series split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            self.logger.info("Using stratified random split")
        
        self.logger.info(f"  Train set: {len(X_train)} samples ({y_train.sum()} positive)")
        self.logger.info(f"  Test set: {len(X_test)} samples ({y_test.sum()} positive)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight for imbalanced data
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.logger.info(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Detailed metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
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
        
        self.logger.info(f"\nTraining complete:")
        self.logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"  Train AUC: {train_auc:.4f}")
        self.logger.info(f"  Test AUC: {test_auc:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1 Score: {f1:.4f}")
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"  TP: {tp}, FP: {fp}")
        self.logger.info(f"  FN: {fn}, TN: {tn}")
        
        return results
    
    def save_model(self, model, scaler, metadata: Dict, version: str = None):
        """Save model with versioning and metadata"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archive old model if exists
        old_model_path = self.model_dir / "best_model.pkl"
        old_scaler_path = self.model_dir / "scaler.pkl"
        
        if old_model_path.exists():
            archive_model = self.archive_dir / f"best_model_{version}.pkl"
            old_model_path.rename(archive_model)
            self.logger.info(f"Archived old model to {archive_model}")
        
        if old_scaler_path.exists():
            archive_scaler = self.archive_dir / f"scaler_{version}.pkl"
            old_scaler_path.rename(archive_scaler)
        
        # Save new model
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        metadata['model_version'] = version
        
        metadata_path = self.model_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✓ Saved model: {model_path}")
        self.logger.info(f"✓ Saved scaler: {scaler_path}")
        self.logger.info(f"✓ Saved metadata: {metadata_path}")
    
    def calculate_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Calculate and save feature importance"""
        
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_path = self.model_dir / "feature_importance.csv"
        df.to_csv(importance_path, index=False)
        
        self.logger.info(f"✓ Saved feature importance: {importance_path}")
        
        return df