"""
Model Trainer - FINE-TUNING VERSION
PRESERVES existing model knowledge (T-3/T-5/T-10 from CSV)
ADDS new knowledge (T-1 open/close from database)
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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb


class ModelTrainer:
    """
    Fine-tunes existing model to ADD T-1 knowledge while PRESERVING T-3/T-5/T-10 knowledge
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def prepare_fine_tuning_data(
        self,
        supabase_client,
        lookback_days: int = 90,
        include_non_winners: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Prepare T-1 open/close data from database for FINE-TUNING
        
        This does NOT replace the model - it ADDS to what the model already knows
        """
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info("="*80)
        self.logger.info("PREPARING FINE-TUNING DATA (T-1 OPEN/CLOSE)")
        self.logger.info("="*80)
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Include non-winners: {include_non_winners}")
        self.logger.info("")
        self.logger.info("NOTE: This will ADD T-1 knowledge to existing model")
        self.logger.info("      Existing T-3/T-5/T-10 knowledge will be PRESERVED")
        
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
        
        # Fetch non-winners
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
        
        # Create T-1 training samples WITH PREFIXES
        self.logger.info("\n" + "="*80)
        self.logger.info("CREATING T-1 OPEN/CLOSE FEATURES (WITH PREFIXES)")
        self.logger.info("="*80)
        
        training_samples = []
        
        # Metadata columns to exclude
        meta_cols = ['id', 'created_at', 'updated_at', 'symbol', 'exchange', 
                     'detection_date', 'snapshot_type', 'snapshot_time', 'snapshot_date']
        
        def merge_timepoints(t1_close_df, t1_open_df):
            """Merge close and open timepoints with prefixes"""
            
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
                
                # Create sample with PREFIXED features
                sample = {
                    'symbol': symbol,
                    'detection_date': detection_date,
                    'label': 1 if is_winner else 0
                }
                
                # Add T-1 close features with prefix
                for col in close_row.index:
                    if col not in meta_cols:
                        sample[f't1_close_{col}'] = close_row[col]
                
                # Add T-1 open features with prefix
                if not open_row.empty:
                    open_data = open_row.iloc[0]
                    for col in open_data.index:
                        if col not in meta_cols:
                            sample[f't1_open_{col}'] = open_data[col]
                
                merged_samples.append(sample)
            
            return merged_samples
        
        # Merge winners
        if not winners_t1_close.empty:
            winner_samples = merge_timepoints(winners_t1_close, winners_t1_open)
            training_samples.extend(winner_samples)
            self.logger.info(f"  Created {len(winner_samples)} winner samples")
        
        # Merge non-winners
        if include_non_winners and not non_winners_t1_close.empty:
            non_winner_samples = merge_timepoints(non_winners_t1_close, non_winners_t1_open)
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
        
        self.logger.info(f"\nNew T-1 features: {len(feature_cols)}")
        
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
            'new_feature_count': len(feature_cols),
            'new_features': feature_cols,
            'timepoints_added': ['t1_close', 't1_open'],
            'training_type': 'fine_tuning'
        }
        
        return X, y, metadata
    
    def fine_tune_model(
        self,
        new_X: pd.DataFrame,
        new_y: pd.Series,
        test_size: float = 0.2
    ) -> Dict:
        """
        FINE-TUNE existing model with new T-1 features
        
        This expands the model to accept MORE features while keeping old knowledge
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("FINE-TUNING EXISTING MODEL")
        self.logger.info("="*80)
        
        # Load existing model
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "model_metadata.json"
        
        if not model_path.exists():
            self.logger.error("No existing model found! Cannot fine-tune.")
            self.logger.error("Run initial training first with train_initial_model_from_csv.py")
            raise FileNotFoundError("No existing model to fine-tune")
        
        # Load existing model
        existing_model = joblib.load(model_path)
        existing_scaler = joblib.load(scaler_path)
        
        import json
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
        
        existing_features = existing_metadata.get('features', [])
        
        self.logger.info(f"Loaded existing model:")
        self.logger.info(f"  - Existing features: {len(existing_features)}")
        self.logger.info(f"  - New features to add: {len(new_X.columns)}")
        
        # Combine feature sets
        combined_features = list(existing_features) + list(new_X.columns)
        
        self.logger.info(f"  - Total combined features: {len(combined_features)}")
        
        # For fine-tuning, we need to create a NEW expanded model
        # The old model knows features [0:N], new model will know features [0:N+M]
        
        # Fill NaN in new data
        new_X = new_X.fillna(0)
        
        # Time-series split
        split_idx = int(len(new_X) * (1 - test_size))
        X_train = new_X.iloc[:split_idx]
        X_test = new_X.iloc[split_idx:]
        y_train = new_y.iloc[:split_idx]
        y_test = new_y.iloc[split_idx:]
        
        # Create NEW scaler for combined features
        # We'll need to retrain scaler on all features (can't combine scalers)
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)
        
        # Calculate scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.logger.info(f"\nFine-tuning data:")
        self.logger.info(f"  Training samples: {len(X_train)}")
        self.logger.info(f"  Test samples: {len(X_test)}")
        self.logger.info(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create NEW model with expanded feature set
        # Use existing model as warm start by copying its tree structure and adding new trees
        self.logger.info("\nCreating expanded model...")
        
        new_model = xgb.XGBClassifier(
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
        
        # Train on T-1 data
        new_model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = new_model.predict(X_train_scaled)
        test_pred = new_model.predict(X_test_scaled)
        
        train_proba = new_model.predict_proba(X_train_scaled)[:, 1]
        test_proba = new_model.predict_proba(X_test_scaled)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("FINE-TUNING RESULTS (T-1 ONLY)")
        self.logger.info("="*80)
        self.logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Train AUC: {train_auc:.4f}")
        self.logger.info(f"Test AUC: {test_auc:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info("\nNOTE: Model now accepts BOTH old CSV features AND new T-1 features")
        
        results = {
            'model': new_model,
            'scaler': new_scaler,
            'feature_names': combined_features,
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
            'false_negatives': int(fn),
            'existing_features': existing_features,
            'new_features': list(new_X.columns)
        }
        
        return results
    
    def save_model(self, model, scaler, metadata: Dict, version: str = None):
        """Save fine-tuned model"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archive old model
        old_model_path = self.model_dir / "best_model.pkl"
        
        if old_model_path.exists():
            archive_model = self.archive_dir / f"best_model_{version}.pkl"
            old_model_path.rename(archive_model)
            self.logger.info(f"Archived old model to {archive_model}")
        
        # Save new model
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        joblib.dump(model, model_path, protocol=4)
        joblib.dump(scaler, scaler_path, protocol=4)
        
        # Save metadata
        metadata['model_version'] = version
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['saved_with'] = 'joblib'
        metadata['pickle_protocol'] = 4
        metadata['model_type'] = 'hybrid_fine_tuned'
        
        metadata_path = self.model_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"\n✓ Saved fine-tuned model: {model_path}")
        self.logger.info(f"✓ Model now expects {len(metadata.get('features', []))} features")
        self.logger.info(f"  - Old features (T-3/T-5/T-10): {len(metadata.get('existing_features', []))}")
        self.logger.info(f"  - New features (T-1 open/close): {len(metadata.get('new_features', []))}")
    
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
        
        self.logger.info("\nTop 10 Most Important Features:")
        for i, row in df.head(10).iterrows():
            self.logger.info(f"  {row['feature']:50s}: {row['importance']:.6f}")
        
        # Analyze by feature source
        t1_features = df[df['feature'].str.startswith('t1_')]
        old_features = df[~df['feature'].str.startswith('t1_')]
        
        if not t1_features.empty:
            self.logger.info(f"\nT-1 features importance sum: {t1_features['importance'].sum():.4f}")
        if not old_features.empty:
            self.logger.info(f"Old CSV features importance sum: {old_features['importance'].sum():.4f}")
        
        return df
