"""
Model Trainer - FIXED VERSION WITH NON-WINNERS SUPPORT
Addresses all issues from the analysis conversation:
1. Training data leakage (only uses T-1 data)
2. Selection bias (includes negative examples)
3. Incremental learning (preserves historical patterns)
4. Same-day data exclusion (never trains on current day)
"""

import logging
import joblib  # Use joblib instead of pickle
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
    INCREMENTAL: Preserves original research data while adding new learnings
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Archive directory for old models
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        # CRITICAL: Historical data directory
        self.historical_data_dir = Path("ml_models/historical_data")
        self.historical_data_dir.mkdir(exist_ok=True)
    
    def load_historical_training_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Load original historical training data (10k stocks × 2 years)
        This preserves your original research insights with T-3, T-5, T-10 lags
        
        Expected file: ml_models/historical_data/original_training_data.pkl
        Format: {'X': DataFrame, 'y': Series, 'metadata': dict}
        """
        
        historical_file = self.historical_data_dir / "original_training_data.pkl"
        
        if not historical_file.exists():
            self.logger.warning(f"Historical data not found at {historical_file}")
            self.logger.warning("Model will train ONLY on daily winners data (not recommended)")
            self.logger.info("To preserve original research:")
            self.logger.info("  1. Save your original training data as:")
            self.logger.info(f"     {historical_file}")
            self.logger.info("  2. Format: joblib dump with dict containing 'X', 'y', 'metadata'")
            return None, None, None
        
        try:
            data = joblib.load(historical_file)
            
            X = data['X']
            y = data['y']
            metadata = data.get('metadata', {})
            
            self.logger.info(f"✓ Loaded historical training data:")
            self.logger.info(f"  Samples: {len(X)}")
            self.logger.info(f"  Features: {len(X.columns)}")
            self.logger.info(f"  Positives: {y.sum()}")
            self.logger.info(f"  Date range: {metadata.get('date_range', 'unknown')}")
            self.logger.info(f"  Source: {metadata.get('source', 'Original research')}")
            
            return X, y, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return None, None, None
    
    def prepare_training_data_from_daily_winners(
        self,
        supabase_client,
        lookback_days: int = 90,
        use_all_timepoints: bool = True,
        include_non_winners: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Prepare NEW training data from Daily Winners AND Non-Winners systems
        FIXED: Only uses T-1 data (day_prior_open, day_prior_close)
        NEVER uses same-day data (market_open, market_close) per the analysis
        
        Strategy:
        1. Get T-1 close data (PRIMARY - most important)
        2. Optionally add T-1 open for more examples
        3. Include both winners (label=1) and non-winners (label=0)
        4. CRITICAL: Never train on same-day data (prevents leakage)
        """
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info(f"Preparing NEW training data ({start_date} to {end_date})")
        self.logger.info(f"Include non-winners (negative examples): {include_non_winners}")
        self.logger.info("TRAINING STRATEGY: Use ONLY T-1 data (day_prior_close, day_prior_open)")
        self.logger.info("                   NEVER use same-day data (prevents leakage)")
        
        client = supabase_client.client
        
        # ===== STEP 1: Get WINNERS T-1 close data (POSITIVE EXAMPLES) =====
        self.logger.info("\nLoading WINNERS day_prior_close data (positive examples)...")
        
        response = client.table("winners_day_prior_close")\
            .select("*")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        if not response.data:
            self.logger.warning("No winners day_prior_close data available")
            winners_day_prior_close_df = pd.DataFrame()
        else:
            winners_day_prior_close_df = pd.DataFrame(response.data)
            self.logger.info(f"  Loaded {len(winners_day_prior_close_df)} winners T-1 close records")
        
        # ===== STEP 1b: Get NON-WINNERS T-1 close data (NEGATIVE EXAMPLES) =====
        non_winners_day_prior_close_df = pd.DataFrame()
        
        if include_non_winners:
            self.logger.info("Loading NON-WINNERS day_prior_close data (negative examples)...")
            
            try:
                response = client.table("non_winners_day_prior_close")\
                    .select("*")\
                    .gte("detection_date", start_date.isoformat())\
                    .lte("detection_date", end_date.isoformat())\
                    .execute()
                
                if response.data:
                    non_winners_day_prior_close_df = pd.DataFrame(response.data)
                    self.logger.info(f"  Loaded {len(non_winners_day_prior_close_df)} non-winners T-1 close records")
                else:
                    self.logger.warning("  No non-winners data found - run daily_non_winners_main.py")
            except Exception as e:
                self.logger.warning(f"  Could not load non-winners data: {e}")
                self.logger.warning("  Training WITHOUT negative examples (high false positive risk!)")
        
        # Combine winners and non-winners
        if not winners_day_prior_close_df.empty and not non_winners_day_prior_close_df.empty:
            day_prior_close_df = pd.concat([winners_day_prior_close_df, non_winners_day_prior_close_df], ignore_index=True)
            self.logger.info(f"  Combined: {len(day_prior_close_df)} total T-1 close records")
        elif not winners_day_prior_close_df.empty:
            day_prior_close_df = winners_day_prior_close_df
            self.logger.warning("  Using ONLY winners data (no negative examples - BIASED!)")
        else:
            self.logger.error("  No training data available!")
            return pd.DataFrame(), pd.Series(), {}
        
        # ===== STEP 2: Get actual winners for labeling =====
        self.logger.info("\nLoading actual winners for labeling...")
        
        response = client.table("daily_winners")\
            .select("symbol,detection_date,change_pct,price")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        if not response.data:
            self.logger.warning("No winner data available")
            return pd.DataFrame(), pd.Series(), {}
        
        winners_df = pd.DataFrame(response.data)
        self.logger.info(f"  Loaded {len(winners_df)} actual winner records")
        
        # ===== STEP 3: Create training samples from T-1 CLOSE =====
        self.logger.info("\nCreating labeled training samples from T-1 CLOSE data...")
        
        training_samples = []
        
        for _, row in day_prior_close_df.iterrows():
            symbol = row['symbol']
            detection_date = row['detection_date']
            
            # Check if this stock became a winner
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
                'timepoint': 'day_prior_close'
            }
            
            # Add all indicator features
            for col in row.index:
                if col not in ['symbol', 'detection_date', 'exchange', 'snapshot_type',
                              'snapshot_time', 'snapshot_date', 'id', 'created_at']:
                    sample[col] = row[col]
            
            training_samples.append(sample)
        
        # ===== STEP 4: OPTIONAL - Add T-1 OPEN data =====
        if use_all_timepoints:
            self.logger.info("\nAdding T-1 OPEN data for more training samples...")
            
            # Load winners T-1 open
            response = client.table("winners_day_prior_open")\
                .select("*")\
                .gte("detection_date", start_date.isoformat())\
                .lte("detection_date", end_date.isoformat())\
                .execute()
            
            winners_day_prior_open_df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
            
            if not winners_day_prior_open_df.empty:
                self.logger.info(f"  Loaded {len(winners_day_prior_open_df)} winners T-1 open records")
            
            # Load non-winners T-1 open
            non_winners_day_prior_open_df = pd.DataFrame()
            if include_non_winners:
                try:
                    response = client.table("non_winners_day_prior_open")\
                        .select("*")\
                        .gte("detection_date", start_date.isoformat())\
                        .lte("detection_date", end_date.isoformat())\
                        .execute()
                    
                    if response.data:
                        non_winners_day_prior_open_df = pd.DataFrame(response.data)
                        self.logger.info(f"  Loaded {len(non_winners_day_prior_open_df)} non-winners T-1 open records")
                except Exception as e:
                    self.logger.warning(f"  Could not load non-winners T-1 open: {e}")
            
            # Combine T-1 open data
            if not winners_day_prior_open_df.empty and not non_winners_day_prior_open_df.empty:
                day_prior_open_df = pd.concat([winners_day_prior_open_df, non_winners_day_prior_open_df], ignore_index=True)
            elif not winners_day_prior_open_df.empty:
                day_prior_open_df = winners_day_prior_open_df
            else:
                day_prior_open_df = pd.DataFrame()
            
            if not day_prior_open_df.empty:
                self.logger.info(f"  Combined T-1 open: {len(day_prior_open_df)} records")
                
                for _, row in day_prior_open_df.iterrows():
                    symbol = row['symbol']
                    detection_date = row['detection_date']
                    
                    # Skip if we already have this stock from T-1 close
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
        
        if not training_samples:
            return pd.DataFrame(), pd.Series(), {}
        
        # Convert to DataFrame
        df = pd.DataFrame(training_samples)
        
        self.logger.info(f"\n✓ Created {len(df)} NEW training samples")
        self.logger.info(f"  - Positives (winners): {df['label'].sum()}")
        self.logger.info(f"  - Negatives (non-winners): {len(df) - df['label'].sum()}")
        self.logger.info(f"  - Positive rate: {df['label'].mean()*100:.2f}%")
        self.logger.info(f"  - Timepoints used: {list(df['timepoint'].unique())}")
        
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
            'source': 'daily_winners_and_non_winners' if include_non_winners else 'daily_winners_only',
            'timepoints_used': list(df['timepoint'].unique()),
            'includes_negative_examples': include_non_winners,
            'uses_only_t1_data': True,  # Confirms no same-day data leakage
            'same_day_data_excluded': True  # Per analysis requirements
        }
        
        return X, y, metadata
    
    def combine_training_data(
        self,
        historical_X: pd.DataFrame,
        historical_y: pd.Series,
        new_X: pd.DataFrame,
        new_y: pd.Series,
        historical_weight: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series, Dict, np.ndarray]:
        """
        CRITICAL: Combine historical and new data intelligently
        Per analysis: Keep historical weight at 1.0 to preserve T-3, T-5, T-10 insights
        
        Strategy:
        1. Keep ALL historical data (preserve research insights)
        2. Add new daily winners/non-winners data (real-world calibration)
        3. Apply sample weighting: EQUAL weight to both (1.0 each)
        
        Args:
            historical_weight: Weight for historical samples (1.0 = equal, RECOMMENDED)
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("COMBINING HISTORICAL + NEW DATA")
        self.logger.info("="*80)
        
        if historical_X is None or historical_X.empty:
            self.logger.warning("No historical data - using ONLY new data")
            self.logger.warning("Model may forget time-lag patterns (T-3, T-5, T-10)!")
            return new_X, new_y, {'source': 'new_only'}, None
        
        if new_X.empty:
            self.logger.warning("No new data - using ONLY historical data")
            return historical_X, historical_y, {'source': 'historical_only'}, None
        
        # Align features (use union of all features)
        all_features = sorted(list(set(historical_X.columns) | set(new_X.columns)))
        
        self.logger.info(f"Feature alignment:")
        self.logger.info(f"  Historical features: {len(historical_X.columns)}")
        self.logger.info(f"  New features: {len(new_X.columns)}")
        self.logger.info(f"  Combined features: {len(all_features)}")
        
        # Add missing features with zeros
        for feat in all_features:
            if feat not in historical_X.columns:
                historical_X[feat] = 0
            if feat not in new_X.columns:
                new_X[feat] = 0
        
        # Reorder to match
        historical_X = historical_X[all_features]
        new_X = new_X[all_features]
        
        # Combine datasets
        X_combined = pd.concat([historical_X, new_X], ignore_index=True)
        y_combined = pd.concat([historical_y, new_y], ignore_index=True)
        
        # Create sample weights - EQUAL weighting per analysis
        historical_samples = len(historical_X)
        new_samples = len(new_X)
        
        sample_weights = np.concatenate([
            np.full(historical_samples, historical_weight),
            np.full(new_samples, historical_weight)  # EQUAL weight
        ])
        
        self.logger.info(f"\nCombined training data:")
        self.logger.info(f"  Historical samples: {historical_samples} (weight: {historical_weight})")
        self.logger.info(f"  New samples: {new_samples} (weight: {historical_weight})")
        self.logger.info(f"  Total samples: {len(X_combined)}")
        self.logger.info(f"  Total positives: {y_combined.sum()}")
        self.logger.info(f"  Positive rate: {y_combined.mean()*100:.2f}%")
        self.logger.info("  ✓ Equal weighting preserves historical T-lag insights")
        
        metadata = {
            'n_samples': len(X_combined),
            'n_historical': historical_samples,
            'n_new': new_samples,
            'historical_weight': historical_weight,
            'n_positives': int(y_combined.sum()),
            'positive_rate': float(y_combined.mean()),
            'features': all_features,
            'source': 'combined'
        }
        
        return X_combined, y_combined, metadata, sample_weights
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray = None,
        use_time_series_split: bool = True,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train XGBoost model with proper validation
        
        Args:
            X: Features
            y: Labels
            sample_weights: Sample importance weights (for incremental learning)
            use_time_series_split: Use time-series aware split
            test_size: Test set size
        
        Returns:
            Dict with model, scaler, and metrics
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING XGBOOST MODEL")
        self.logger.info("="*80)
        self.logger.info(f"  Training samples: {len(X)}")
        self.logger.info(f"  Features: {len(X.columns)}")
        self.logger.info(f"  Using sample weights: {sample_weights is not None}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        if use_time_series_split:
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            if sample_weights is not None:
                weights_train = sample_weights[:split_idx]
                weights_test = sample_weights[split_idx:]
            else:
                weights_train = None
                weights_test = None
            
            self.logger.info("Using time-series split (preserves temporal order)")
        else:
            if sample_weights is not None:
                X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
                    X, y, sample_weights, test_size=test_size, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                weights_train = None
                weights_test = None
            
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
        
        # Fit with sample weights if provided
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
        """Save model with versioning and metadata - FIXED PICKLE PROTOCOL"""
        
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
        
        # Save new model - Use joblib
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata['model_version'] = version
        metadata['saved_with'] = 'joblib'
        
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
