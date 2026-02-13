"""
Adaptive Feature Mapper
Maps available indicators to model features intelligently
Handles missing features gracefully
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set


class FeatureMapper:
    """
    Intelligently maps Daily Winners indicators to model features
    Adapts to available data and fills missing features appropriately
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define comprehensive mapping from Daily Winners to Model features
        # This covers ALL possible indicators from your intraday collector
        self.indicator_mapping = {
            # ===== MOMENTUM INDICATORS =====
            'rsi': ['RSI_14', 'RSI'],
            'rsi[1]': ['RSI_14[1]', 'RSI[1]'],
            'rsi[2]': ['RSI_14[2]', 'RSI[2]'],
            'mom': ['Mom', 'MOM_10'],
            'mom[1]': ['Mom[1]', 'MOM_10[1]'],
            'stoch.k': ['STOCHk_14_3_3', 'Stoch.K', 'STOCHk_Fast'],
            'stoch.d': ['STOCHd_14_3_3', 'Stoch.D', 'STOCHd_Fast'],
            'stoch.k[1]': ['STOCHk_14_3_3[1]', 'Stoch.K[1]'],
            'stoch.d[1]': ['STOCHd_14_3_3[1]', 'Stoch.D[1]'],
            'w.r': ['W.R', 'WILLR_14'],
            'ao': ['AO', 'Awesome_Oscillator'],
            'uo': ['UO', 'Ultimate_Oscillator'],
            'roc': ['ROC', 'ROC_10', 'ROC_20'],
            'kama': ['KAMA', 'KAMA_10'],
            'tsi': ['TSI', 'TSI_13_25_13'],
            
            # ===== TREND INDICATORS =====
            'macd.macd': ['MACD_12_26_9', 'MACD.macd'],
            'macd.signal': ['MACDs_12_26_9', 'MACD.signal'],
            'macd_diff': ['MACDh_12_26_9', 'MACD_diff'],
            'adx': ['ADX', 'ADX_14'],
            'adx+di': ['ADX+DI', 'ADX_pos'],
            'adx-di': ['ADX-DI', 'ADX_neg'],
            'cci20': ['CCI20', 'CCI_20', 'CCI_14'],
            'aroon_up': ['AROON_UP', 'AROONU_25'],
            'aroon_down': ['AROON_DOWN', 'AROOND_25'],
            'aroon_indicator': ['AROON_INDICATOR', 'AROONOSC_25'],
            'psar': ['PSAR', 'PSARl_0.02_0.2'],
            'vortex_pos': ['VORTEX_POS'],
            'vortex_neg': ['VORTEX_NEG'],
            'mass_index': ['MASS_INDEX'],
            'dpo': ['DPO'],
            'kst': ['KST'],
            'kst_signal': ['KST_SIGNAL'],
            
            # ===== MOVING AVERAGES =====
            'ema5': ['EMA5', 'EMA_5'],
            'ema10': ['EMA10', 'EMA_10'],
            'ema20': ['EMA20', 'EMA_20'],
            'ema50': ['EMA50', 'EMA_50'],
            'ema100': ['EMA100', 'EMA_100'],
            'ema200': ['EMA200', 'EMA_200'],
            'sma5': ['SMA5', 'SMA_5'],
            'sma10': ['SMA10', 'SMA_10'],
            'sma20': ['SMA20', 'SMA_20'],
            'sma50': ['SMA50', 'SMA_50'],
            'sma100': ['SMA100', 'SMA_100'],
            'sma200': ['SMA200', 'SMA_200'],
            
            # ===== VOLATILITY INDICATORS =====
            'atr': ['ATR', 'ATR_14', 'ATR_20'],
            'atr_pct': ['ATR_PCT', 'ATR%'],
            'bb.upper': ['BB.upper', 'BBU_20_2.0', 'BBU_20_2.0_2.0'],
            'bb.lower': ['BB.lower', 'BBL_20_2.0', 'BBL_20_2.0_2.0'],
            'bb.middle': ['BB.middle', 'BBM_20_2.0', 'BBM_20_2.0_2.0'],
            'bb_width': ['BB_Width', 'BBB_20_2.0', 'BBB_20_2.0_2.0'],
            'bbpower': ['BBPower', 'BBP_20_2.0'],
            'volatility_20d': ['HV_20', 'Volatility_20d', 'volatility_20d'],
            'keltner_upper': ['KELTNER_UPPER', 'KCU_20_2'],
            'keltner_lower': ['KELTNER_LOWER', 'KCL_20_2'],
            'keltner_middle': ['KELTNER_MIDDLE', 'KCM_20_2'],
            'donchian_upper': ['DONCHIAN_UPPER', 'DCU_20_20'],
            'donchian_lower': ['DONCHIAN_LOWER', 'DCL_20_20'],
            'donchian_middle': ['DONCHIAN_MIDDLE', 'DCM_20_20'],
            
            # ===== VOLUME INDICATORS =====
            'volume': ['Volume', 'volume'],
            'volume_sma5': ['Volume_MA5', 'volume_sma5'],
            'volume_sma10': ['Volume_MA10', 'volume_sma10'],
            'volume_sma20': ['Volume_MA20', 'volume_sma20'],
            'volume_ratio': ['Volume_Ratio', 'volume_ratio'],
            'obv': ['OBV'],
            'cmf': ['CMF', 'CMF_20'],
            'force_index': ['FORCE_INDEX'],
            'eom': ['EOM'],
            'eom_signal': ['EOM_SIGNAL'],
            'vpt': ['VPT'],
            'nvi': ['NVI'],
            
            # ===== PRICE DATA =====
            'close': ['Close', 'close'],
            'open': ['Open', 'open'],
            'high': ['High', 'high'],
            'low': ['Low', 'low'],
            'vwap': ['VWAP'],
            
            # ===== PRICE CHANGES =====
            'price_change_1d': ['price_change_1d', 'Change_1d'],
            'price_change_2d': ['price_change_2d', 'Change_2d'],
            'price_change_3d': ['price_change_3d', 'Change_3d'],
            'price_change_5d': ['price_change_5d', 'Change_5d'],
            'price_change_10d': ['price_change_10d', 'Change_10d'],
            'price_change_20d': ['price_change_20d', 'Change_20d'],
            
            # ===== 52-WEEK HIGH/LOW =====
            'high_52w': ['high_52w', 'High_52w'],
            'low_52w': ['low_52w', 'Low_52w'],
            'price_vs_high_52w': ['price_vs_high_52w'],
            'price_vs_low_52w': ['price_vs_low_52w'],
            
            # ===== GAPS =====
            'gap_%': ['gap_%', 'Gap_%'],
            'gap_up': ['gap_up'],
            'gap_down': ['gap_down'],
            
            # ===== TREND FLAGS =====
            'ema20_above_ema50': ['EMA20_above_EMA50'],
            'ema50_above_ema200': ['EMA50_above_EMA200'],
            'price_above_ema20': ['price_above_EMA20'],
            'ema10_above_ema20': ['EMA10_above_EMA20'],
            'sma50_above_sma200': ['SMA50_above_SMA200'],
            
            # ===== CANDLESTICK PATTERNS =====
            'doji': ['doji', 'Doji'],
            'hammer': ['hammer', 'Hammer'],
            'bullish_engulfing': ['bullish_engulfing', 'Bullish_Engulfing'],
        }
        
        # Reverse mapping for quick lookup
        self.reverse_mapping = {}
        for source_col, target_cols in self.indicator_mapping.items():
            for target_col in target_cols:
                self.reverse_mapping[target_col.lower()] = source_col
    
    def map_features(
        self, 
        available_data: pd.DataFrame,
        expected_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map available indicators to expected model features
        
        Args:
            available_data: DataFrame with Daily Winners indicators
            expected_features: List of features the model expects
        
        Returns:
            Tuple of (mapped_features_df, mapping_report)
        """
        
        self.logger.info(f"Mapping {len(available_data.columns)} available indicators to {len(expected_features)} model features")
        
        # Create output DataFrame with all expected features
        mapped_df = pd.DataFrame(index=available_data.index)
        
        # Track what we found and what's missing
        mapping_report = {
            'found': [],
            'missing': [],
            'filled_with_default': [],
            'derived': []
        }
        
        for feature in expected_features:
            feature_lower = feature.lower()
            
            # Try direct match first
            if feature in available_data.columns:
                mapped_df[feature] = available_data[feature]
                mapping_report['found'].append(feature)
                continue
            
            # Try reverse mapping
            if feature_lower in self.reverse_mapping:
                source_col = self.reverse_mapping[feature_lower]
                if source_col in available_data.columns:
                    mapped_df[feature] = available_data[source_col]
                    mapping_report['found'].append(f"{feature} <- {source_col}")
                    continue
            
            # Try fuzzy matching
            matched = False
            for col in available_data.columns:
                col_lower = col.lower()
                # Check if column name contains the feature name or vice versa
                if feature_lower in col_lower or col_lower in feature_lower:
                    # Additional check for common indicators
                    if self._is_likely_match(feature_lower, col_lower):
                        mapped_df[feature] = available_data[col]
                        mapping_report['found'].append(f"{feature} <- {col} (fuzzy)")
                        matched = True
                        break
            
            if matched:
                continue
            
            # Try deriving from other features
            derived_value = self._derive_feature(feature, available_data)
            if derived_value is not None:
                mapped_df[feature] = derived_value
                mapping_report['derived'].append(feature)
                continue
            
            # Feature not found - use intelligent default
            default_value = self._get_default_value(feature, available_data)
            mapped_df[feature] = default_value
            mapping_report['missing'].append(feature)
            mapping_report['filled_with_default'].append(f"{feature} = {default_value if isinstance(default_value, (int, float)) else 'calculated'}")
        
        # Log mapping summary
        self.logger.info(f"Feature mapping complete:")
        self.logger.info(f"  - Found: {len(mapping_report['found'])}")
        self.logger.info(f"  - Derived: {len(mapping_report['derived'])}")
        self.logger.info(f"  - Missing (filled): {len(mapping_report['missing'])}")
        
        if mapping_report['missing']:
            self.logger.warning(f"  Missing features: {', '.join(mapping_report['missing'][:10])}...")
        
        return mapped_df, mapping_report
    
    def _is_likely_match(self, feature: str, column: str) -> bool:
        """Check if feature and column are likely the same indicator"""
        # Remove common separators
        feature_clean = feature.replace('_', '').replace('.', '').replace('-', '')
        column_clean = column.replace('_', '').replace('.', '').replace('-', '')
        
        # Check exact match after cleaning
        if feature_clean == column_clean:
            return True
        
        # Check for common indicator abbreviations
        common_matches = {
            'rsi': ['rsi', 'relativestrengthindex'],
            'macd': ['macd', 'movingaverageconvergence'],
            'adx': ['adx', 'averagedirectional'],
            'stoch': ['stoch', 'stochastic'],
            'ema': ['ema', 'exponentialmovingaverage'],
            'sma': ['sma', 'simplemovingaverage'],
            'bb': ['bb', 'bollinger'],
            'atr': ['atr', 'averagetruerange'],
        }
        
        for key, variations in common_matches.items():
            if any(v in feature_clean for v in variations) and any(v in column_clean for v in variations):
                return True
        
        return False
    
    def _derive_feature(self, feature: str, data: pd.DataFrame) -> pd.Series:
        """Try to derive a feature from other available features"""
        
        feature_lower = feature.lower()
        
        # Derive MA crossovers
        if 'above' in feature_lower:
            parts = feature_lower.split('_above_')
            if len(parts) == 2:
                ma1, ma2 = parts
                # Try to find both MAs
                ma1_col = self._find_column(ma1, data)
                ma2_col = self._find_column(ma2, data)
                
                if ma1_col and ma2_col:
                    return (data[ma1_col] > data[ma2_col]).astype(int)
        
        # Derive price changes from close
        if 'price_change' in feature_lower and 'close' in data.columns:
            # Extract number of days
            import re
            match = re.search(r'(\d+)d', feature_lower)
            if match:
                days = int(match.group(1))
                return data['close'].pct_change(days) * 100
        
        # Derive volume ratio
        if 'volume_ratio' in feature_lower and 'volume' in data.columns:
            if 'volume_sma20' in data.columns:
                return data['volume'] / data['volume_sma20']
            else:
                # Calculate it
                vol_ma = data['volume'].rolling(window=20).mean()
                return data['volume'] / vol_ma
        
        # Derive BB width
        if 'bb_width' in feature_lower or 'bbb' in feature_lower:
            if 'bb.upper' in data.columns and 'bb.lower' in data.columns and 'bb.middle' in data.columns:
                return (data['bb.upper'] - data['bb.lower']) / data['bb.middle'] * 100
        
        return None
    
    def _find_column(self, pattern: str, data: pd.DataFrame) -> str:
        """Find a column matching a pattern"""
        pattern_lower = pattern.lower()
        for col in data.columns:
            if pattern_lower in col.lower() or col.lower() in pattern_lower:
                return col
        return None
    
    def _get_default_value(self, feature: str, data: pd.DataFrame) -> float:
        """
        Get intelligent default value for missing feature
        Uses statistical properties of available data
        """
        
        feature_lower = feature.lower()
        
        # Normalized indicators (0-100 range) - use neutral value
        if any(ind in feature_lower for ind in ['rsi', 'stoch', 'w.r', 'cci']):
            return 50.0
        
        # Percentage-based indicators - use 0
        if any(ind in feature_lower for ind in ['change', 'pct', '%', 'ratio']):
            return 0.0
        
        # Boolean flags - use 0 (False)
        if any(word in feature_lower for word in ['above', 'below', 'cross', 'flag', 'doji', 'hammer']):
            return 0.0
        
        # Volume - use median if available
        if 'volume' in feature_lower:
            if 'volume' in data.columns:
                return data['volume'].median()
            return 100000.0  # Default volume
        
        # Price-related - use median close if available
        if any(word in feature_lower for word in ['price', 'close', 'open', 'high', 'low']):
            if 'close' in data.columns:
                return data['close'].median()
            return 50.0  # Neutral default price
        
        # Oscillators around zero - use 0
        if any(ind in feature_lower for ind in ['macd', 'ao', 'roc', 'mom']):
            return 0.0
        
        # Moving averages - use close if available
        if any(ind in feature_lower for ind in ['ema', 'sma', 'wma', 'vwap']):
            if 'close' in data.columns:
                return data['close'].median()
            return 50.0
        
        # Volatility indicators - use small positive value
        if any(ind in feature_lower for ind in ['atr', 'volatility', 'hv']):
            return 1.0
        
        # Default: 0 for unknown features
        return 0.0
    
    def get_feature_coverage_report(self, mapped_df: pd.DataFrame, expected_features: List[str]) -> Dict:
        """Generate comprehensive report on feature coverage"""
        
        report = {
            'total_features': len(expected_features),
            'features_found': 0,
            'features_missing': 0,
            'coverage_pct': 0.0,
            'missing_features': [],
            'zero_variance_features': [],
            'high_missing_rate_features': []
        }
        
        for feature in expected_features:
            if feature in mapped_df.columns:
                # Check if actually has data (not all zeros or NaN)
                non_zero = (mapped_df[feature] != 0).sum()
                non_na = mapped_df[feature].notna().sum()
                
                if non_zero > 0 and non_na > 0:
                    report['features_found'] += 1
                else:
                    report['features_missing'] += 1
                    report['missing_features'].append(feature)
                
                # Check for zero variance
                if mapped_df[feature].std() == 0:
                    report['zero_variance_features'].append(feature)
                
                # Check for high missing rate
                missing_rate = 1 - (non_na / len(mapped_df))
                if missing_rate > 0.5:
                    report['high_missing_rate_features'].append(feature)
            else:
                report['features_missing'] += 1
                report['missing_features'].append(feature)
        
        report['coverage_pct'] = (report['features_found'] / report['total_features']) * 100
        
        return report