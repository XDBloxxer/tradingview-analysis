#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional, List
from datetime import timedelta
import pandas as pd
import yfinance as yf
import time

from ta.momentum import RSIIndicator


class StrategyBacktester:

    INDICATOR_WARMUP_DAYS = 120   # ← REQUIRED, internal only
    MIN_PRICE = 0.50
    MIN_VOLUME = 50000
    MAX_CRITERIA_MATCHES = 50

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.price_cache = {}
        self.indicator_cache = {}
        self.universe: List[str] = []

    # ============================================================
    # MAIN
    # ============================================================

    def run_backtest(self, strategy_config: Dict[str, Any], progress_callback=None):

        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()

        self._build_dynamic_universe()
        self._download_price_data(start_date, end_date)

        trading_days = self._get_trading_days(start_date, end_date)

        all_trades = []
        daily_results = []

        for idx, day in enumerate(trading_days):
            if progress_callback:
                progress_callback(idx + 1, len(trading_days), day)

            matches = self._get_criteria_matches_previous_day(
                day,
                strategy_config['indicator_criteria'],
                strategy_config.get('min_price', self.MIN_PRICE),
                strategy_config.get('max_price'),
                strategy_config.get('min_volume', self.MIN_VOLUME),
            )

            trades = self._evaluate_matches(
                day,
                matches,
                strategy_config['target_min_gain_pct'],
                strategy_config['target_days']
            )

            all_trades.extend(trades)

            daily_results.append(
                self._aggregate_daily_results(
                    day,
                    trades,
                    total_scanned=len(self.indicator_cache),
                    criteria_matches=len(matches)
                )
            )

        return {
            "trades": all_trades,
            "daily_results": daily_results,
            "overall_stats": self._calculate_overall_stats(all_trades)
        }

    # ============================================================
    # UNIVERSE
    # ============================================================

    def _build_dynamic_universe(self):
        sources = [
            "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        ]

        symbols = set()
        for src in sources:
            try:
                df = pd.read_csv(src, sep="|")
                col = "Symbol" if "Symbol" in df.columns else df.columns[0]
                symbols |= {s for s in df[col].astype(str) if s.isalpha()}
            except Exception:
                continue

        self.universe = sorted(symbols)
        self.logger.info(f"Universe size: {len(self.universe)}")

    # ============================================================
    # DATA (WITH INDICATOR WARMUP)
    # ============================================================

    def _download_price_data(self, start, end):

        fetch_start = start - timedelta(days=self.INDICATOR_WARMUP_DAYS)
        fetch_end = end + timedelta(days=5)

        for i in range(0, len(self.universe), 50):
            batch = self.universe[i:i + 50]

            try:
                data = yf.download(
                    batch,
                    start=fetch_start,
                    end=fetch_end,
                    group_by="ticker",
                    threads=True,
                    progress=False
                )

                for symbol in batch:
                    try:
                        df = data if len(batch) == 1 else data[symbol]
                        if isinstance(df, pd.DataFrame) and len(df) > 20:
                            self.price_cache[symbol] = df
                            self.indicator_cache[symbol] = self._calculate_indicators(df)
                    except Exception:
                        continue

                time.sleep(0.3)
            except Exception:
                continue

    # ============================================================
    # CRITERIA
    # ============================================================

    def _get_criteria_matches_previous_day(
        self, date, criteria, min_price, max_price, min_volume
    ):
        matches = []

        for symbol, df in self.indicator_cache.items():
            past = [d for d in df.index if d < date]
            if not past:
                continue

            row = df.loc[past[-1]]

            if pd.isna(row["rsi"]):
                continue

            if row["close"] < min_price or row["volume"] < min_volume:
                continue
            if max_price and row["close"] > max_price:
                continue

            if self._check_criteria(row, criteria):
                matches.append({
                    "symbol": symbol,
                    "signal_date": past[-1],
                    "entry_price": float(row["close"]),
                    "volume": int(row["volume"]),
                })

                if len(matches) >= self.MAX_CRITERIA_MATCHES:
                    break

        return matches

    # ============================================================
    # TRADES
    # ============================================================

    def _evaluate_matches(self, date, matches, target_pct, target_days):
        trades = []

        for m in matches:
            peak, exit_price, gain = self._calculate_outcome(
                m["symbol"], date, m["entry_price"], target_days
            )

            hit = peak is not None and peak >= target_pct

            trades.append({
                "symbol": m["symbol"],
                "exchange": None,
                "signal_date": m["signal_date"].isoformat(),
                "entry_price": m["entry_price"],
                "entry_volume": m["volume"],
                "indicator_values": {},
                "matched_criteria": True,
                "hit_target": hit,
                "actual_gain_pct": gain,
                "exit_price": exit_price,
                "trade_type": "true_positive" if hit else "false_positive",
            })

        return trades

    # ============================================================
    # METRICS
    # ============================================================

    def _aggregate_daily_results(self, date, trades, total_scanned, criteria_matches):
        gains = [t["actual_gain_pct"] for t in trades if t["actual_gain_pct"] is not None]

        tp = sum(t["trade_type"] == "true_positive" for t in trades)
        fp = sum(t["trade_type"] == "false_positive" for t in trades)

        return {
            "test_date": date.isoformat(),
            "total_scanned": total_scanned,
            "criteria_matches": criteria_matches,
            "true_positives": tp,
            "false_positives": fp,
            "missed_opportunities": 0,
            "avg_match_gain_pct": sum(gains) / len(gains) if gains else None,
            "avg_miss_gain_pct": None,
            "max_gain_pct": max(gains) if gains else None,
            "min_gain_pct": min(gains) if gains else None,
        }

    def _calculate_overall_stats(self, trades):
        tp = sum(t["trade_type"] == "true_positive" for t in trades)
        fp = sum(t["trade_type"] == "false_positive" for t in trades)

        total = tp + fp
        gains = [t["actual_gain_pct"] for t in trades if t["actual_gain_pct"] is not None]

        return {
            "total_trades": len(trades),
            "total_matches": total,
            "true_positives": tp,
            "false_positives": fp,
            "missed_opportunities": 0,
            "accuracy_pct": round((tp / total) * 100, 2) if total else 0,
            "avg_gain_pct": sum(gains) / len(gains) if gains else None,
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def _calculate_indicators(self, df):
        out = pd.DataFrame(index=pd.to_datetime(df.index).date)
        out["close"] = df["Close"]
        out["volume"] = df["Volume"]
        out["rsi"] = RSIIndicator(df["Close"], window=14).rsi()
        return out

    def _check_criteria(self, row, criteria):
        for c in criteria:
            val = row.get(c["indicator"])
            if pd.isna(val):
                return False
            if c["operator"] == ">" and not val > c["value"]:
                return False
            if c["operator"] == "<" and not val < c["value"]:
                return False
        return True

    def _calculate_outcome(self, symbol, entry_date, entry_price, days):
        df = self.price_cache.get(symbol)
        if df is None:
            return None, None, None

        dates = pd.to_datetime(df.index).date
        future = [d for d in dates if d >= entry_date]
        if len(future) < 2:
            return None, None, None

        entry_idx = list(dates).index(future[0])
        end_idx = min(entry_idx + days, len(df) - 1)

        peak = df.iloc[entry_idx + 1:end_idx + 1]["High"].max()
        exit_price = df.iloc[end_idx]["Close"]

        peak_gain = ((peak - entry_price) / entry_price) * 100
        exit_gain = ((exit_price - entry_price) / entry_price) * 100

        return float(peak_gain), float(exit_price), float(exit_gain)

    def _get_trading_days(self, start, end):
        return [d.date() for d in pd.date_range(start, end, freq="B")]
