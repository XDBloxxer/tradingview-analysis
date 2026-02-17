"""
Daily Winners Detector - DEBUG VERSION
This version has extensive logging to diagnose why stale stocks pass validation

FIX: All three fetch paths now populate high, low, open, close so the
     daily_winners table is fully populated after the schema migration.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time

try:
    from tradingview_scraper.symbols.market_movers import MarketMovers
except ImportError:
    raise ImportError(
        "tradingview-scraper is required. Install with: pip install tradingview-scraper"
    )

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners - DEBUG VERSION with extensive logging
    """
    
    EXCLUDED_PATTERNS = [
        'OTC',  # Over-the-counter
        '.PK',  # Pink sheets
        '.OB',  # OTC Bulletin Board
        '-',    # Often delisted stocks
    ]
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        self.min_price = detection_config.get("min_price", 0.25)
        self.min_volume = detection_config.get("min_volume", 10000)
        
        self.rate_limiter = RateLimiter(config)
        self.market_movers = MarketMovers(export_result=False)
        self.freshness_cache = {}
        
        self.logger.info(
            f"Daily Winners detector initialized (DEBUG MODE): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 15, target_date: datetime = None) -> List[Dict[str, Any]]:
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Fetching top {top_n} day gainers from TradingView MarketMovers...")
        
        # Fetch MORE than we need initially to account for validation rejections
        initial_fetch_size = top_n * 2
        
        candidates = self._fetch_from_tradingview(target_date, initial_fetch_size)
        
        # Supplement with yfinance if needed
        needed = initial_fetch_size - len(candidates)
        if needed > 0:
            self.logger.info(
                f"TradingView returned {len(candidates)}/{initial_fetch_size}. "
                f"Fetching {needed} more from yfinance..."
            )
            existing_symbols = {c['symbol'] for c in candidates}
            yf_candidates = self._fetch_from_yfinance_screener(target_date, needed * 2, existing_symbols)
            candidates.extend(yf_candidates[:needed])
        
        if not candidates:
            self.logger.warning("⚠️ No winners found!")
            return []
        
        # ── Backfill OHLC for any candidate missing it (TradingView path) ──
        candidates = self._backfill_ohlc(candidates)
        
        self.logger.info(f"Total candidates before validation: {len(candidates)}")
        
        skip_validation = self.config.get('daily_winners', {}).get('skip_freshness_validation', False)
        self.logger.info(f"🔍 skip_freshness_validation config: {skip_validation}")
        
        if skip_validation:
            self.logger.warning("⚠️ SKIPPING freshness validation per config!")
            validated_candidates = candidates
        else:
            self.logger.info(f"🔍 Running batch freshness validation on {len(candidates)} candidates...")
            validated_candidates = self._batch_verify_freshness(candidates, target_date)
        
        if not validated_candidates:
            self.logger.warning("⚠️ All candidates failed freshness validation!")
            return []
        
        # Sort by change percentage
        df_results = pd.DataFrame(validated_candidates)
        df_results = df_results.sort_values('change_pct', ascending=False)
        
        # Backfill shortage
        if len(df_results) < top_n:
            shortage = top_n - len(df_results)
            self.logger.warning(
                f"⚠️ Only {len(df_results)} stocks passed validation (need {top_n}). "
                f"Shortage: {shortage} stocks"
            )
            self.logger.info("💡 Trying to backfill with additional candidates...")
            
            existing_symbols = {c['symbol'] for c in candidates}
            additional_candidates = self._fetch_from_yfinance_screener(
                target_date, shortage * 3, existing_symbols
            )
            
            if additional_candidates:
                additional_candidates = self._backfill_ohlc(additional_candidates)
                self.logger.info(f"Fetched {len(additional_candidates)} additional candidates")
                
                if not skip_validation:
                    additional_validated = self._batch_verify_freshness(additional_candidates, target_date)
                else:
                    additional_validated = additional_candidates
                
                if additional_validated:
                    self.logger.info(f"✅ {len(additional_validated)} additional candidates passed validation")
                    validated_candidates.extend(additional_validated)
                    df_results = pd.DataFrame(validated_candidates)
                    df_results = df_results.sort_values('change_pct', ascending=False)
        
        top_winners = df_results.head(top_n).to_dict('records')
        
        for winner in top_winners:
            winner['detection_date'] = target_date_str
            winner['detection_time'] = '16:00:00'
        
        self.logger.info(f"✅ Found top {len(top_winners)} daily winners (target: {top_n}):")
        for i, winner in enumerate(top_winners, 1):
            source = winner.pop('source', 'unknown')
            self.logger.info(
                f"  #{i}: {winner['exchange']}:{winner['symbol']} "
                f"(+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}, "
                f"vol={winner['volume']:,} [{source}]"
            )
        
        if len(top_winners) < top_n:
            self.logger.warning(
                f"⚠️ WARNING: Only found {len(top_winners)}/{top_n} winners after validation."
            )
        
        return top_winners
    
    # ─────────────────────────────────────────────────────────────────────
    # OHLC backfill — fetches daily bar OHLC for any candidate that came
    # in without it (mainly the TradingView MarketMovers path).
    # Uses a single yf.download() batch call for efficiency.
    # ─────────────────────────────────────────────────────────────────────

    def _backfill_ohlc(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For any candidate missing high/low/open/close, fetch from yfinance
        daily bars (1d interval) and fill them in.

        TradingView MarketMovers only returns close + change + volume.
        yfinance Screener returns regularMarketPrice but not the full OHLC bar.
        This method patches both.
        """
        missing = [c for c in candidates
                   if any(c.get(f) is None for f in ('high', 'low', 'open', 'close'))]

        if not missing:
            return candidates

        symbols = list({c['symbol'] for c in missing})
        self.logger.info(f"Backfilling OHLC for {len(symbols)} candidates via yfinance daily bars...")

        try:
            data = yf.download(
                symbols,
                period='2d',
                interval='1d',
                group_by='ticker',
                progress=False,
                threads=True,
                auto_adjust=True,
            )

            if data.empty:
                self.logger.warning("yfinance returned no OHLC data for backfill")
                return candidates

            # Build a quick lookup: symbol → latest OHLC bar
            ohlc_lookup: Dict[str, Dict[str, float]] = {}

            if len(symbols) == 1:
                # Single-symbol download: columns are Open/High/Low/Close/Volume (no ticker level)
                if not data.empty:
                    row = data.iloc[-1]
                    ohlc_lookup[symbols[0]] = {
                        'open':  float(row.get('Open',  row.get('open',  0))),
                        'high':  float(row.get('High',  row.get('high',  0))),
                        'low':   float(row.get('Low',   row.get('low',   0))),
                        'close': float(row.get('Close', row.get('close', 0))),
                    }
            else:
                # Multi-symbol: columns are MultiIndex (ticker, field)
                for sym in symbols:
                    try:
                        if sym not in data.columns.get_level_values(0):
                            continue
                        sym_data = data[sym].dropna(how='all')
                        if sym_data.empty:
                            continue
                        row = sym_data.iloc[-1]
                        ohlc_lookup[sym] = {
                            'open':  float(row.get('Open',  row.get('open',  0))),
                            'high':  float(row.get('High',  row.get('high',  0))),
                            'low':   float(row.get('Low',   row.get('low',   0))),
                            'close': float(row.get('Close', row.get('close', 0))),
                        }
                    except Exception:
                        continue

        except Exception as e:
            self.logger.warning(f"OHLC backfill failed: {e}")
            return candidates

        # Patch candidates in-place
        patched = 0
        for c in candidates:
            sym = c.get('symbol')
            if sym in ohlc_lookup:
                bar = ohlc_lookup[sym]
                for field in ('open', 'high', 'low', 'close'):
                    if c.get(field) is None:
                        c[field] = bar[field]
                # Also align price with close if it was 0 / None
                if not c.get('price') and bar['close']:
                    c['price'] = bar['close']
                patched += 1

        self.logger.info(f"✓ OHLC backfilled for {patched}/{len(missing)} candidates")
        return candidates

    # ─────────────────────────────────────────────────────────────────────
    # Existing helpers (unchanged except _fetch_from_tradingview which now
    # includes high/low/open when the scraper returns them)
    # ─────────────────────────────────────────────────────────────────────

    def _is_excluded_symbol(self, symbol: str, exchange: str) -> bool:
        if exchange == 'OTC':
            return True
        symbol_upper = symbol.upper()
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in symbol_upper:
                return True
        if len(symbol) > 5:
            return True
        return False
    
    def _fetch_from_tradingview(self, target_date: datetime, top_n: int) -> List[Dict[str, Any]]:
        try:
            fetch_limit = max(top_n * 10, 500)
            
            result = self.market_movers.scrape(
                market='stocks-usa',
                category='gainers',
                limit=fetch_limit
            )
            
            if result['status'] != 'success':
                self.logger.error(f"TradingView API error: {result.get('status')}")
                return []
            
            data = result.get('data', [])
            if not data:
                self.logger.warning("No data from TradingView")
                return []
            
            self.logger.info(f"Received {len(data)} gainers from TradingView")
            
            all_candidates = []
            filtered_counts = {
                'excluded_pattern': 0,
                'low_price': 0,
                'low_volume': 0,
                'no_change': 0,
                'parse_error': 0
            }
            
            for item in data:
                try:
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                    else:
                        symbol = symbol_full
                        exchange_prefix = 'NASDAQ'
                    
                    if not symbol:
                        continue
                    
                    exchange_map = {
                        'NASDAQ': 'NASDAQ',
                        'NYSE':   'NYSE',
                        'AMEX':   'AMEX',
                        'BATS':   'NASDAQ'
                    }
                    exchange = exchange_map.get(exchange_prefix, exchange_prefix)
                    
                    if self._is_excluded_symbol(symbol, exchange):
                        filtered_counts['excluded_pattern'] += 1
                        self.logger.debug(f"🚫 Filtered (pattern): {exchange}:{symbol}")
                        continue
                    
                    price      = float(item.get('close', 0))
                    change_pct = float(item.get('change', 0))
                    volume     = int(item.get('volume', 0))
                    
                    if price < self.min_price:
                        filtered_counts['low_price'] += 1
                        self.logger.debug(f"🚫 Filtered (price): {symbol} ${price:.2f}")
                        continue
                    
                    if volume < self.min_volume:
                        filtered_counts['low_volume'] += 1
                        self.logger.debug(f"🚫 Filtered (volume): {symbol} {volume:,}")
                        continue
                    
                    if change_pct <= 0:
                        filtered_counts['no_change'] += 1
                        continue
                    
                    # Capture OHLC from TradingView if available; None otherwise
                    # (None fields are backfilled by _backfill_ohlc)
                    all_candidates.append({
                        'symbol':     symbol.strip().upper(),
                        'exchange':   exchange,
                        'price':      float(price),
                        'change_pct': float(change_pct),
                        'volume':     int(volume),
                        'high':       float(item['high'])  if item.get('high')  else None,
                        'low':        float(item['low'])   if item.get('low')   else None,
                        'open':       float(item['open'])  if item.get('open')  else None,
                        'close':      float(item['close']) if item.get('close') else float(price),
                        'source':     'tradingview',
                    })
                    
                    self.logger.debug(f"✅ Passed filters: {exchange}:{symbol} +{change_pct:.2f}%")
                    
                except Exception:
                    filtered_counts['parse_error'] += 1
                    continue
            
            self.logger.info(
                f"TradingView filtering: total={len(data)}, "
                f"excluded={filtered_counts['excluded_pattern']}, "
                f"low_price={filtered_counts['low_price']}, "
                f"low_volume={filtered_counts['low_volume']}, "
                f"no_change={filtered_counts['no_change']}, "
                f"passed={len(all_candidates)}"
            )
            
            return all_candidates
            
        except Exception as e:
            self.logger.error(f"Error fetching from TradingView: {e}", exc_info=True)
            return []
    
    def _fetch_from_yfinance_screener(
        self, 
        target_date: datetime, 
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        try:
            self.logger.info("Fetching from yfinance screener...")
            
            try:
                from yfinance import Screener
                
                screener = Screener()
                gainers_data = screener.get_screeners(['day_gainers'], count=limit * 2)
                
                if not gainers_data or 'day_gainers' not in gainers_data:
                    return self._fetch_from_liquid_stocks(target_date, limit, exclude_symbols)
                
                quotes = gainers_data['day_gainers'].get('quotes', [])
                candidates = []
                
                for quote in quotes:
                    symbol = quote.get('symbol', '')
                    if not symbol or symbol in exclude_symbols:
                        continue
                    if self._is_excluded_symbol(symbol, 'NASDAQ'):
                        continue
                    
                    price      = quote.get('regularMarketPrice', 0)
                    change_pct = quote.get('regularMarketChangePercent', 0)
                    volume     = quote.get('regularMarketVolume', 0)
                    
                    if price < self.min_price or volume < self.min_volume or change_pct <= 0:
                        continue
                    
                    # yfinance Screener provides regularMarketDayHigh/Low/Open
                    candidates.append({
                        'symbol':     symbol.upper(),
                        'exchange':   quote.get('exchange', 'NASDAQ'),
                        'price':      float(price),
                        'change_pct': float(change_pct),
                        'volume':     int(volume),
                        'high':       float(quote['regularMarketDayHigh'])  if quote.get('regularMarketDayHigh')  else None,
                        'low':        float(quote['regularMarketDayLow'])   if quote.get('regularMarketDayLow')   else None,
                        'open':       float(quote['regularMarketOpen'])     if quote.get('regularMarketOpen')     else None,
                        'close':      float(price),
                        'source':     'yfinance_screener',
                    })
                    
                    if len(candidates) >= limit:
                        break
                
                self.logger.info(f"✅ Found {len(candidates)} from yfinance Screener")
                return candidates
                
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Screener unavailable: {e}")
                return self._fetch_from_liquid_stocks(target_date, limit, exclude_symbols)
                
        except Exception as e:
            self.logger.error(f"Error in yfinance screener: {e}", exc_info=True)
            return []
    
    def _fetch_from_liquid_stocks(
        self,
        target_date: datetime,
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        try:
            self.logger.info("Scanning liquid stocks...")
            
            candidates = []
            liquid_stocks = self._get_liquid_stocks_list()
            liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]
            
            batch_size = 50
            for i in range(0, len(liquid_stocks), batch_size):
                batch = liquid_stocks[i:i + batch_size]
                
                try:
                    data = yf.download(
                        batch,
                        period='2d',
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        threads=True,
                        auto_adjust=True,
                    )
                    
                    if data.empty:
                        continue
                    
                    for symbol in batch:
                        if symbol in exclude_symbols:
                            continue
                        
                        try:
                            if len(batch) == 1:
                                stock_data = data
                            else:
                                if symbol not in data.columns.get_level_values(0):
                                    continue
                                stock_data = data[symbol]
                            
                            if stock_data.empty or len(stock_data) < 2:
                                continue
                            
                            latest   = stock_data.iloc[-1]
                            previous = stock_data.iloc[-2]
                            
                            close      = float(latest['Close'])
                            prev_close = float(previous['Close'])
                            volume     = int(latest['Volume'])
                            change_pct = ((close - prev_close) / prev_close) * 100
                            
                            if close < self.min_price or volume < self.min_volume or change_pct <= 0:
                                continue
                            if self._is_excluded_symbol(symbol, 'NASDAQ'):
                                continue
                            
                            candidates.append({
                                'symbol':     symbol.upper(),
                                'exchange':   'NASDAQ',
                                'price':      close,
                                'change_pct': float(change_pct),
                                'volume':     volume,
                                # Full OHLC available from daily bar
                                'high':       float(latest['High']),
                                'low':        float(latest['Low']),
                                'open':       float(latest['Open']),
                                'close':      close,
                                'source':     'yfinance_liquid',
                            })
                            
                        except Exception:
                            continue
                
                except Exception:
                    continue
                
                if len(candidates) >= limit:
                    break
                
                time.sleep(0.3)
            
            if candidates:
                candidates = sorted(candidates, key=lambda x: x['change_pct'], reverse=True)
                candidates = candidates[:limit]
            
            self.logger.info(f"✅ Found {len(candidates)} from liquid stocks")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error in liquid stocks: {e}", exc_info=True)
            return []
    
    def _batch_verify_freshness(
        self, 
        candidates: List[Dict[str, Any]], 
        target_date: datetime,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Enhanced debug version — shows exactly why each symbol passes or fails."""
        if not candidates:
            return []
        
        valid_candidates = []
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        
        symbols = [c['symbol'] for c in candidates]
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 FRESHNESS VALIDATION DEBUG - Target date: {target_date_obj}")
        self.logger.info("=" * 80)
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols    = symbols[i:i + batch_size]
            batch_candidates = candidates[i:i + batch_size]
            
            self.logger.info(f"📦 Batch {i//batch_size + 1}: Checking {len(batch_symbols)} symbols")
            
            try:
                data = yf.download(
                    batch_symbols,
                    period='5d',
                    interval='1d',
                    group_by='ticker',
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )
                
                if data.empty:
                    self.logger.warning(f"⚠️ No data returned for batch {i//batch_size + 1}")
                    continue
                
                for symbol, candidate in zip(batch_symbols, batch_candidates):
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"🔍 Checking: {symbol}")
                    
                    try:
                        if len(batch_symbols) == 1:
                            symbol_data = data
                        else:
                            if symbol not in data.columns.get_level_values(0):
                                self.logger.warning(f"  ❌ {symbol}: Not in response data")
                                continue
                            symbol_data = data[symbol]
                        
                        if symbol_data.empty:
                            self.logger.warning(f"  ❌ {symbol}: Empty data frame")
                            continue
                        
                        available_dates = symbol_data.index.date
                        self.logger.info(f"  📅 Available dates: {[str(d) for d in available_dates]}")
                        
                        last_date   = symbol_data.index[-1].date()
                        last_close  = symbol_data['Close'].iloc[-1]
                        last_volume = symbol_data['Volume'].iloc[-1]
                        
                        self.logger.info(f"  📊 Last trading day: {last_date}")
                        self.logger.info(f"  💰 Last close: {'NaN' if pd.isna(last_close) else f'${last_close:.2f}'}")
                        self.logger.info(f"  📈 Last volume: {'NaN' if pd.isna(last_volume) else f'{last_volume:,}'}")
                        
                        days_diff = (target_date_obj - last_date).days
                        self.logger.info(f"  ⏱️  Age: {days_diff} days old")
                        
                        if days_diff > 5:
                            self.logger.warning(f"  ❌ REJECTED: Data is {days_diff} days old (limit: 5)")
                            continue
                        if pd.isna(last_close) or pd.isna(last_volume):
                            self.logger.warning(f"  ❌ REJECTED: NaN close or volume")
                            continue
                        if last_volume < self.min_volume:
                            self.logger.warning(f"  ❌ REJECTED: volume {last_volume:,} < {self.min_volume:,}")
                            continue
                        
                        # ── Opportunistically fill OHLC from this validation fetch ──
                        row = symbol_data.iloc[-1]
                        for field, col in (('high', 'High'), ('low', 'Low'),
                                           ('open', 'Open'), ('close', 'Close')):
                            if candidate.get(field) is None and not pd.isna(row.get(col, float('nan'))):
                                candidate[field] = float(row[col])
                        
                        self.logger.info(f"  ✅✅ {symbol} PASSED ALL VALIDATION CHECKS")
                        valid_candidates.append(candidate)
                        
                    except Exception as e:
                        self.logger.error(f"  ❌ {symbol}: Error: {e}", exc_info=True)
                        continue
                
                if i + batch_size < len(symbols):
                    time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"❌ Batch {i//batch_size + 1} error: {e}", exc_info=True)
                continue
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"📊 VALIDATION SUMMARY: "
                         f"total={len(candidates)}, "
                         f"passed={len(valid_candidates)}, "
                         f"rejected={len(candidates)-len(valid_candidates)}")
        self.logger.info("=" * 80 + "\n")
        
        return valid_candidates
    
    def _get_liquid_stocks_list(self) -> List[str]:
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'COST',
            'AVGO', 'KO', 'LLY', 'ADBE', 'WMT', 'MCD', 'CSCO', 'ACN', 'TMO', 'DIS',
            'ABT', 'DHR', 'VZ', 'NKE', 'NFLX', 'CRM', 'CMCSA', 'TXN', 'INTC', 'ORCL',
            'AMD', 'QCOM', 'HON', 'UNP', 'PM', 'NEE', 'RTX', 'UPS', 'LOW', 'INTU',
            'IBM', 'BA', 'AMGN', 'SPGI', 'GS', 'BLK', 'CAT', 'ELV', 'SBUX', 'DE',
            'AXP', 'ISRG', 'BKNG', 'GILD', 'ADI', 'TJX', 'MMC', 'MDLZ', 'VRTX', 'ADP',
            'CI', 'SYK', 'REGN', 'ZTS', 'PLD', 'AMT', 'DUK', 'SO', 'PGR', 'BDX',
            'MO', 'TGT', 'CL', 'USB', 'BMY', 'SCHW', 'CVS', 'CB', 'BSX', 'LRCX',
            'SLB', 'EOG', 'ITW', 'NOC', 'EQIX', 'MMM', 'C', 'PNC', 'EMR', 'AMAT',
        ]
