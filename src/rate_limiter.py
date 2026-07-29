"""
Rate limiting functionality to avoid TradingView API throttling
"""

import time
import logging
import threading
from typing import Callable, Any
from functools import wraps


class RateLimiter:
    """
    Rate limiter for API requests with exponential backoff
    """
    
    def __init__(self, config: dict, backfill_mode: bool = False):
        """
        Initialize rate limiter

        Args:
            config: Configuration dictionary with rate_limiting settings
            backfill_mode: When True, use the `backfill_rate_limiting`
                overrides (if present in config) layered on top of the
                base `rate_limiting` section. Backfills can just re-run
                whatever symbols failed, so there's no need to burn
                20-40 minutes patiently retrying one broken/delisted
                ticker the way a live unattended daily run should.
                Fail fast instead and move on.
        """
        self.logger = logging.getLogger(__name__)

        rate_config = dict(config.get("rate_limiting", {}))
        self.backfill_mode = backfill_mode
        if backfill_mode:
            overrides = config.get("backfill_rate_limiting", {})
            if overrides:
                self.logger.info(f"RateLimiter: backfill_mode=True, applying overrides: {overrides}")
            rate_config.update(overrides)

        self.requests_per_minute = rate_config.get("requests_per_minute", 30)
        self.delay_between_symbols = rate_config.get("delay_between_symbols", 2.0)
        self.max_retries = rate_config.get("max_retries", 3)
        self.retry_delay = rate_config.get("retry_delay", 5)
        self.exponential_backoff = rate_config.get("exponential_backoff", True)

        # Backoff ceiling + attempt cap for call_with_backoff. This is the
        # "keep retrying, don't just give up" path used for the actual
        # yfinance network calls during mass backfills. Uncapped-attempts
        # would risk a single permanently-broken symbol hanging a whole
        # backfill run forever, so this retries persistently through
        # transient rate-limit/connection blips (capped delay between
        # tries) but still eventually gives up and moves on if a symbol
        # is genuinely, consistently failing.
        #
        # In backfill_mode these come from backfill_rate_limiting instead
        # (fail fast — a handful of consistently-broken symbols shouldn't
        # each cost 20+ minutes of a backfill run; just skip and move on).
        self.max_backoff_attempts = rate_config.get("max_backoff_attempts", 10)
        self.max_backoff_delay = rate_config.get("max_backoff_delay", 300)  # 5 min ceiling per wait
        
        # Calculate minimum delay between requests
        self.min_delay = 60.0 / self.requests_per_minute
        
        # Track last request time
        self.last_request_time = 0
        self.request_count = 0

        # _process_winner runs across a ThreadPoolExecutor (multiple workers
        # sharing this same RateLimiter instance), and the detectors' scan
        # methods use even more worker threads (SCAN_WORKERS=20) -- wait()
        # and call_with_backoff() are called concurrently from many threads,
        # so the read-then-write on last_request_time needs a lock or the
        # pacing guarantee (requests_per_minute) is just wrong under
        # concurrency (multiple threads can all read a stale last_request_time
        # and all decide they're free to fire at once).
        self._lock = threading.Lock()
        
        self.logger.info(
            f"Rate limiter initialized: {self.requests_per_minute} req/min, "
            f"{self.delay_between_symbols}s between symbols"
        )
    
    def wait(self):
        """
        Wait appropriate time before next request. Thread-safe: the
        check-then-sleep-then-update sequence is done under a lock so
        concurrent callers can't all observe the same stale last_request_time
        and fire simultaneously.
        """
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            # Ensure minimum delay between requests
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                time.sleep(wait_time)
            
            self.last_request_time = time.time()
            self.request_count += 1
    
    def delay_between_symbols_wait(self):
        """
        Wait the configured delay between processing different symbols
        """
        time.sleep(self.delay_between_symbols)
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic and exponential backoff
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.wait()
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff
                    if self.exponential_backoff:
                        delay = self.retry_delay * (2 ** attempt)
                    else:
                        delay = self.retry_delay
                    
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Request failed after {self.max_retries} attempts: {str(e)}"
                    )
        
        raise last_exception

    def call_with_backoff(self, func: Callable, *args, label: str = "", **kwargs) -> Any:
        """
        Paced + resilient call for actual network requests (yfinance, etc.).

        Unlike with_retry (3 tries and give up — fine for quick in-process
        retries, but not enough for a multi-hour unattended backfill), this
        keeps retrying transient failures (HTTP 429s, timeouts, connection
        resets) with capped exponential backoff up to max_backoff_attempts,
        so a temporary rate-limit block from Yahoo doesn't silently drop a
        row of data for a symbol/date -- it waits it out and tries again.

        Still bounded (not truly infinite) so one permanently-broken symbol
        (bad ticker, delisted, etc.) can't hang an entire backfill run
        forever; after max_backoff_attempts it logs and re-raises so the
        caller's existing per-symbol try/except can skip just that symbol.

        Args:
            func: Callable to invoke (e.g. ticker.history)
            label: Optional human-readable label for logging (e.g. symbol)
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                self.wait()
                return func(*args, **kwargs)
            except Exception as e:
                if attempt >= self.max_backoff_attempts:
                    self.logger.error(
                        f"{label + ': ' if label else ''}giving up after {attempt} "
                        f"attempts — {e}"
                    )
                    raise
                if self.exponential_backoff:
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), self.max_backoff_delay)
                else:
                    delay = min(self.retry_delay, self.max_backoff_delay)
                self.logger.warning(
                    f"{label + ': ' if label else ''}request failed "
                    f"(attempt {attempt}/{self.max_backoff_attempts}): {e}. "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)


def rate_limited(rate_limiter: RateLimiter):
    """
    Decorator to apply rate limiting to a function
    
    Args:
        rate_limiter: RateLimiter instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return rate_limiter.with_retry(func, *args, **kwargs)
        return wrapper
    return decorator
