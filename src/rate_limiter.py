"""
Rate limiting functionality to avoid TradingView API throttling
"""

import time
import logging
from typing import Callable, Any
from functools import wraps


class RateLimiter:
    """
    Rate limiter for API requests with exponential backoff
    """
    
    def __init__(self, config: dict):
        """
        Initialize rate limiter
        
        Args:
            config: Configuration dictionary with rate_limiting settings
        """
        self.logger = logging.getLogger(__name__)
        
        rate_config = config.get("rate_limiting", {})
        
        self.requests_per_minute = rate_config.get("requests_per_minute", 30)
        self.delay_between_symbols = rate_config.get("delay_between_symbols", 2.0)
        self.max_retries = rate_config.get("max_retries", 3)
        self.retry_delay = rate_config.get("retry_delay", 5)
        self.exponential_backoff = rate_config.get("exponential_backoff", True)
        
        # Calculate minimum delay between requests
        self.min_delay = 60.0 / self.requests_per_minute
        
        # Track last request time
        self.last_request_time = 0
        self.request_count = 0
        
        self.logger.info(
            f"Rate limiter initialized: {self.requests_per_minute} req/min, "
            f"{self.delay_between_symbols}s between symbols"
        )
    
    def wait(self):
        """
        Wait appropriate time before next request
        """
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
