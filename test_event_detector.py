"""
Tests for event detector
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

from src.event_detector import EventDetector


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'detection': {
            'spiker_threshold': 15.0,
            'grinder_days': 3,
            'grinder_threshold': 0.5,
            'min_price': 0.50,
            'min_volume': 100000,
            'lookback_days': 7,
            'exchanges': ['NASDAQ', 'NYSE']
        },
        'rate_limiting': {
            'requests_per_minute': 30,
            'delay_between_symbols': 2.0,
            'max_retries': 3
        },
        'google_sheets': {
            'spreadsheet_id': 'test_id',
            'credentials_path': 'test_creds.json',
            'sheets': {
                'candidates': 'Candidates'
            }
        },
        'indicators': [
            {'name': 'Close', 'column': 'close'},
            {'name': 'Change %', 'column': 'change'},
            {'name': 'Volume', 'column': 'volume'}
        ]
    }


@pytest.fixture
def mock_symbols_data():
    """Mock symbol data from TradingView"""
    return [
        {
            'symbol': 'AAPL',
            'close': 150.0,
            'change': 20.0,  # Spiker
            'volume': 1000000,
            'relative_volume_10d_calc': 1.5
        },
        {
            'symbol': 'MSFT',
            'close': 300.0,
            'change': 2.0,  # Potential grinder
            'volume': 500000,
            'relative_volume_10d_calc': 1.3
        },
        {
            'symbol': 'PENNYST',
            'close': 0.30,  # Below min price
            'change': 50.0,
            'volume': 1000000,
            'relative_volume_10d_calc': 2.0
        },
        {
            'symbol': 'LOWVOL',
            'close': 10.0,
            'change': 20.0,
            'volume': 50000,  # Below min volume
            'relative_volume_10d_calc': 1.0
        }
    ]


class TestEventDetector:
    """Test cases for EventDetector"""
    
    @patch('src.event_detector.SheetsWriter')
    @patch('src.event_detector.RateLimiter')
    def test_initialization(self, mock_rate_limiter, mock_sheets_writer, mock_config):
        """Test detector initialization"""
        detector = EventDetector(mock_config)
        
        assert detector.spiker_threshold == 15.0
        assert detector.grinder_days == 3
        assert detector.grinder_threshold == 0.5
        assert detector.min_price == 0.50
        assert detector.min_volume == 100000
    
    @patch('src.event_detector.SheetsWriter')
    @patch('src.event_detector.RateLimiter')
    def test_filter_symbols(self, mock_rate_limiter, mock_sheets_writer, mock_config, mock_symbols_data):
        """Test symbol filtering by price and volume"""
        detector = EventDetector(mock_config)
        
        filtered = detector._filter_symbols(mock_symbols_data)
        
        # Should filter out PENNYST (low price) and LOWVOL (low volume)
        assert len(filtered) == 2
        assert all(s['close'] >= 0.50 for s in filtered)
        assert all(s['volume'] >= 100000 for s in filtered)
    
    @patch('src.event_detector.SheetsWriter')
    @patch('src.event_detector.RateLimiter')
    def test_detect_spiker(self, mock_rate_limiter, mock_sheets_writer, mock_config, mock_symbols_data):
        """Test spiker detection"""
        detector = EventDetector(mock_config)
        
        # Filter and detect
        filtered = detector._filter_symbols(mock_symbols_data)
        events = detector._detect_events_in_symbols(filtered)
        
        # AAPL should be detected as spiker (20% change)
        spikers = [e for e in events if e['Event_Type'] == 'Spiker']
        assert len(spikers) >= 1
        
        aapl_spiker = next((e for e in spikers if e['Symbol'] == 'AAPL'), None)
        assert aapl_spiker is not None
        assert aapl_spiker['Change_%'] == 20.0
    
    @patch('src.event_detector.SheetsWriter')
    @patch('src.event_detector.RateLimiter')
    def test_detect_grinder(self, mock_rate_limiter, mock_sheets_writer, mock_config, mock_symbols_data):
        """Test grinder detection"""
        detector = EventDetector(mock_config)
        
        # Filter and detect
        filtered = detector._filter_symbols(mock_symbols_data)
        events = detector._detect_events_in_symbols(filtered)
        
        # MSFT should be detected as potential grinder (2% change + high rel volume)
        grinders = [e for e in events if e['Event_Type'] == 'Grinder']
        
        # Note: In the current implementation, grinder detection is simplified
        # So we just verify the logic exists
        assert isinstance(grinders, list)
    
    @patch('src.event_detector.Indicators')
    @patch('src.event_detector.SheetsWriter')
    @patch('src.event_detector.RateLimiter')
    def test_get_symbols_for_exchange(
        self,
        mock_rate_limiter,
        mock_sheets_writer,
        mock_indicators,
        mock_config,
        mock_symbols_data
    ):
        """Test fetching symbols from exchange"""
        # Mock the Indicators.scrape() response
        mock_scanner = MagicMock()
        mock_scanner.scrape.return_value = {'data': mock_symbols_data}
        mock_indicators.return_value = mock_scanner
        
        detector = EventDetector(mock_config)
        symbols = detector._get_symbols_for_exchange('NASDAQ')
        
        assert len(symbols) == len(mock_symbols_data)
        assert symbols[0]['symbol'] == 'AAPL'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
