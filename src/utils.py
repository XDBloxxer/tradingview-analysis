"""
Utility functions for the analysis system
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import colorlog


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config.example.yaml to config.yaml and configure it."
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(level: str = "INFO", logging_config: Dict = None) -> logging.Logger:
    """
    Setup logging with color support
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        logging_config: Logging configuration dictionary
        
    Returns:
        Configured logger
    """
    if logging_config is None:
        logging_config = {}
    
    # Create logs directory if needed
    log_file = logging_config.get("file", "logs/analysis.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Plain formatter for file
    file_formatter = logging.Formatter(
        logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if logging_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def validate_indicators(config: Dict) -> bool:
    """
    Validate that required indicators are configured
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ["name", "column"]
    
    indicators = config.get("indicators", [])
    if not indicators:
        raise ValueError("No indicators configured in config.yaml")
    
    for indicator in indicators:
        for field in required_fields:
            if field not in indicator:
                raise ValueError(
                    f"Indicator missing required field '{field}': {indicator}"
                )
    
    return True


def get_indicator_mapping(config: Dict) -> Dict[str, str]:
    """
    Get mapping of indicator names to column names
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping indicator names to TradingView column names
    """
    return {
        indicator["name"]: indicator["column"]
        for indicator in config.get("indicators", [])
    }
