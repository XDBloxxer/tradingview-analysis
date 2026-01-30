#!/usr/bin/env python3
"""
Health check script to verify system setup
"""

import sys
from pathlib import Path
import yaml

def check_python_version():
    """Check Python version"""
    print("Checking Python version...", end=" ")
    if sys.version_info >= (3, 8):
        print(f"✓ {sys.version.split()[0]}")
        return True
    else:
        print(f"✗ {sys.version.split()[0]} (Need 3.8+)")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...", end=" ")
    
    required = [
        'tradingview_scraper',
        'pandas',
        'gspread',
        'yaml',
        'pytest'
    ]
    
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if not missing:
        print(f"✓ All dependencies installed")
        return True
    else:
        print(f"✗ Missing: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

def check_config():
    """Check configuration file"""
    print("Checking configuration...", end=" ")
    
    if not Path('config.yaml').exists():
        print("✗ config.yaml not found")
        print("  Run: cp config.example.yaml config.yaml")
        return False
    
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = [
            'google_sheets',
            'detection',
            'indicators',
            'time_lags',
            'rate_limiting'
        ]
        
        missing = [f for f in required_fields if f not in config]
        
        if missing:
            print(f"✗ Missing fields: {', '.join(missing)}")
            return False
        
        print("✓ Valid")
        return True
        
    except Exception as e:
        print(f"✗ Error reading config: {str(e)}")
        return False

def check_credentials():
    """Check Google Sheets credentials"""
    print("Checking credentials...", end=" ")
    
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        
        creds_path = config.get('google_sheets', {}).get('credentials_path')
        
        if not creds_path:
            print("✗ credentials_path not set in config.yaml")
            return False
        
        if not Path(creds_path).exists():
            print(f"✗ {creds_path} not found")
            print("  Download from Google Cloud Console")
            return False
        
        # Try to load the JSON
        import json
        with open(creds_path) as f:
            creds = json.load(f)
        
        required_keys = [
            'type',
            'project_id',
            'private_key',
            'client_email'
        ]
        
        missing = [k for k in required_keys if k not in creds]
        
        if missing:
            print(f"✗ Invalid credentials (missing: {', '.join(missing)})")
            return False
        
        print(f"✓ Found")
        print(f"  Service account: {creds['client_email']}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def check_directories():
    """Check required directories"""
    print("Checking directories...", end=" ")
    
    required_dirs = [
        'src',
        'tests',
        'credentials',
        'data',
        'logs'
    ]
    
    missing = [d for d in required_dirs if not Path(d).exists()]
    
    if missing:
        print(f"✗ Missing: {', '.join(missing)}")
        print("  Creating missing directories...")
        for d in missing:
            Path(d).mkdir(parents=True, exist_ok=True)
        return True
    else:
        print("✓ All present")
        return True

def test_google_sheets_connection():
    """Test connection to Google Sheets"""
    print("Testing Google Sheets connection...", end=" ")
    
    try:
        from src.sheets_writer import SheetsWriter
        
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        
        writer = SheetsWriter(config)
        
        # Try to get spreadsheet title
        title = writer.spreadsheet.title
        
        print(f"✓ Connected to '{title}'")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        print("  Make sure you shared the sheet with service account")
        return False

def test_tradingview_scraper():
    """Test TradingView scraper"""
    print("Testing TradingView scraper...", end=" ")
    
    try:
        from tradingview_scraper.symbols.technicals import Indicators
        
        # Try a simple query
        scanner = Indicators(
            exchange='NASDAQ',
            symbol='AAPL',
            indicators=['close'],
            export_result=False
        )
        
        result = scanner.scrape()
        
        if result and 'data' in result:
            print("✓ Working")
            return True
        else:
            print("✗ No data returned")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    """Run all health checks"""
    print("=" * 60)
    print("TradingView Analysis System - Health Check")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Credentials", check_credentials),
        ("Directories", check_directories),
        ("Google Sheets", test_google_sheets_connection),
        ("TradingView", test_tradingview_scraper),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"✗ Unexpected error: {str(e)}")
            results[name] = False
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print()
        print("🎉 All checks passed! System is ready.")
        print()
        print("Next steps:")
        print("  1. Review config.yaml settings")
        print("  2. Run: python main.py --detect --verbose")
        print("  3. Check Google Sheets for results")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Copy config: cp config.example.yaml config.yaml")
        print("  - Download credentials from Google Cloud Console")
        print("  - Share Google Sheet with service account email")
        return 1

if __name__ == '__main__':
    sys.exit(main())
