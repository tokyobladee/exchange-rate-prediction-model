#!/usr/bin/env python3

import sys
import warnings
from pathlib import Path

src_path = Path(__file__).parent.parent
root_path = src_path.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(root_path))

warnings.filterwarnings('ignore')

def test_imports():
    
    print("Testing imports...")
    
    try:
        import pandas as pd
        print(f" pandas {pd.__version__}")
    except ImportError as e:
        print(f" pandas: {e}")
        return False
    
    try:
        import numpy as np
        print(f" numpy {np.__version__}")
    except ImportError as e:
        print(f" numpy: {e}")
        return False
    
    try:
        import yfinance as yf
        print(f" yfinance")
    except ImportError as e:
        print(f" yfinance: {e}")
        return False
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        print(" CurrencyDataFetcher")
    except ImportError as e:
        print(f" CurrencyDataFetcher: {e}")
        return False
    
    try:
        from utils.feature_engineering import FeatureEngineer
        print(" FeatureEngineer")
    except ImportError as e:
        print(f" FeatureEngineer: {e}")
        return False
    
    try:
        from utils.visualization import CurrencyVisualization
        print(" CurrencyVisualization")
    except ImportError as e:
        print(f" CurrencyVisualization: {e}")
        return False
    
    try:
        from models.ml_models import CurrencyPredictionModels
        print(" CurrencyPredictionModels")
    except ImportError as e:
        print(f" CurrencyPredictionModels: {e}")
        return False
    
    return True

def test_data_fetcher():
    
    print("\nTesting data fetcher...")
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        
        fetcher = CurrencyDataFetcher()
        pairs = fetcher.list_supported_pairs()
        print(f" Found {len(pairs)} supported currency pairs")
        
        print("Fetching sample EUR/USD data...")
        data = fetcher.get_currency_data('EURUSD=X', period='5d', interval='1d')
        
        if not data.empty:
            print(f" Successfully fetched {len(data)} records")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            return True
        else:
            print(" No data returned")
            return False
            
    except Exception as e:
        print(f" Data fetcher error: {e}")
        return False

def test_feature_engineering():
    
    print("\nTesting feature engineering...")
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        from utils.feature_engineering import FeatureEngineer
        
        fetcher = CurrencyDataFetcher()
        data = fetcher.get_currency_data('EURUSD=X', period='1mo', interval='1d')
        
        if data.empty:
            print(" No data for feature engineering test")
            return False
        
        engineer = FeatureEngineer()
        
        data_with_indicators = engineer.add_technical_indicators(data)
        print(f" Added technical indicators. Shape: {data_with_indicators.shape}")
        
        data_with_features = engineer.add_price_features(data_with_indicators)
        print(f" Added price features. Shape: {data_with_features.shape}")
        
        return True
        
    except Exception as e:
        print(f" Feature engineering error: {e}")
        return False

def main():
    
    print(" CURRENCY PREDICTION SYSTEM - BASIC TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_data_fetcher():
        tests_passed += 1
    
    if test_feature_engineering():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print(" All tests passed! The system is ready to use.")
        print("\nTo run the full analysis, execute:")
        print("  python currency_prediction_app.py")
        print("\nTo see examples, execute:")
        print("  python example_usage.py")
    else:
        print(" Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 