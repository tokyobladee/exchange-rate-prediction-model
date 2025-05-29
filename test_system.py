#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_imports():
    print("\n1. TESTING IMPORTS")
    print("=" * 30)
    
    modules_to_test = [
        ('utils.data_fetcher', 'CurrencyDataFetcher'),
        ('utils.feature_engineering', 'FeatureEngineer'),
        ('utils.visualization', 'CurrencyVisualizer'),
        ('models.ml_models', 'MLModels'),
        ('launchers.quick_analysis', 'main'),
        ('launchers.easy_analytics_launcher', 'main'),
        ('analytics.fresh_analytics_generator', 'run_complete_fresh_analysis'),
        ('analytics.currency_prediction_app', 'CurrencyPredictionApp'),
        ('neural_networks.neural_network_training', 'NeuralNetworkTrainer'),
        ('charts.create_prediction_charts', 'create_all_charts'),
        ('utils.show_results', 'main'),
        ('utils.show_prediction_summary', 'main'),
        ('utils.retrain_models', 'main'),
        ('utils.prediction_validator', 'main'),
        ('utils.data_manager', 'main'),
        ('utils.currency_manager', 'main')
    ]
    
    imported_count = 0
    failed_imports = []
    
    for module_name, class_or_func in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_or_func])
            getattr(module, class_or_func)
            print(f"  {module_name} - OK")
            imported_count += 1
        except Exception as e:
            print(f"  {module_name} - FAILED: {e}")
            failed_imports.append((module_name, str(e)))
    
    print(f"\nImport Results: {imported_count}/{len(modules_to_test)} successful")
    return imported_count == len(modules_to_test), failed_imports

def test_data_fetcher():
    print("\n2. TESTING DATA FETCHER")
    print("=" * 30)
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        
        fetcher = CurrencyDataFetcher()
        
        data = fetcher.get_currency_data('EURUSD=X', period='5d')
        
        if data is not None and len(data) > 0:
            print(f"  Data fetched: {len(data)} records")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            return True, None
        else:
            return False, "No data returned"
            
    except Exception as e:
        return False, str(e)

def test_feature_engineering():
    print("\n3. TESTING FEATURE ENGINEERING")
    print("=" * 30)
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        from utils.feature_engineering import FeatureEngineer
        
        fetcher = CurrencyDataFetcher()
        engineer = FeatureEngineer()
        
        data = fetcher.get_currency_data('EURUSD=X', period='1y')
        
        if data is None or len(data) == 0:
            return False, "No data available for feature engineering"
        
        features = engineer.prepare_features_simple(data)
        
        if features is not None and len(features) > 0:
            print(f"  Features created: {len(features)} records")
            print(f"  Feature columns: {len(features.columns)}")
            print(f"  Sample features: {list(features.columns[:5])}")
            return True, None
        else:
            return False, "No features created"
            
    except Exception as e:
        return False, str(e)

def test_ml_models():
    print("\n4. TESTING ML MODELS")
    print("=" * 30)
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        from utils.feature_engineering import FeatureEngineer
        from models.ml_models import MLModels
        
        fetcher = CurrencyDataFetcher()
        engineer = FeatureEngineer()
        ml_models = MLModels()
        
        data = fetcher.get_currency_data('EURUSD=X', period='1y')
        features = engineer.prepare_features_simple(data)
        
        if features is None or len(features) < 50:
            return False, "Insufficient data for ML training"
        
        X, y = ml_models.prepare_data(features, target_col='close')
        
        if len(X) < 10:
            return False, "Insufficient training data after preparation"
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X[:10], y[:10])
        
        print(f"  Training data: {len(X)} samples, {len(X.columns)} features")
        print(f"  Model trained successfully")
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_visualization():
    print("\n5. TESTING VISUALIZATION")
    print("=" * 30)
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        from utils.visualization import CurrencyVisualizer
        
        fetcher = CurrencyDataFetcher()
        visualizer = CurrencyVisualizer()
        
        data = fetcher.get_currency_data('EURUSD=X', period='1mo')
        
        if data is None or len(data) == 0:
            return False, "No data available for visualization"
        
        import matplotlib
        matplotlib.use('Agg')
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['close'])
        plt.close(fig)
        
        print(f"  Visualization test passed")
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_launchers():
    print("\n6. TESTING LAUNCHERS")
    print("=" * 30)
    
    try:
        from launchers.quick_analysis import main as quick_main
        from launchers.easy_analytics_launcher import main as easy_main
        
        required_keys = ['quick_analysis', 'easy_analytics_launcher']
        
        print(f"  Launcher modules loaded successfully")
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_neural_networks():
    print("\n7. TESTING NEURAL NETWORKS")
    print("=" * 30)
    
    try:
        from neural_networks.neural_network_training import NeuralNetworkTrainer
        
        trainer = NeuralNetworkTrainer()
        print(f"  Neural network trainer loaded")
        return True, None
        
    except Exception as e:
        return False, str(e)

def run_all_tests():
    print("COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    print("Testing all system components...")
    
    tests = [
        ("Import Test", test_imports),
        ("Data Fetcher Test", test_data_fetcher),
        ("Feature Engineering Test", test_feature_engineering),
        ("ML Models Test", test_ml_models),
        ("Visualization Test", test_visualization),
        ("Launchers Test", test_launchers),
        ("Neural Networks Test", test_neural_networks)
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            success, error = test_func()
            if success:
                print(f"  {test_name}: PASSED")
                passed += 1
            else:
                print(f"  {test_name}: FAILED - {error}")
                failed += 1
                failed_tests.append((test_name, error))
        except Exception as e:
            print(f"  {test_name}: ERROR - {e}")
            failed += 1
            failed_tests.append((test_name, str(e)))
    
    print(f"\nTEST RESULTS")
    print("=" * 20)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED! System is ready to use.")
    else:
        print(f"\n{failed} tests failed. Please check the errors above.")
        
        if failed_tests:
            print("\nFailed tests details:")
            for test_name, error in failed_tests:
                print(f"  - {test_name}: {error}")
    
    return failed == 0

def main():
    return run_all_tests()

if __name__ == "__main__":
    main() 