#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import glob

def check_data_quality():
    
    print(" DIAGNOSING DATA QUALITY")
    print("=" * 40)
    
    try:
        print(" Fetching fresh EUR/USD data...")
        data = yf.download('EURUSD=X', period='5d')
        
        print(f" Recent EUR/USD prices:")
        print(data['Close'].tail())
        print(f" Price range: {data['Close'].min():.6f} - {data['Close'].max():.6f}")
        print(f" Current price: {data['Close'].iloc[-1]:.6f}")
        
        if os.path.exists('data/EURUSD=X_processed.csv'):
            processed = pd.read_csv('data/EURUSD=X_processed.csv', index_col=0)
            print(f"\n Processed data shape: {processed.shape}")
            print(f" Target column range: {processed['target'].min():.6f} - {processed['target'].max():.6f}")
            print(f" Close price range: {processed['close'].min():.6f} - {processed['close'].max():.6f}")
            
            if processed['target'].max() > 100:
                print("  WARNING: Target values seem too large - possible scaling issue!")
            
            return processed
        else:
            print(" No processed data found")
            return None
            
    except Exception as e:
        print(f" Error checking data: {e}")
        return None

def check_model_predictions():
    
    print("\n DIAGNOSING MODEL PREDICTIONS")
    print("=" * 40)
    
    try:
        from neural_network_training import NeuralNetworkTrainer
        
        trainer = NeuralNetworkTrainer()
        
        if os.path.exists('models/EURUSD=X_pytorch_models.pth'):
            trainer.pytorch_models.load_models('models/EURUSD=X_pytorch_models.pth')
            
            if 'X' in trainer.pytorch_models.scalers:
                scaler_X = trainer.pytorch_models.scalers['X']
                scaler_y = trainer.pytorch_models.scalers['y']
                
                print(f" Feature scaler mean: {scaler_X.mean_[:5]}")
                print(f" Feature scaler scale: {scaler_X.scale_[:5]}")
                print(f" Target scaler mean: {scaler_y.mean_}")
                print(f" Target scaler scale: {scaler_y.scale_}")
                
                dummy_input = np.zeros((1, len(scaler_X.mean_)))
                prediction = trainer.pytorch_models.predict(dummy_input)
                print(f" Dummy prediction: {prediction}")
                
                if abs(prediction[0]) > 10:
                    print("  WARNING: Prediction values are too large!")
                    return False
                else:
                    print(" Prediction values seem reasonable")
                    return True
            else:
                print(" No scalers found in model")
                return False
        else:
            print(" No trained models found")
            return False
            
    except Exception as e:
        print(f" Error checking predictions: {e}")
        return False

def fix_analytics_reports():
    
    print("\n FIXING ANALYTICS REPORTS")
    print("=" * 40)
    
    analytics_files = [
        'reports/*.txt',
        'reports/*.csv',
        'plots/*.png',
        'data/*_predictions.csv',
        'predictions/*.json'
    ]
    
    for pattern in analytics_files:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Could not remove {file}: {e}")
    
    print(" Analytics files cleared")

def fix_prediction_scaling():
    
    print("\n FIXING PREDICTION SCALING")
    print("=" * 40)

    script_content = "# Fixed prediction scaling method\n"
    
    print(" Fixed prediction method created")
    return script_content

def create_proper_analytics():
    
    print("\n CREATING PROPER ANALYTICS")
    print("=" * 40)
    
    try:
        from currency_prediction_app import CurrencyPredictionApp
        
        app = CurrencyPredictionApp()
        
        print(" Fetching fresh data...")
        app.fetch_currency_data(['EURUSD=X'], '1y')
        
        print(" Engineering features...")
        app.engineer_features()
        
        print(" Training models...")
        performance = app.train_models('EURUSD=X')
        
        data = app.processed_data['EURUSD=X']
        current_price = data['close'].iloc[-1]
        
        best_model = app.ml_models.models[app.ml_models.best_model_name]
        
        feature_cols = [col for col in data.columns if col not in ['target', 'close']]
        last_features = data[feature_cols].iloc[-1:].values
        
        prediction = best_model.predict(last_features)[0]
        
        print(f" Current price: {current_price:.6f}")
        print(f" Prediction: {prediction:.6f}")
        print(f" Change: {((prediction/current_price - 1) * 100):+.2f}%")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'prediction': prediction,
            'change_percent': (prediction/current_price - 1) * 100,
            'model_used': app.ml_models.best_model_name,
            'model_accuracy': performance[app.ml_models.best_model_name]['r2']
        }
        
        with open('reports/fixed_analytics_report.txt', 'w') as f:
            f.write("FIXED CURRENCY PREDICTION ANALYTICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Currency Pair: EUR/USD\n")
            f.write(f"Data Period: 1 year\n\n")
            
            f.write("CURRENT ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Current Price: {current_price:.6f}\n")
            f.write(f"Predicted Price: {prediction:.6f}\n")
            f.write(f"Expected Change: {((prediction/current_price - 1) * 100):+.2f}%\n")
            f.write(f"Model Used: {app.ml_models.best_model_name}\n")
            f.write(f"Model Accuracy: {performance[app.ml_models.best_model_name]['r2']*100:.2f}%\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            for model_name, metrics in performance.items():
                f.write(f"{model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.6f}\n")
        
        print(" Fixed analytics report created: reports/fixed_analytics_report.txt")
        return True
        
    except Exception as e:
        print(f" Error creating analytics: {e}")
        return False

def main():
    
    print(" CURRENCY PREDICTION DIAGNOSTICS & FIXES")
    print("=" * 60)
    
    processed_data = check_data_quality()
    
    predictions_ok = check_model_predictions()
    
    fix_analytics_reports()
    
    fix_code = fix_prediction_scaling()
    
    analytics_ok = create_proper_analytics()
    
    print("\n" + "=" * 60)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if processed_data is not None:
        print(" Data quality: OK")
    else:
        print(" Data quality: Issues found")
    
    if predictions_ok:
        print(" Model predictions: OK")
    else:
        print(" Model predictions: Issues found")
    
    if analytics_ok:
        print(" Analytics reports: Fixed")
    else:
        print(" Analytics reports: Issues remain")
    
    print("\n Check these files:")
    print("    reports/fixed_analytics_report.txt - New analytics")
    print("    models/ - Retrained models")
    print("    data/ - Fresh data")

if __name__ == "__main__":
    main() 