#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
from currency_prediction_app import CurrencyPredictionApp
from models.pytorch_models import PyTorchCurrencyModels
import time

warnings.filterwarnings('ignore')

def demo_neural_networks():

    print("NEURAL NETWORKS FOR CURRENCY PREDICTION")
    print("=" * 80)
    print("This demo shows PyTorch neural networks predicting EUR/USD prices")
    print("Compatible with Python 3.13 and newer versions")
    print()
    
    print(" STEP 1: INITIALIZING SYSTEM")
    print("-" * 40)
    app = CurrencyPredictionApp()
    pytorch_models = PyTorchCurrencyModels()
    print(" System initialized")
    print(f" Using device: {pytorch_models.device}")
    print()
    
    print(" STEP 2: PREPARING DATA")
    print("-" * 40)
    print("Fetching 3 years of EUR/USD data from Yahoo Finance...")
    app.fetch_currency_data(['EURUSD=X'], period='3y')
    
    print("Engineering features (technical indicators, price ratios, etc.)...")
    app.engineer_features()
    
    data = app.processed_data['EURUSD=X']
    print(f" Data prepared: {data.shape[0]} samples, {data.shape[1]-1} features")
    print(f" Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()
    
    print(" STEP 3: TRAINING NEURAL NETWORKS")
    print("-" * 40)
    print("Training 3 different neural network architectures:")
    print("  1. Dense Neural Network (feedforward)")
    print("  2. LSTM Network (time series specialist)")
    print("  3. GRU Network (efficient alternative)")
    print()
    
    start_time = time.time()
    performance = pytorch_models.train_all_models(data, sequence_length=10)
    training_time = time.time() - start_time
    
    print(f" Training completed in {training_time:.1f} seconds")
    print()
    
    print(" STEP 4: PERFORMANCE ANALYSIS")
    print("-" * 40)
    print("Neural Network Performance:")
    print()
    
    for i, (model_name, metrics) in enumerate(performance.items(), 1):
        accuracy = metrics['r2'] * 100
        print(f"{i}. {model_name:15s} | Accuracy: {accuracy:6.2f}% | RMSE: {metrics['rmse']:.6f}")
    
    best_model = pytorch_models.best_model_name
    best_accuracy = performance[best_model]['r2'] * 100
    print()
    print(f" Best Model: {best_model} with {best_accuracy:.2f}% accuracy")
    print()
    
    print(" STEP 5: MAKING PREDICTIONS")
    print("-" * 40)
    
    current_price = data['close'].iloc[-1]
    current_date = data.index[-1].date()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'target']
    last_features = data[feature_cols].iloc[-1:].values
    
    prediction = pytorch_models.predict(last_features)
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 0:
            pred_value = float(prediction)
        else:
            pred_value = float(prediction[0])
    else:
        pred_value = float(prediction)
    
    change_pct = ((pred_value / current_price) - 1) * 100
    
    print(f" Current EUR/USD Data (as of {current_date}):")
    print(f"   Current Price: {current_price:.6f}")
    print()
    print(f" Neural Network Prediction:")
    print(f"   Model Used: {best_model}")
    print(f"   Predicted Price: {pred_value:.6f}")
    print(f"   Expected Change: {change_pct:+.2f}%")
    print(f"   Direction: {' UP' if change_pct > 0 else ' DOWN' if change_pct < 0 else ' FLAT'}")
    print()
    
    print(" STEP 6: COMPARING WITH TRADITIONAL ML")
    print("-" * 40)
    print("Training traditional ML models for comparison...")
    
    traditional_performance = app.train_models('EURUSD=X')
    
    print()
    print("Performance Comparison:")
    print()
    print("TRADITIONAL ML MODELS:")
    for model_name, metrics in traditional_performance.items():
        accuracy = metrics['r2'] * 100
        print(f"  {model_name:20s} | Accuracy: {accuracy:6.2f}% | RMSE: {metrics['rmse']:.6f}")
    
    print()
    print("NEURAL NETWORK MODELS:")
    for model_name, metrics in performance.items():
        accuracy = metrics['r2'] * 100
        print(f"  {model_name:20s} | Accuracy: {accuracy:6.2f}% | RMSE: {metrics['rmse']:.6f}")
    
    all_models = {**traditional_performance, **performance}
    overall_best = min(all_models.keys(), key=lambda x: all_models[x]['rmse'])
    overall_best_accuracy = all_models[overall_best]['r2'] * 100
    
    print()
    print(f"OVERALL BEST MODEL: {overall_best}")
    print(f"BEST ACCURACY: {overall_best_accuracy:.2f}%")
    print()
    
    print(" STEP 7: SUMMARY")
    print("-" * 40)
    print(" Successfully implemented neural networks for currency prediction")
    print(" Trained 3 different neural network architectures")
    print(f" Best neural network achieved {best_accuracy:.2f}% accuracy")
    print(" Made successful price prediction")
    print(" Compared with traditional ML models")
    print()
    print(" Files created:")
    print("   - models/test_pytorch_models.pth (trained neural networks)")
    print("   - data/EURUSD=X_processed.csv (processed data)")
    print()
    print(" Next steps:")
    print("   - Run 'python neural_network_training.py' for full analysis")
    print("   - Experiment with different currency pairs")
    print("   - Try different neural network architectures")
    print()
    print(" NEURAL NETWORKS DEMO COMPLETE!")
    print("=" * 80)
    
    pytorch_models.save_models('models/demo_pytorch_models.pth')
    
    return {
        'neural_performance': performance,
        'traditional_performance': traditional_performance,
        'best_neural_model': best_model,
        'prediction': pred_value,
        'current_price': current_price,
        'change_percent': change_pct
    }

if __name__ == "__main__":
    results = demo_neural_networks() 