#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from analytics.currency_prediction_app import CurrencyPredictionApp
from models.pytorch_models import PyTorchCurrencyModels

def test_neural_networks():
    
    print("TESTING NEURAL NETWORK MODELS")
    print("=" * 40)
    
    app = CurrencyPredictionApp()
    pytorch_models = PyTorchCurrencyModels()
    
    app.fetch_currency_data(['EURUSD=X'], '1y')
    app.engineer_features()
    
    data = app.processed_data['EURUSD=X']
    
    print(f"Training neural networks on {len(data)} records...")
    
    performance = pytorch_models.train_all_models(data, sequence_length=10)
    
    print("\nNeural Network Performance:")
    print("-" * 40)
    for model_name, metrics in performance.items():
        print(f"{model_name:15s} | RMSE: {metrics['rmse']:.6f} | R²: {metrics['r2']:.4f}")
    
    print(f"\nBest Neural Network: {pytorch_models.best_model_name}")
    print(f"Best RMSE: {performance[pytorch_models.best_model_name]['rmse']:.6f}")
    print(f"Best R²: {performance[pytorch_models.best_model_name]['r2']:.4f}")
    
    pytorch_models.save_models('models/test_neural_models.pth')
    print("\nModels saved to models/test_neural_models.pth")
    
    return performance

def main():
    test_neural_networks()

if __name__ == "__main__":
    main() 
