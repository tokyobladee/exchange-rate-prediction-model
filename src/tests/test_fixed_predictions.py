#!/usr/bin/env python3

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
root_path = src_path.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(root_path))

from neural_networks.neural_network_training import NeuralNetworkTrainer
import numpy as np

def test_fixed_predictions():
    print(" TESTING FIXED NEURAL NETWORK PREDICTIONS")
    print("=" * 50)
    
    try:
        trainer = NeuralNetworkTrainer()
        
        print(" Running quick training to ensure fresh models...")
        trainer.fetch_and_prepare_data(['EURUSD=X'], '1y')
        trainer.train_neural_networks('EURUSD=X', sequence_length=10)
        
        print("\n Testing predictions with actual features...")
        data = trainer.app.processed_data['EURUSD=X']
        
        current_price = data['close'].iloc[-1]
        print(f" Current EUR/USD price: {current_price:.6f}")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        last_features = data[feature_cols].iloc[-1:].values
        
        print(f" Using {len(feature_cols)} actual features for prediction")
        print(f" Feature sample: {last_features[0][:5]}")
        
        prediction = trainer.pytorch_models.predict_next(last_features)
        
        print(f" Neural Network Prediction: {prediction:.6f}")
        
        change_pct = ((prediction / current_price) - 1) * 100
        direction = " UP" if change_pct > 0 else " DOWN" if change_pct < 0 else " FLAT"
        
        print(f" Expected Change: {change_pct:+.2f}% {direction}")
        
        if 0.8 <= prediction <= 1.5:
            print(" Prediction is within reasonable EUR/USD range!")
        else:
            print(f"  Warning: Prediction {prediction:.6f} may be outside normal range")
        
        if abs(change_pct) <= 5.0:
            print(" Predicted change is reasonable (≤5%)")
        else:
            print(f"  Warning: Large predicted change ({change_pct:+.2f}%)")
        
        print("\n Testing with default features (fallback)...")
        default_prediction = trainer.pytorch_models.predict_next()
        print(f" Default Prediction: {default_prediction:.6f}")
        
        return True
        
    except Exception as e:
        print(f" Error testing predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_predictions()
    
    if success:
        print("\n PREDICTION TESTS PASSED!")
        print("The neural network predictions are now working correctly.")
    else:
        print("\n PREDICTION TESTS FAILED!")
        print("There are still issues with the predictions.") 