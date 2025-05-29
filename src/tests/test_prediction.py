#!/usr/bin/env python3

from neural_network_training import NeuralNetworkTrainer

def test_prediction():
    print(" Testing Neural Network Predictions")
    print("=" * 40)
    
    try:
        trainer = NeuralNetworkTrainer()
        trainer.pytorch_models.load_models('models/EURUSD=X_pytorch_models.pth')
        
        prediction = trainer.pytorch_models.predict_next()
        
        print(f" Neural Network Prediction: {prediction:.6f}")
        print(f" Best Model: {trainer.pytorch_models.best_model_name}")
        print(f" Model Performance: R² = {trainer.pytorch_models.model_performance[trainer.pytorch_models.best_model_name]['r2']:.4f}")
        
        return prediction
        
    except Exception as e:
        print(f" Error: {e}")
        return None

if __name__ == "__main__":
    test_prediction() 