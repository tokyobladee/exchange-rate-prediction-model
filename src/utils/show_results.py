#!/usr/bin/env python3

import pandas as pd
import joblib

def show_ml_results():
    print("MACHINE LEARNING RESULTS")
    print("=" * 50)

    try:
        models_data = joblib.load('models/EURUSD=X_models.joblib')
        
        print(f"Trained {len(models_data['models'])} ML Models:")
        print("-" * 30)
        
        for name, perf in models_data['performance'].items():
            print(f"{name}:")
            print(f"  RMSE: {perf['rmse']:.6f}")
            print(f"  MAE:  {perf['mae']:.6f}")
            print(f"  R²:   {perf['r2']:.4f}")
            print()
        
        print(f"Best Model: {models_data['best_model_name']}")
        best_rmse = models_data['performance'][models_data['best_model_name']]['rmse']
        best_r2 = models_data['performance'][models_data['best_model_name']]['r2']
        print(f"Best Performance: RMSE={best_rmse:.6f}, R²={best_r2:.4f}")
        
    except Exception as e:
        print(f"Error loading models: {e}")

def show_predictions():
    print("\n" + "=" * 50)
    print("FUTURE PREDICTIONS")
    print("=" * 50)

    try:
        predictions = pd.read_csv('data/EURUSD=X_predictions.csv')
        
        print(f"Generated {len(predictions)} future predictions for EUR/USD")
        print(f"Period: {predictions['date'].iloc[0]} to {predictions['date'].iloc[-1]}")
        print(f"Predicted Price: {predictions['predicted_price'].iloc[0]:.6f}")
        print(f"Using Model: {predictions['model'].iloc[0]}")
        
        print("\nSample Predictions:")
        print("-" * 30)
        sample = predictions[['date', 'predicted_price']].head(10)
        for _, row in sample.iterrows():
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            print(f"{date_str}: {row['predicted_price']:.6f}")
        
        if len(predictions) > 10:
            print(f"... and {len(predictions) - 10} more predictions")
            
    except Exception as e:
        print(f"Error loading predictions: {e}")

def show_system_info():
    print("\n" + "=" * 50)
    print("SYSTEM CAPABILITIES:")
    print("=" * 50)
    print("1. Fetches 3 years of EUR/USD price data (780 records)")
    print("2. Engineers 11 features (returns, ratios, moving averages, etc.)")
    print("3. Trains 7 ML algorithms (Linear, Random Forest, XGBoost, etc.)")
    print("4. Evaluates and selects best model (Linear Regression won!)")
    print("5. Generates 30-day future price predictions")
    print("6. Creates visualizations and analysis reports")
    print("\nThis is a complete ML pipeline for currency prediction!")

def main():
    show_ml_results()
    show_predictions()
    show_system_info()

if __name__ == "__main__":
    main() 