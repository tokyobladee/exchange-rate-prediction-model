#!/usr/bin/env python3

import json
import os
from pathlib import Path

def show_prediction_summary():
    prediction_files = []
    predictions_dir = Path("predictions")
    
    if predictions_dir.exists():
        prediction_files = list(predictions_dir.glob("*.json"))
    
    if not prediction_files:
        print("No prediction files found")
        return
    
    for file in prediction_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            print(f"\nPrediction: {file.name}")
            print(f"Current Price: {data['current_price']:.6f}")
            print(f"Predicted Price: {data['predicted_price']:.6f}")
            print(f"Expected Change: {data['change_percent']:+.2f}%")
            print(f"Model Used: {data['model']}")
            print(f"Prediction Date: {data['prediction_date']}")
            
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

def main():
    show_prediction_summary()

if __name__ == "__main__":
    main() 