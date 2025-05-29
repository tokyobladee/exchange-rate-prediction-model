#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from datetime import datetime
from neural_network_training import NeuralNetworkTrainer
import matplotlib.pyplot as plt
import seaborn as sns

def create_comprehensive_analytics():
    
    print(" CREATING COMPREHENSIVE ANALYTICS")
    print("=" * 50)
    
    try:
        trainer = NeuralNetworkTrainer()
        
        print(" Running complete neural network analysis...")
        trainer.run_complete_neural_analysis(['EURUSD=X'], '2y', 10)
        
        data = trainer.app.processed_data['EURUSD=X']
        current_price = data['close'].iloc[-1]
        current_date = data.index[-1]
        
        print("\n Generating fixed neural network prediction...")
        prediction_data = trainer.generate_neural_predictions('EURUSD=X')
        
        report_path = 'reports/comprehensive_analytics_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE CURRENCY PREDICTION ANALYTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Currency Pair: EUR/USD\n")
            f.write(f"Analysis Period: 2 years\n")
            f.write(f"Neural Network Framework: PyTorch\n")
            f.write(f"Python Version: 3.13 Compatible\n\n")
            
            f.write("CURRENT MARKET STATUS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Current Price: {current_price:.6f}\n")
            f.write(f"Current Date: {current_date.date()}\n")
            f.write(f"Price Range (30d): {data['close'].tail(30).min():.6f} - {data['close'].tail(30).max():.6f}\n")
            f.write(f"30-day Volatility: {data['close'].tail(30).std():.6f}\n\n")
            
            if prediction_data:
                f.write("NEURAL NETWORK PREDICTION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Model Used: {prediction_data['model']}\n")
                f.write(f"Predicted Price: {prediction_data['predicted_price']:.6f}\n")
                f.write(f"Expected Change: {prediction_data['change_percent']:+.2f}%\n")
                f.write(f"Direction: {'UP' if prediction_data['change_percent'] > 0 else 'DOWN'}\n")
                f.write(f"Prediction Confidence: High (R² = {trainer.pytorch_models.model_performance[trainer.pytorch_models.best_model_name]['r2']:.4f})\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            if hasattr(trainer.app, 'ml_models') and trainer.app.ml_models.models:
                traditional_performance = trainer.app.ml_models.model_performance
                f.write("Traditional ML Models:\n")
                for model_name, metrics in traditional_performance.items():
                    f.write(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.6f}\n")
                f.write(f"\nBest Traditional: {trainer.app.ml_models.best_model_name}\n")
            
            if trainer.pytorch_models.model_performance:
                f.write("\nNeural Network Models:\n")
                for model_name, metrics in trainer.pytorch_models.model_performance.items():
                    f.write(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.6f}\n")
                f.write(f"\nBest Neural Network: {trainer.pytorch_models.best_model_name}\n\n")
            
            f.write("TECHNICAL ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"5-day SMA: {data['sma_5'].iloc[-1]:.6f}\n")
            f.write(f"10-day SMA: {data['sma_10'].iloc[-1]:.6f}\n")
            f.write(f"Recent Returns: {data['returns'].iloc[-1]:.6f}\n")
            f.write(f"Volatility (5d): {data['volatility_5'].iloc[-1]:.6f}\n")
            
            recent_trend = "UP" if data['close'].iloc[-1] > data['close'].iloc[-5] else "DOWN"
            f.write(f"5-day Trend: {recent_trend}\n\n")
            
            f.write("RISK ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            volatility = data['volatility_5'].iloc[-1]
            if volatility < 0.01:
                risk_level = "LOW"
            elif volatility < 0.02:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            f.write(f"Current Risk Level: {risk_level}\n")
            f.write(f"Volatility Score: {volatility:.6f}\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 20 + "\n")
            f.write(" plots/EURUSD=X_neural_network_comparison.png - Performance visualization\n")
            f.write(" reports/model_comparison.csv - Detailed model comparison\n")
            f.write("models/EURUSD=X_pytorch_models.pth - Trained neural networks\n")
            f.write(" predictions/EURUSD=X_neural_prediction.json - Latest prediction\n")
        
        print(f" Comprehensive analytics report created: {report_path}")
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'prediction': float(prediction_data['predicted_price']) if prediction_data else None,
            'change_percent': float(prediction_data['change_percent']) if prediction_data else None,
            'model_accuracy': float(trainer.pytorch_models.model_performance[trainer.pytorch_models.best_model_name]['r2']),
            'volatility': float(data['volatility_5'].iloc[-1]),
            'trend_5d': recent_trend,
            'risk_level': risk_level
        }
        
        import json
        with open('reports/dashboard_data.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(" Dashboard data created: reports/dashboard_data.json")
        
        return True
        
    except Exception as e:
        print(f" Error creating analytics: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_prediction_summary():
    
    print("\n CREATING PREDICTION SUMMARY")
    print("=" * 40)
    
    try:
        import json
        if os.path.exists('predictions/EURUSD=X_neural_prediction.json'):
            with open('predictions/EURUSD=X_neural_prediction.json', 'r') as f:
                prediction = json.load(f)
            
            summary_path = 'reports/prediction_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("LATEST EUR/USD PREDICTION SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Generated: {prediction['timestamp']}\n")
                f.write(f"Current Price: {prediction['current_price']:.6f}\n")
                f.write(f"Predicted Price: {prediction['predicted_price']:.6f}\n")
                f.write(f"Expected Change: {prediction['change_percent']:+.2f}%\n")
                f.write(f"Model Used: {prediction['model']}\n")
                f.write(f"Prediction Date: {prediction['prediction_date']}\n\n")
                
                change = prediction['change_percent']
                if abs(change) < 0.1:
                    interpretation = "Minimal change expected - market stability"
                elif abs(change) < 0.5:
                    interpretation = "Small movement expected - normal volatility"
                elif abs(change) < 1.0:
                    interpretation = "Moderate movement expected - watch closely"
                else:
                    interpretation = "Significant movement expected - high attention"
                
                f.write(f"Interpretation: {interpretation}\n")
                
                direction = "upward" if change > 0 else "downward" if change < 0 else "stable"
                f.write(f"Market Direction: {direction}\n")
            
            print(f" Prediction summary created: {summary_path}")
            return True
        else:
            print(" No prediction data found")
            return False
            
    except Exception as e:
        print(f" Error creating prediction summary: {e}")
        return False

def main():
    
    print(" CREATING FRESH ANALYTICS REPORTS")
    print("=" * 60)
    print("This will generate comprehensive, up-to-date analytics with fixed predictions")
    print("=" * 60)
    
    analytics_success = create_comprehensive_analytics()
    
    summary_success = create_prediction_summary()
    
    print("\n" + "=" * 60)
    print(" ANALYTICS GENERATION SUMMARY")
    print("=" * 60)
    
    if analytics_success:
        print(" Comprehensive analytics: Created successfully")
    else:
        print(" Comprehensive analytics: Failed")
    
    if summary_success:
        print(" Prediction summary: Created successfully")
    else:
        print(" Prediction summary: Failed")
    
    if analytics_success and summary_success:
        print("\n ALL ANALYTICS REPORTS CREATED SUCCESSFULLY!")
        print("\n Check these files:")
        print("    reports/comprehensive_analytics_report.txt - Main analytics")
        print("    reports/prediction_summary.txt - Latest prediction")
        print("    reports/dashboard_data.json - Dashboard data")
        print("    plots/EURUSD=X_neural_network_comparison.png - Visualization")
        print("    predictions/EURUSD=X_neural_prediction.json - Raw prediction data")
    else:
        print("\n  Some analytics reports failed to generate")

if __name__ == "__main__":
    main() 