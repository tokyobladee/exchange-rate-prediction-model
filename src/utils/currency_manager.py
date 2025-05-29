#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from analytics.currency_prediction_app import CurrencyPredictionApp
from models.pytorch_models import PyTorchCurrencyModels

warnings.filterwarnings('ignore')

class CurrencyManager:

    def __init__(self):
        self.app = CurrencyPredictionApp()
        self.pytorch_models = PyTorchCurrencyModels()
        self.current_pair = None
        self.pair_data = {}
        self.pair_models = {}
        self.pair_performance = {}
        
        self.setup_directories()
        
        self.load_configurations()
    
    def setup_directories(self):
        
        base_dirs = ['data', 'models', 'plots', 'reports', 'config']
        
        for base_dir in base_dirs:
            os.makedirs(base_dir, exist_ok=True)
            
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
            for pair in major_pairs:
                os.makedirs(f'{base_dir}/{pair}', exist_ok=True)
    
    def load_configurations(self):
        
        config_file = 'config/currency_pairs.json'
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.pair_data = json.load(f)
        else:
            self.pair_data = {}
    
    def save_configurations(self):
        
        config_file = 'config/currency_pairs.json'
        
        with open(config_file, 'w') as f:
            json.dump(self.pair_data, f, indent=2, default=str)
    
    def add_currency_pair(self, pair: str, period: str = '3y', description: str = None):
        
        print(f" ADDING CURRENCY PAIR: {pair}")
        print("=" * 50)
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        
        pair_dirs = ['data', 'models', 'plots', 'reports']
        for dir_name in pair_dirs:
            os.makedirs(f'{dir_name}/{clean_pair}', exist_ok=True)
        
        print(f" Fetching data for {pair}...")
        try:
            self.app.fetch_currency_data([pair], period)
            self.app.engineer_features()
            
            if pair in self.app.processed_data:
                data = self.app.processed_data[pair]
                data.to_csv(f'data/{clean_pair}/{clean_pair}_processed.csv')
                
                self.pair_data[pair] = {
                    'clean_name': clean_pair,
                    'description': description or f"{pair} currency pair",
                    'period': period,
                    'data_shape': data.shape,
                    'date_range': {
                        'start': data.index[0].isoformat(),
                        'end': data.index[-1].isoformat()
                    },
                    'last_updated': datetime.now().isoformat(),
                    'status': 'active'
                }
                
                print(f" {pair} added successfully")
                print(f" Data shape: {data.shape}")
                print(f" Date range: {data.index[0].date()} to {data.index[-1].date()}")
                
                self.save_configurations()
                
            else:
                print(f" Failed to fetch data for {pair}")
                
        except Exception as e:
            print(f" Error adding {pair}: {e}")
    
    def switch_currency_pair(self, pair: str):
        
        if pair not in self.pair_data:
            print(f" Currency pair {pair} not found. Available pairs:")
            self.list_currency_pairs()
            return False
        
        self.current_pair = pair
        clean_pair = self.pair_data[pair]['clean_name']
        
        print(f" SWITCHED TO: {pair}")
        print(f" Data directory: data/{clean_pair}/")
        print(f" Models directory: models/{clean_pair}/")
        print(f" Plots directory: plots/{clean_pair}/")
        
        data_file = f'data/{clean_pair}/{clean_pair}_processed.csv'
        if os.path.exists(data_file):
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            self.app.processed_data[pair] = data
            print(f" Loaded existing data: {data.shape}")
        else:
            print(f"  No processed data found. Run training to generate data.")
        
        return True
    
    def list_currency_pairs(self):
        
        print("\n AVAILABLE CURRENCY PAIRS:")
        print("-" * 50)
        
        if not self.pair_data:
            print("No currency pairs configured yet.")
            return
        
        for i, (pair, info) in enumerate(self.pair_data.items(), 1):
            status_icon = "ACTIVE" if info['status'] == 'active' else ""
            print(f"{i:2d}. {status_icon} {pair:12s} | {info['description']}")
            print(f"      Shape: {info['data_shape']} | Period: {info['period']}")
            print(f"      Updated: {info['last_updated'][:10]}")
            print()
    
    def train_current_pair(self, include_neural: bool = True, include_traditional: bool = True):
        
        if not self.current_pair:
            print(" No currency pair selected. Use switch_currency_pair() first.")
            return
        
        pair = self.current_pair
        clean_pair = self.pair_data[pair]['clean_name']
        
        print(f" TRAINING MODELS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.app.processed_data:
            print(" Fetching fresh data...")
            period = self.pair_data[pair]['period']
            self.app.fetch_currency_data([pair], period)
            self.app.engineer_features()
        
        data = self.app.processed_data[pair]
        all_performance = {}
        
        if include_traditional:
            print("\n Training Traditional ML Models...")
            traditional_performance = self.app.train_models(pair)
            all_performance.update(traditional_performance)
            
            self.app.ml_models.save_models(f'models/{clean_pair}/{clean_pair}_traditional.joblib')
            print(f" Traditional models saved to models/{clean_pair}/")
        
        if include_neural:
            print("\n Training Neural Networks...")
            nn_performance = self.pytorch_models.train_all_models(data)
            all_performance.update(nn_performance)
            
            self.pytorch_models.save_models(f'models/{clean_pair}/{clean_pair}_neural.pth')
            print(f" Neural networks saved to models/{clean_pair}/")
        
        self.pair_performance[pair] = all_performance
        
        performance_file = f'reports/{clean_pair}/{clean_pair}_performance.json'
        with open(performance_file, 'w') as f:
            json.dump(all_performance, f, indent=2)
        
        self.pair_data[pair]['last_trained'] = datetime.now().isoformat()
        self.pair_data[pair]['models_available'] = {
            'traditional': include_traditional,
            'neural': include_neural
        }
        self.save_configurations()
        
        print(f"\n TRAINING RESULTS FOR {pair}:")
        print("-" * 50)
        for model_name, metrics in all_performance.items():
            accuracy = metrics['r2'] * 100
            print(f"{model_name:20s} | Accuracy: {accuracy:6.2f}% | RMSE: {metrics['rmse']:.6f}")
        
        best_model = min(all_performance.keys(), key=lambda x: all_performance[x]['rmse'])
        best_accuracy = all_performance[best_model]['r2'] * 100
        print(f"\n Best Model: {best_model} ({best_accuracy:.2f}% accuracy)")
        
        return all_performance
    
    def predict_current_pair(self, days_ahead: int = 1, model_type: str = 'best'):
        
        if not self.current_pair:
            print(" No currency pair selected. Use switch_currency_pair() first.")
            return None
        
        pair = self.current_pair
        clean_pair = self.pair_data[pair]['clean_name']
        
        print(f" MAKING PREDICTIONS FOR {pair}")
        print("=" * 50)
        
        if pair not in self.app.processed_data:
            data_file = f'data/{clean_pair}/{clean_pair}_processed.csv'
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                self.app.processed_data[pair] = data
            else:
                print(" No data available. Train the models first.")
                return None
        
        data = self.app.processed_data[pair]
        
        if model_type in ['best', 'neural'] or 'neural' in model_type.lower():
            neural_model_file = f'models/{clean_pair}/{clean_pair}_neural.pth'
            if os.path.exists(neural_model_file):
                self.pytorch_models.load_models(neural_model_file)
        
        if model_type in ['best', 'traditional'] or 'traditional' in model_type.lower():
            traditional_model_file = f'models/{clean_pair}/{clean_pair}_traditional.joblib'
            if os.path.exists(traditional_model_file):
                self.app.ml_models.load_models(traditional_model_file)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        last_features = data[feature_cols].iloc[-1:].values
        
        current_price = data['close'].iloc[-1]
        current_date = data.index[-1]
        
        predictions = {}
        
        if model_type == 'best':
            if self.pytorch_models.models and self.pytorch_models.best_model:
                prediction = self.pytorch_models.predict(last_features)
                predictions['Neural_Best'] = prediction.item() if hasattr(prediction, 'item') else float(prediction)
            
            if self.app.ml_models.models and self.app.ml_models.best_model:
                prediction = self.app.ml_models.predict(last_features)
                predictions['Traditional_Best'] = prediction.item() if hasattr(prediction, 'item') else float(prediction)
        
        elif model_type == 'neural':
            if self.pytorch_models.models:
                prediction = self.pytorch_models.predict(last_features)
                predictions['Neural'] = prediction.item() if hasattr(prediction, 'item') else float(prediction)
        
        elif model_type == 'traditional':
            if self.app.ml_models.models:
                prediction = self.app.ml_models.predict(last_features)
                predictions['Traditional'] = prediction.item() if hasattr(prediction, 'item') else float(prediction)
        
        print(f" Current {pair} Data:")
        print(f"   Current Price: {current_price:.6f}")
        print(f"   Current Date: {current_date.date()}")
        print()
        
        print(f" Predictions ({days_ahead} day{'s' if days_ahead > 1 else ''} ahead):")
        for model_name, pred_price in predictions.items():
            change_pct = ((pred_price / current_price) - 1) * 100
            direction = " UP" if change_pct > 0 else " DOWN" if change_pct < 0 else " FLAT"
            print(f"   {model_name:15s}: {pred_price:.6f} ({change_pct:+.2f}%) {direction}")
        
        prediction_data = {
            'pair': pair,
            'current_price': current_price,
            'current_date': current_date.isoformat(),
            'prediction_date': (current_date + timedelta(days=days_ahead)).isoformat(),
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        prediction_file = f'data/{clean_pair}/{clean_pair}_latest_prediction.json'
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=2, default=str)
        
        print(f"\n Predictions saved to {prediction_file}")
        
        return predictions
    
    def compare_pairs(self, pairs: List[str] = None):
        
        if pairs is None:
            pairs = list(self.pair_data.keys())
        
        print(" CURRENCY PAIRS COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        
        for pair in pairs:
            if pair not in self.pair_data:
                print(f"  {pair} not found, skipping...")
                continue
            
            clean_pair = self.pair_data[pair]['clean_name']
            performance_file = f'reports/{clean_pair}/{clean_pair}_performance.json'
            
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    performance = json.load(f)
                
                best_model = min(performance.keys(), key=lambda x: performance[x]['rmse'])
                best_metrics = performance[best_model]
                
                comparison_data.append({
                    'Pair': pair,
                    'Best_Model': best_model,
                    'Accuracy': best_metrics['r2'] * 100,
                    'RMSE': best_metrics['rmse'],
                    'MAE': best_metrics['mae']
                })
            else:
                comparison_data.append({
                    'Pair': pair,
                    'Best_Model': 'Not trained',
                    'Accuracy': 0,
                    'RMSE': float('inf'),
                    'MAE': float('inf')
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(" PERFORMANCE RANKING:")
        print("-" * 80)
        for i, row in comparison_df.iterrows():
            if row['Best_Model'] != 'Not trained':
                print(f"{row.name+1:2d}. {row['Pair']:12s} | {row['Best_Model']:20s} | "
                      f"Accuracy: {row['Accuracy']:6.2f}% | RMSE: {row['RMSE']:.6f}")
            else:
                print(f"{row.name+1:2d}. {row['Pair']:12s} | {'Not trained':20s} | "
                      f"Accuracy: {'N/A':>6s} | RMSE: {'N/A':>10s}")
        
        comparison_df.to_csv('reports/currency_pairs_comparison.csv', index=False)
        print(f"\n Comparison saved to reports/currency_pairs_comparison.csv")
        
        return comparison_df
    
    def get_pair_status(self, pair: str = None):
        
        if pair is None:
            pair = self.current_pair
        
        if not pair or pair not in self.pair_data:
            print(" Invalid or no currency pair specified")
            return None
        
        clean_pair = self.pair_data[pair]['clean_name']
        info = self.pair_data[pair]
        
        print(f" STATUS FOR {pair}")
        print("=" * 50)
        print(f"Description: {info['description']}")
        print(f"Data Period: {info['period']}")
        print(f"Data Shape: {info['data_shape']}")
        print(f"Date Range: {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
        print(f"Last Updated: {info['last_updated'][:10]}")
        print(f"Status: {info['status']}")
        
        print(f"\n Available Files:")
        
        data_file = f'data/{clean_pair}/{clean_pair}_processed.csv'
        print(f"   Data: {'' if os.path.exists(data_file) else ''} {data_file}")
        
        neural_file = f'models/{clean_pair}/{clean_pair}_neural.pth'
        traditional_file = f'models/{clean_pair}/{clean_pair}_traditional.joblib'
        print(f"   Neural Models: {'' if os.path.exists(neural_file) else ''} {neural_file}")
        print(f"   Traditional Models: {'' if os.path.exists(traditional_file) else ''} {traditional_file}")
        
        performance_file = f'reports/{clean_pair}/{clean_pair}_performance.json'
        print(f"   Performance: {'' if os.path.exists(performance_file) else ''} {performance_file}")
        
        prediction_file = f'data/{clean_pair}/{clean_pair}_latest_prediction.json'
        print(f"   Latest Prediction: {'' if os.path.exists(prediction_file) else ''} {prediction_file}")
        
        return info

def main():
    
    manager = CurrencyManager()
    
    print("TESTING CURRENCY MANAGER")
    print("=" * 50)
    
    manager.add_currency_pair('EURUSD=X', '2y', 'Euro to US Dollar')
    
    manager.switch_currency_pair('EURUSD=X')
    
    manager.train_current_pair()
    
    manager.predict_current_pair()
    
    manager.get_pair_status()
    
    print("\n Currency manager test completed!")

if __name__ == "__main__":
    main() 