#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

src_path = Path(__file__).parent.parent
root_path = src_path.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(root_path))

from utils.data_fetcher import CurrencyDataFetcher
from utils.feature_engineering import FeatureEngineer
from utils.visualization import CurrencyVisualization
from models.ml_models import CurrencyPredictionModels

warnings.filterwarnings('ignore')

class CurrencyPredictionApp:

    def __init__(self):
        self.data_fetcher = CurrencyDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = CurrencyVisualization()
        self.ml_models = CurrencyPredictionModels()
        
        self.raw_data = {}
        self.processed_data = {}
        self.predictions = {}
        
        self.create_output_directories()
    
    def create_output_directories(self):
        
        directories = ['plots', 'models', 'data', 'reports']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def fetch_currency_data(self, pairs: List[str], period: str = "2y", 
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        
        print("=" * 60)
        print("FETCHING CURRENCY DATA")
        print("=" * 60)
        
        valid_pairs = []
        for pair in pairs:
            if self.data_fetcher.validate_pair(pair):
                valid_pairs.append(pair)
            else:
                print(f"Warning: {pair} is not a supported currency pair")
        
        if not valid_pairs:
            print("No valid currency pairs provided")
            return {}
        
        for pair in valid_pairs:
            print(f"Fetching data for {pair}...")
            data = self.data_fetcher.get_currency_data(pair, period='3y', interval='1d')
            if data is not None and len(data) > 0:
                self.raw_data[pair] = data
                print(f" Fetched {len(data)} records")
            else:
                print(f" Failed to fetch data for {pair}")
                
        if not self.raw_data:
            raise ValueError("No data was successfully fetched for any currency pair")
        
        return self.raw_data
    
    def engineer_features(self, target_col: str = 'close', 
                         prediction_horizon: int = 1, 
                         target_type: str = 'price') -> Dict[str, pd.DataFrame]:
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        for pair, data in self.raw_data.items():
            print(f"\nProcessing {pair}...")
            
            processed_data = self.feature_engineer.prepare_features_simple(
                data, target_col, prediction_horizon, target_type
            )
            
            self.processed_data[pair] = processed_data
            
            processed_data.to_csv(f'data/{pair}_processed.csv')
            print(f"Processed data saved to data/{pair}_processed.csv")
        
        return self.processed_data
    
    def train_models(self, pair: str, test_size: float = 0.2) -> Dict[str, Dict]:
        
        print(f"\n" + "=" * 60)
        print(f"TRAINING MODELS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.processed_data:
            print(f"No processed data available for {pair}")
            return {}
        
        data = self.processed_data[pair]
        print(f"Training with {len(data)} samples and {len(data.columns)-1} features")
        
        performance = self.ml_models.train_all_models(data, test_size)
        
        self.ml_models.save_models(f'models/{pair}_models.joblib')
        
        return performance
    
    def create_visualizations(self, pair: str, show_predictions: bool = True):
        
        print(f"\n" + "=" * 60)
        print(f"CREATING VISUALIZATIONS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.raw_data:
            print(f"No data available for {pair}")
            return
        
        raw_data = self.raw_data[pair]
        
        print("Creating price chart...")
        self.visualizer.plot_price_data(
            raw_data, pair, 
            save_path=f'plots/{pair}_price_chart.png'
        )
        
        if pair in self.processed_data:
            processed_data = self.processed_data[pair]
            
            print("Creating technical indicators chart...")
            self.visualizer.plot_technical_indicators(
                processed_data, pair,
                indicators=['sma_20', 'sma_50', 'rsi', 'macd'],
                save_path=f'plots/{pair}_technical_indicators.png'
            )
            
            print("Creating correlation matrix...")
            self.visualizer.plot_correlation_matrix(
                processed_data.select_dtypes(include=[np.number]).iloc[:, :20],
                save_path=f'plots/{pair}_correlation_matrix.png'
            )
            
            print("Creating price distribution analysis...")
            self.visualizer.plot_price_distribution(
                processed_data, pair,
                save_path=f'plots/{pair}_price_distribution.png'
            )
        
        if hasattr(self.ml_models, 'model_performance') and self.ml_models.model_performance:
            print("Creating model performance comparison...")
            self.visualizer.plot_model_performance(
                self.ml_models.model_performance,
                save_path=f'plots/{pair}_model_performance.png'
            )
            
            if self.ml_models.best_model_name:
                importance_df = self.ml_models.get_feature_importance()
                if not importance_df.empty:
                    print("Creating feature importance plot...")
                    self.visualizer.plot_feature_importance(
                        importance_df,
                        save_path=f'plots/{pair}_feature_importance.png'
                    )
        
        if show_predictions and hasattr(self.ml_models, 'best_model') and self.ml_models.best_model:
            print("Creating predictions vs actual plot...")
            self.create_prediction_plots(pair)
        
        if pair in self.processed_data:
            print("Creating dashboard summary...")
            performance_dict = getattr(self.ml_models, 'model_performance', None)
            self.visualizer.create_dashboard_summary(
                self.processed_data[pair], pair, performance_dict,
                save_path=f'plots/{pair}_dashboard.png'
            )
    
    def create_prediction_plots(self, pair: str):
        
        if pair not in self.processed_data:
            return
        
        data = self.processed_data[pair]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        
        last_features = data[feature_cols].iloc[-1:].values
        
        X = data[feature_cols].values
        y = data['target'].values
        
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        if self.ml_models.best_model_name in ['LSTM', 'GRU']:
            X_test_lstm, y_test_lstm = self.ml_models.prepare_lstm_data(X_test, y_test, 60)
            if len(X_test_lstm) > 0:
                y_pred = self.ml_models.best_model.predict(X_test_lstm, verbose=0).flatten()
                y_true = y_test_lstm
                dates = data.index[split_idx + 60:]
            else:
                return
        else:
            y_pred = self.ml_models.best_model.predict(X_test)
            y_true = y_test
            dates = data.index[split_idx:]
        
        self.visualizer.plot_predictions_vs_actual(
            y_true, y_pred, self.ml_models.best_model_name, dates,
            save_path=f'plots/{pair}_predictions_vs_actual.png'
        )
    
    def generate_predictions(self, pair: str, days_ahead: int = 30) -> pd.DataFrame:
        
        print(f"\n" + "=" * 60)
        print(f"GENERATING ITERATIVE PREDICTIONS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.processed_data:
            print(f"No processed data available for {pair}")
            return pd.DataFrame()
        
        if not hasattr(self.ml_models, 'best_model') or self.ml_models.best_model is None:
            print("No trained model available")
            return pd.DataFrame()
        
        data = self.processed_data[pair].copy()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        
        predictions = []
        dates = []
        
        last_date = data.index[-1]
        current_data = data.copy()
        
        price_mean = data['close'].mean()
        price_std = data['close'].std()
        price_min = data['close'].min()
        price_max = data['close'].max()
        
        print(f"Starting iterative predictions from {last_date.date()}")
        print(f"Price bounds: {price_min:.6f} - {price_max:.6f} (mean: {price_mean:.6f})")
        
        for i in range(days_ahead):
            current_features = current_data[feature_cols].iloc[-1:].values
            
            if self.ml_models.best_model_name in ['LSTM', 'GRU']:
                pred = self.ml_models.best_model.predict(current_features.reshape(1, 1, -1), verbose=0)[0, 0]
            else:
                pred = self.ml_models.best_model.predict(current_features)[0]
            
            pred = max(price_min * 0.8, min(price_max * 1.2, pred))
            
            last_price = current_data['close'].iloc[-1]
            max_change = last_price * 0.05
            if abs(pred - last_price) > max_change:
                if pred > last_price:
                    pred = last_price + max_change
                else:
                    pred = last_price - max_change
            
            predictions.append(pred)
            
            next_date = last_date + timedelta(days=i+1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            dates.append(next_date)
            
            new_row = current_data.iloc[-1:].copy()
            new_row.index = [next_date]
            
            new_row['close'] = pred
            new_row['high'] = pred * (1 + abs(np.random.normal(0, 0.0005)))
            new_row['low'] = pred * (1 - abs(np.random.normal(0, 0.0005)))
            new_row['open'] = current_data['close'].iloc[-1]
            
            if 'returns' in new_row.columns:
                new_row['returns'] = (pred - current_data['close'].iloc[-1]) / current_data['close'].iloc[-1]
                new_row['returns'] = max(-0.05, min(0.05, new_row['returns'].iloc[0]))
            
            if 'price_change' in new_row.columns:
                new_row['price_change'] = pred - current_data['close'].iloc[-1]
            
            if 'high_low_ratio' in new_row.columns:
                new_row['high_low_ratio'] = new_row['high'].iloc[0] / new_row['low'].iloc[0]
            
            if 'close_open_ratio' in new_row.columns:
                new_row['close_open_ratio'] = new_row['close'].iloc[0] / new_row['open'].iloc[0]
            
            if 'price_position' in new_row.columns:
                new_row['price_position'] = (new_row['close'].iloc[0] - new_row['low'].iloc[0]) / (new_row['high'].iloc[0] - new_row['low'].iloc[0])
            
            if 'close_lag_1' in new_row.columns:
                new_row['close_lag_1'] = current_data['close'].iloc[-1]
            if 'close_lag_2' in new_row.columns:
                new_row['close_lag_2'] = current_data['close'].iloc[-2] if len(current_data) > 1 else current_data['close'].iloc[-1]
            
            if 'sma_5' in new_row.columns:
                recent_prices = current_data['close'].tail(4).tolist() + [pred]
                new_row['sma_5'] = np.mean(recent_prices)
            
            if 'sma_10' in new_row.columns:
                recent_prices = current_data['close'].tail(9).tolist() + [pred]
                new_row['sma_10'] = np.mean(recent_prices)
            
            if 'volatility_5' in new_row.columns:
                recent_returns = current_data['returns'].tail(4).tolist() + [new_row['returns'].iloc[0]]
                new_row['volatility_5'] = np.std(recent_returns)
            
            if 'day_of_week' in new_row.columns:
                new_row['day_of_week'] = next_date.weekday()
            if 'month' in new_row.columns:
                new_row['month'] = next_date.month
            
            current_data = pd.concat([current_data, new_row])
            
            if len(current_data) > 100:
                current_data = current_data.tail(100)
        
        predictions_df = pd.DataFrame({
            'date': dates,
            'predicted_price': predictions,
            'pair': pair,
            'model': self.ml_models.best_model_name
        })
        
        predictions_df.to_csv(f'data/{pair}_predictions.csv', index=False)
        
        print(f"Generated {days_ahead} iterative predictions for {pair}")
        print(f"Price range: {min(predictions):.6f} - {max(predictions):.6f}")
        print(f"Average predicted price: {np.mean(predictions):.6f}")
        print(f"Predictions saved to data/{pair}_predictions.csv")
        
        return predictions_df
    
    def run_complete_analysis(self, pairs: List[str], target_pair: str = None, 
                             period: str = "2y", prediction_days: int = 30):
        
        print(" STARTING CURRENCY PREDICTION ANALYSIS")
        print("=" * 80)
        
        self.fetch_currency_data(pairs, period)
        
        if not self.raw_data:
            print(" No data fetched. Exiting...")
            return
        
        self.engineer_features()
        
        if target_pair is None:
            target_pair = list(self.processed_data.keys())[0]
        
        if target_pair not in self.processed_data:
            print(f" Target pair {target_pair} not available")
            return
        
        print(f"\n Focusing analysis on: {target_pair}")
        
        performance = self.train_models(target_pair)
        
        self.create_visualizations(target_pair)
        
        predictions_df = self.generate_predictions(target_pair, prediction_days)
        
        if len(self.raw_data) > 1:
            print("\nCreating multi-pair comparison...")
            self.visualizer.plot_multiple_pairs_comparison(
                self.raw_data,
                save_path='plots/multi_pair_comparison.png'
            )
        
        self.generate_summary_report(target_pair, performance, predictions_df)
        
        print("\n" + "=" * 80)
        print(" ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f" Visualizations saved in: plots/")
        print(f"Models saved in: models/")
        print(f" Data saved in: data/")
        print(f" Report saved in: reports/")
    
    def generate_summary_report(self, pair: str, performance: Dict, predictions_df: pd.DataFrame):
        
        report_path = f'reports/{pair}_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CURRENCY PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Currency Pair: {pair}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {self.raw_data[pair].index[0].date()} to {self.raw_data[pair].index[-1].date()}\n")
            f.write(f"Total Records: {len(self.raw_data[pair])}\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            raw_data = self.raw_data[pair]
            f.write(f"Price Range: {raw_data['close'].min():.6f} - {raw_data['close'].max():.6f}\n")
            f.write(f"Average Price: {raw_data['close'].mean():.6f}\n")
            f.write(f"Price Volatility: {raw_data['close'].std():.6f}\n\n")
            
            if performance:
                f.write("MODEL PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                for model_name, metrics in performance.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
                    f.write(f"  MAE: {metrics['mae']:.6f}\n")
                    f.write(f"  R-squared: {metrics['r2']:.4f}\n\n")
                
                f.write(f"Best Model: {self.ml_models.best_model_name}\n")
                f.write(f"Best RMSE: {performance[self.ml_models.best_model_name]['rmse']:.6f}\n\n")
            
            if not predictions_df.empty:
                f.write("FUTURE PREDICTIONS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Prediction Period: {predictions_df['date'].iloc[0].date()} to {predictions_df['date'].iloc[-1].date()}\n")
                f.write(f"Number of Predictions: {len(predictions_df)}\n")
                f.write(f"Predicted Price Range: {predictions_df['predicted_price'].min():.6f} - {predictions_df['predicted_price'].max():.6f}\n")
                f.write(f"Average Predicted Price: {predictions_df['predicted_price'].mean():.6f}\n\n")
            
            if pair in self.processed_data:
                processed_data = self.processed_data[pair]
                f.write("FEATURE ENGINEERING\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Features: {len(processed_data.columns) - 1}\n")
                f.write(f"Processed Records: {len(processed_data)}\n")
                f.write(f"Features Include: Technical indicators, Price features, Time features, Lag features\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 20 + "\n")
            f.write(f"- Price chart: plots/{pair}_price_chart.png\n")
            f.write(f"- Technical indicators: plots/{pair}_technical_indicators.png\n")
            f.write(f"- Model performance: plots/{pair}_model_performance.png\n")
            f.write(f"- Predictions: plots/{pair}_predictions_vs_actual.png\n")
            f.write(f"- Dashboard: plots/{pair}_dashboard.png\n")
            f.write(f"- Trained models: models/{pair}_models.joblib\n")
            f.write(f"- Processed data: data/{pair}_processed.csv\n")
            f.write(f"- Future predictions: data/{pair}_predictions.csv\n")
        
        print(f"Summary report saved to {report_path}")

def main():
    
    app = CurrencyPredictionApp()
    
    currency_pairs = [
        'EURUSD=X',
        'GBPUSD=X',
        'USDJPY=X',
        'USDCHF=X',
        'AUDUSD=X'
    ]
    
    app.run_complete_analysis(
        pairs=currency_pairs,
        target_pair='EURUSD=X',
        period="2y",
        prediction_days=30
    )

if __name__ == "__main__":
    main() 