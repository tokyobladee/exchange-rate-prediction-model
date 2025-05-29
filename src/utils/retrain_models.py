#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from analytics.currency_prediction_app import CurrencyPredictionApp
from models.pytorch_models import PyTorchCurrencyModels

warnings.filterwarnings('ignore')

class ModelRetrainer:

    def __init__(self):
        self.app = CurrencyPredictionApp()
        self.pytorch_models = PyTorchCurrencyModels()
        self.training_history = {}
        self.best_configurations = {}
        self.ensemble_models = {}
        
        os.makedirs('config', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('scripts', exist_ok=True)
        
    def hyperparameter_tuning(self, pair: str = 'EURUSD=X', period: str = '3y'):
        
        print(f" HYPERPARAMETER TUNING FOR {pair}")
        print("=" * 60)
        
        self.app.fetch_currency_data([pair], period)
        self.app.engineer_features()
        data = self.app.processed_data[pair]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        X = data[feature_cols]
        y = data['target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f" Training data: {X_train.shape}")
        print(f" Test data: {X_test.shape}")
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
        
        best_models = {}
        tuning_results = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"\n Tuning {model_name}...")
            
            try:
                if model_name == 'RandomForest':
                    from sklearn.ensemble import RandomForestRegressor
                    base_model = RandomForestRegressor(random_state=42)
                elif model_name == 'XGBoost':
                    try:
                        import xgboost as xgb
                        base_model = xgb.XGBRegressor(random_state=42)
                    except ImportError:
                        print(f"  XGBoost not available, skipping...")
                        continue
                
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_score = -grid_search.best_score_
                
                y_pred = best_model.predict(X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_r2 = r2_score(y_test, y_pred)
                
                best_models[model_name] = best_model
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'cv_rmse': np.sqrt(best_score),
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                }
                
                print(f" {model_name} - Best CV RMSE: {np.sqrt(best_score):.6f}")
                print(f" {model_name} - Test RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
                
            except Exception as e:
                print(f" Error tuning {model_name}: {e}")
        
        self.best_configurations[pair] = tuning_results
        
        for model_name, model in best_models.items():
            joblib.dump(model, f'models/{pair}_{model_name}_tuned.joblib')
        
        with open(f'reports/{pair}_hyperparameter_tuning.json', 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        print(f"\n Hyperparameter tuning completed for {pair}")
        print(f" Results saved to reports/{pair}_hyperparameter_tuning.json")
        print(f" Models saved to models/{pair}_*_tuned.joblib")
        
        return tuning_results
    
    def ensemble_models(self, pair: str = 'EURUSD=X', period: str = '3y'):
        
        print(f" CREATING ENSEMBLE MODELS FOR {pair}")
        print("=" * 60)
        
        self.app.fetch_currency_data([pair], period)
        self.app.engineer_features()
        data = self.app.processed_data[pair]
        
        traditional_performance = self.app.train_models(pair)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        X = data[feature_cols]
        y = data['target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        best_traditional = []
        for model_name, model in self.app.ml_models.models.items():
            if traditional_performance[model_name]['r2'] > 0.7:
                best_traditional.append((model_name, model))
        
        print(f" Selected {len(best_traditional)} traditional models for ensemble")
        
        ensemble_results = {}
        
        if len(best_traditional) >= 2:
            print("\n  Creating Voting Ensemble...")
            try:
                voting_ensemble = VotingRegressor(best_traditional)
                voting_ensemble.fit(X_train, y_train)
                
                y_pred = voting_ensemble.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                ensemble_results['Voting_Ensemble'] = {
                    'rmse': rmse,
                    'r2': r2,
                    'models': [name for name, _ in best_traditional]
                }
                
                joblib.dump(voting_ensemble, f'models/{pair}_voting_ensemble.joblib')
                print(f" Voting Ensemble - RMSE: {rmse:.6f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f" Error creating voting ensemble: {e}")
        
        print("\n  Creating Weighted Average Ensemble...")
        
        predictions = {}
        weights = {}
        
        for model_name, model in best_traditional:
            try:
                pred = model.predict(X_test)
                predictions[model_name] = pred
                weights[model_name] = traditional_performance[model_name]['r2']
            except Exception as e:
                print(f"  Could not get predictions from {model_name}: {e}")
        
        if len(predictions) >= 2:
            total_weight = sum(weights.values())
            weighted_pred = np.zeros(len(y_test))
            
            for model_name, pred in predictions.items():
                weight = weights[model_name] / total_weight
                weighted_pred += weight * pred
            
            rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
            r2 = r2_score(y_test, weighted_pred)
            
            ensemble_results['Weighted_Average'] = {
                'rmse': rmse,
                'r2': r2,
                'weights': weights,
                'models': list(predictions.keys())
            }
            
            print(f" Weighted Average - RMSE: {rmse:.6f}, R²: {r2:.4f}")
        
        self.ensemble_models[pair] = ensemble_results
        
        with open(f'reports/{pair}_ensemble_results.json', 'w') as f:
            json.dump(ensemble_results, f, indent=2, default=str)
        
        print(f"\n Ensemble models created for {pair}")
        print(f" Results saved to reports/{pair}_ensemble_results.json")
        
        return ensemble_results
    
    def retrain_all_models(self, pair: str = 'EURUSD=X', period: str = '3y'):
        
        print(f" RETRAINING ALL MODELS FOR {pair}")
        print("=" * 60)
        
        self.app.fetch_currency_data([pair], period)
        self.app.engineer_features()
        
        print("Retraining traditional ML models...")
        traditional_performance = self.app.train_models(pair)
        
        print("Retraining neural networks...")
        data = self.app.processed_data[pair]
        nn_performance = self.pytorch_models.train_all_models(data)
        
        self.app.ml_models.save_models(f'models/{pair}_traditional_retrained.joblib')
        self.pytorch_models.save_models(f'models/{pair}_neural_retrained.pth')
        
        all_performance = {**traditional_performance, **nn_performance}
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'period': period,
            'performance': all_performance,
            'best_model': min(all_performance.keys(), key=lambda x: all_performance[x]['rmse'])
        }
        
        log_file = f'logs/{pair}_retraining_log.json'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f" Retraining completed for {pair}")
        print(f" Best model: {log_entry['best_model']}")
        print(f" Log saved to: {log_file}")
        
        return all_performance

def main():
    
    retrainer = ModelRetrainer()
    
    print(" MODEL RETRAINER TEST")
    print("=" * 40)
    
    tuning_results = retrainer.hyperparameter_tuning('EURUSD=X', '2y')
    
    ensemble_results = retrainer.ensemble_models('EURUSD=X', '2y')
    
    print("\n Retrainer test completed!")

if __name__ == "__main__":
    main() 