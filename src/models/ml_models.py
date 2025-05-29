import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print(" TensorFlow available for deep learning models")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("  TensorFlow not available. Deep learning models (LSTM, GRU) will be skipped.")

class CurrencyPredictionModels:

    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        
        os.makedirs('models', exist_ok=True)
        
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                    time_series_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'target']
            
            self.feature_names = feature_cols
            
            X = data[feature_cols].values
            y = data['target'].values
            
            if np.isnan(X).any() or np.isnan(y).any():
                print("  Warning: NaN values detected in data. Removing...")
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                print(f" Cleaned data shape: {X.shape}")
            
            if time_series_split:
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f" Error in data preparation: {e}")
            raise
    
    def train_linear_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        
        linear_models = {}
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        linear_models['Linear_Regression'] = lr
        
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        linear_models['Ridge_Regression'] = ridge
        
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X_train, y_train)
        linear_models['Lasso_Regression'] = lasso
        
        return linear_models
    
    def train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        
        tree_models = {}
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        tree_models['Random_Forest'] = rf
        
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb.fit(X_train, y_train)
        tree_models['Gradient_Boosting'] = gb
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        tree_models['XGBoost'] = xgb_model
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        tree_models['LightGBM'] = lgb_model
        
        return tree_models
    
    def train_svm_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        
        svm_models = {}
        
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(X_train, y_train)
        svm_models['SVR'] = svr
        
        return svm_models
    
    def create_lstm_model(self, input_shape: Tuple[int, int], units: int = 50):
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot create LSTM model.")
            return None
            
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int], units: int = 50):
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot create GRU model.")
            return None
            
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_lstm_data(self, X: np.ndarray, y: np.ndarray, 
                         timesteps: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        
        X_lstm, y_lstm = [], []
        
        for i in range(timesteps, len(X)):
            X_lstm.append(X[i-timesteps:i])
            y_lstm.append(y[i])
        
        return np.array(X_lstm), np.array(y_lstm)
    
    def train_deep_learning_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  timesteps: int = 60) -> Dict[str, Any]:
        
        dl_models = {}
        
        if not TENSORFLOW_AVAILABLE:
            print("  TensorFlow not available. Skipping deep learning models.")
            return dl_models
        
        X_train_lstm, y_train_lstm = self.prepare_lstm_data(X_train, y_train, timesteps)
        X_test_lstm, y_test_lstm = self.prepare_lstm_data(X_test, y_test, timesteps)
        
        if len(X_train_lstm) == 0:
            print("Not enough data for LSTM/GRU training")
            return dl_models
        
        lstm_model = self.create_lstm_model((timesteps, X_train.shape[1]))
        if lstm_model is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            lstm_history = lstm_model.fit(
                X_train_lstm, y_train_lstm,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_lstm, y_test_lstm),
                callbacks=[early_stopping],
                verbose=0
            )
            
            dl_models['LSTM'] = lstm_model
        
        gru_model = self.create_gru_model((timesteps, X_train.shape[1]))
        if gru_model is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            gru_history = gru_model.fit(
                X_train_lstm, y_train_lstm,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_lstm, y_test_lstm),
                callbacks=[early_stopping],
                verbose=0
            )
            
            dl_models['GRU'] = gru_model
        
        return dl_models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str, is_deep_learning: bool = False,
                      timesteps: int = 60) -> Dict[str, float]:
        
        try:
            if is_deep_learning:
                X_test_lstm, y_test_lstm = self.prepare_lstm_data(X_test, y_test, timesteps)
                if len(X_test_lstm) == 0:
                    return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
                y_pred = model.predict(X_test_lstm, verbose=0).flatten()
                y_true = y_test_lstm
            else:
                y_pred = model.predict(X_test)
                y_true = y_test
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
    
    def train_all_models(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Dict]:
        
        print("Preparing data for training...")
        X_train, X_test, y_train, y_test = self.prepare_data(data, test_size)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        print("\nTraining linear models...")
        linear_models = self.train_linear_models(X_train, y_train)
        self.models.update(linear_models)
        
        print("Training tree-based models...")
        tree_models = self.train_tree_models(X_train, y_train)
        self.models.update(tree_models)
        
        print("Training SVM model...")
        svm_models = self.train_svm_model(X_train, y_train)
        self.models.update(svm_models)
        
        print("Training deep learning models...")
        dl_models = self.train_deep_learning_models(X_train, y_train, X_test, y_test)
        self.models.update(dl_models)
        
        print("\nEvaluating models...")
        for model_name, model in self.models.items():
            is_dl = model_name in ['LSTM', 'GRU']
            performance = self.evaluate_model(model, X_test, y_test, model_name, is_dl)
            self.model_performance[model_name] = performance
            print(f"{model_name}: RMSE={performance['rmse']:.6f}, MAE={performance['mae']:.6f}, R²={performance['r2']:.4f}")
        
        best_model_name = min(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['rmse'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best RMSE: {self.model_performance[best_model_name]['rmse']:.6f}")
        
        return self.model_performance
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model_name in ['LSTM', 'GRU']:
            return model.predict(X, verbose=0).flatten()
        else:
            return model.predict(X)
    
    def save_models(self, filepath: str = 'currency_models.joblib'):
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                'models': self.models,
                'performance': self.model_performance,
                'best_model_name': self.best_model_name,
                'feature_names': self.feature_names
            }
            
            joblib.dump(save_data, filepath)
            print(f" Models saved to {filepath}")
            
        except Exception as e:
            print(f" Error saving models: {e}")
            raise
    
    def load_models(self, filepath: str = 'currency_models.joblib'):
        
        try:
            if not os.path.exists(filepath):
                print(f" Model file not found: {filepath}")
                return False
                
            save_data = joblib.load(filepath)
            
            self.models = save_data.get('models', {})
            self.model_performance = save_data.get('performance', {})
            self.best_model_name = save_data.get('best_model_name', None)
            self.feature_names = save_data.get('feature_names', [])
            
            if self.best_model_name and self.best_model_name in self.models:
                self.best_model = self.models[self.best_model_name]
            
            print(f" Models loaded from {filepath}")
            print(f" Loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f" Error loading models: {e}")
            return False
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance")
            return pd.DataFrame() 