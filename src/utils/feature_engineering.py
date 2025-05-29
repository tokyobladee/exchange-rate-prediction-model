import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:

    def __init__(self):
        self.scaler = None
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        data = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        data['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_30'] = ta.trend.sma_indicator(data['close'], window=30)
        data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
        
        data['macd'] = ta.trend.macd_diff(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        data['macd_histogram'] = ta.trend.macd(data['close'])
        
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        data['bb_upper'] = ta.volatility.bollinger_hband(data['close'])
        data['bb_middle'] = ta.volatility.bollinger_mavg(data['close'])
        data['bb_lower'] = ta.volatility.bollinger_lband(data['close'])
        data['bb_width'] = data['bb_upper'] - data['bb_lower']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        data['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
        data['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
        
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
        
        data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])
        
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
        
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        try:
            data['volume_weighted_price'] = ta.volume.volume_weighted_average_price(
                data['high'], data['low'], data['close'], data['volume']
            )
        except:
            data['volume_weighted_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        return data
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        data = df.copy()
        
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        data['volatility_10'] = data['returns'].rolling(window=10).std()
        data['volatility_20'] = data['returns'].rolling(window=20).std()
        
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        data['gap'] = data['open'] - data['close'].shift(1)
        data['gap_percentage'] = data['gap'] / data['close'].shift(1)
        
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        data['resistance_20'] = data['high'].rolling(window=20).max()
        data['support_20'] = data['low'].rolling(window=20).min()
        data['distance_to_resistance'] = (data['resistance_20'] - data['close']) / data['close']
        data['distance_to_support'] = (data['close'] - data['support_20']) / data['close']
        
        return data
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        data = df.copy()
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        data['year'] = data.index.year
        
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        hour = data.index.hour if hasattr(data.index, 'hour') else 12
        data['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        data['european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        data['american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
        
        return data
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'close', lags: List[int] = None) -> pd.DataFrame:
        
        if lags is None:
            lags = [1, 2, 3, 5]
        
        data = df.copy()
        
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        for window in [5, 10]:
            data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
            data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            data[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window=window).min()
            data[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window=window).max()
        
        return data
    
    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'close', 
                              prediction_horizon: int = 1, target_type: str = 'price') -> pd.DataFrame:
        
        data = df.copy()
        
        if target_type == 'price':
            data['target'] = data[target_col].shift(-prediction_horizon)
        elif target_type == 'return':
            future_price = data[target_col].shift(-prediction_horizon)
            data['target'] = (future_price - data[target_col]) / data[target_col]
        elif target_type == 'direction':
            future_price = data[target_col].shift(-prediction_horizon)
            data['target'] = (future_price > data[target_col]).astype(int)
        else:
            raise ValueError("target_type must be 'price', 'return', or 'direction'")
        
        return data
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'close', 
                        prediction_horizon: int = 1, target_type: str = 'price') -> pd.DataFrame:
        
        print("Starting feature engineering...")
        
        data = self.add_technical_indicators(df)
        print(" Technical indicators added")
        
        data = self.add_price_features(data)
        print(" Price features added")
        
        data = self.add_time_features(data)
        print(" Time features added")
        
        data = self.add_lag_features(data, target_col)
        print(" Lag features added")
        
        data = self.create_target_variable(data, target_col, prediction_horizon, target_type)
        print(" Target variable created")
        
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        print(f" Removed {initial_rows - final_rows} rows with missing values")
        
        self.feature_columns = [col for col in data.columns if col != 'target']
        
        print(f" Feature engineering complete. Dataset shape: {data.shape}")
        return data
    
    def prepare_features_simple(self, df: pd.DataFrame, target_col: str = 'close', 
                               prediction_horizon: int = 1, target_type: str = 'price') -> pd.DataFrame:
        
        print("Starting simplified feature engineering...")
        data = df.copy()
        
        data['returns'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        
        data['close_lag_1'] = data['close'].shift(1)
        data['close_lag_2'] = data['close'].shift(2)
        
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        if target_type == 'price':
            data['target'] = data[target_col].shift(-prediction_horizon)
        elif target_type == 'return':
            future_price = data[target_col].shift(-prediction_horizon)
            data['target'] = (future_price - data[target_col]) / data[target_col]
        elif target_type == 'direction':
            future_price = data[target_col].shift(-prediction_horizon)
            data['target'] = (future_price > data[target_col]).astype(int)
        
        print(" Basic features added")
        
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        print(f" Removed {initial_rows - final_rows} rows with missing values")
        
        feature_cols = [col for col in data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'pair']]
        self.feature_columns = feature_cols
        
        print(f" Simplified feature engineering complete. Dataset shape: {data.shape}")
        print(f" Features: {feature_cols}")
        return data
    
    def scale_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None, 
                      method: str = 'standard') -> tuple:
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        feature_cols = [col for col in train_data.columns if col != 'target']
        train_scaled = train_data.copy()
        train_scaled[feature_cols] = self.scaler.fit_transform(train_data[feature_cols])
        
        if test_data is not None:
            test_scaled = test_data.copy()
            test_scaled[feature_cols] = self.scaler.transform(test_data[feature_cols])
            return train_scaled, test_scaled, self.scaler
        
        return train_scaled, None, self.scaler 