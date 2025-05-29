import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class CurrencyDataFetcher:

    def __init__(self):
        self.supported_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X',
            'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X',
            'EURAUD=X', 'EURCHF=X', 'AUDCAD=X', 'CADJPY=X', 'CHFJPY=X'
        ]
        
    def get_currency_data(self, 
                         pair: str, 
                         period: str = "2y", 
                         interval: str = "1d") -> pd.DataFrame:
        
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {pair}")
                
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            data['pair'] = pair.replace('=X', '')
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_pairs(self, 
                          pairs: List[str], 
                          period: str = "2y", 
                          interval: str = "1d") -> Dict[str, pd.DataFrame]:
        
        data_dict = {}
        
        for pair in pairs:
            print(f"Fetching data for {pair}...")
            data = self.get_currency_data(pair, period, interval)
            if not data.empty:
                data_dict[pair] = data
            
        return data_dict
    
    def get_economic_indicators(self) -> pd.DataFrame:
        
        dates = pd.date_range(start='2022-01-01', end=datetime.now(), freq='D')
        
        np.random.seed(42)
        economic_data = pd.DataFrame({
            'date': dates,
            'us_interest_rate': np.random.normal(2.5, 0.5, len(dates)),
            'eu_interest_rate': np.random.normal(1.0, 0.3, len(dates)),
            'inflation_rate': np.random.normal(3.0, 1.0, len(dates)),
            'gdp_growth': np.random.normal(2.0, 1.5, len(dates))
        })
        
        economic_data.set_index('date', inplace=True)
        return economic_data
    
    def validate_pair(self, pair: str) -> bool:
        
        return pair in self.supported_pairs
    
    def list_supported_pairs(self) -> List[str]:
        
        return self.supported_pairs.copy() 