#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

src_path = Path(__file__).parent.parent
root_path = src_path.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(root_path))

from neural_networks.neural_network_training import NeuralNetworkTrainer

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_prediction_vs_current_chart():
    print("\nCREATING PREDICTION VS CURRENT CHART")
    print("=" * 40)
    
    try:
        if os.path.exists('data/EURUSD=X_processed.csv'):
            data = pd.read_csv('data/EURUSD=X_processed.csv', index_col=0, parse_dates=True)
        else:
            print("No processed data found")
            return False
        
        if os.path.exists('predictions/EURUSD=X_neural_prediction.json'):
            with open('predictions/EURUSD=X_neural_prediction.json', 'r') as f:
                prediction_data = json.load(f)
        else:
            print("No prediction data found")
            return False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('EUR/USD Neural Network Prediction Analysis', fontsize=16, fontweight='bold')
        
        current_price = prediction_data['current_price']
        predicted_price = prediction_data['predicted_price']
        change_pct = prediction_data['change_percent']
        
        prices = [current_price, predicted_price]
        labels = ['Current Price', 'Predicted Price']
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax1.bar(labels, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Current vs Predicted Price', fontweight='bold', fontsize=14)
        ax1.set_ylabel('EUR/USD Price')
        ax1.grid(True, alpha=0.3)
        
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                    f'{price:.6f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.text(0.5, max(prices) * 0.95, f'Expected Change: {change_pct:+.2f}%',
                ha='center', transform=ax1.transData, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        recent_data = data.tail(30)
        ax2.plot(recent_data.index, recent_data['close'], 'b-', linewidth=2, label='Historical Price')
        
        next_date = recent_data.index[-1] + timedelta(days=1)
        ax2.scatter([next_date], [predicted_price], color='red', s=100, zorder=5, label='Prediction')
        ax2.plot([recent_data.index[-1], next_date], [current_price, predicted_price], 
                'r--', linewidth=2, alpha=0.7, label='Predicted Change')
        
        ax2.set_title('30-Day Price Trend with Prediction', fontweight='bold', fontsize=14)
        ax2.set_ylabel('EUR/USD Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.plot(recent_data.index, recent_data['close'], 'b-', linewidth=2, label='Close Price')
        if 'sma_5' in recent_data.columns:
            ax3.plot(recent_data.index, recent_data['sma_5'], 'g--', linewidth=1, label='5-day SMA')
        if 'sma_10' in recent_data.columns:
            ax3.plot(recent_data.index, recent_data['sma_10'], 'orange', linewidth=1, label='10-day SMA')
        
        ax3.set_title('Technical Analysis', fontweight='bold', fontsize=14)
        ax3.set_ylabel('EUR/USD Price')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        ax4.axis('off')
        
        info_text = "NEURAL NETWORK PREDICTION SUMMARY\n\n"
        info_text += f"Current Price:    {current_price:.6f}\n"
        info_text += f"Predicted Price:  {predicted_price:.6f}\n"
        info_text += f"Expected Change:  {change_pct:+.2f}%\n"
        info_text += f"Direction:        {'UP' if change_pct > 0 else 'DOWN' if change_pct < 0 else 'FLAT'}\n\n"
        info_text += f"Model:           {prediction_data.get('model', 'N/A')}\n"
        info_text += f"Prediction Date: {prediction_data.get('prediction_date', 'N/A')[:10]}\n"
        info_text += f"Data Points:     {len(data)} records\n"
        info_text += f"Timeframe:       {data.index[-1].strftime('%Y-%m-%d')} to {data.index[0].strftime('%Y-%m-%d')}\n\n"
        info_text += "Confidence Level: Based on neural network training\n"
        info_text += f"Risk Level:      {'High' if abs(change_pct) > 1 else 'Medium' if abs(change_pct) > 0.5 else 'Low'}"
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('plots/prediction_vs_current_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Prediction vs current chart saved to plots/prediction_vs_current_analysis.png")
        return True
        
    except Exception as e:
        print(f"Error creating prediction vs current chart: {e}")
        return False

def main():
    print("CREATING PREDICTION CHARTS")
    print("=" * 30)
    
    os.makedirs('plots', exist_ok=True)
    
    success = create_prediction_vs_current_chart()
    
    if success:
        print("\nPrediction charts created successfully!")
    else:
        print("\nFailed to create prediction charts")

if __name__ == "__main__":
    main()