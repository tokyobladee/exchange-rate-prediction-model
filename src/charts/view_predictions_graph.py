#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_prediction_graph():

    try:
        predictions = pd.read_csv('data/EURUSD=X_predictions.csv')
        processed_data = pd.read_csv('data/EURUSD=X_processed.csv')
        
        predictions['date'] = pd.to_datetime(predictions['date'])
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        processed_data.set_index('Date', inplace=True)
        
        print(" Creating prediction graphs...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('EUR/USD Currency Prediction Analysis', fontsize=20, fontweight='bold')
        
        ax1 = axes[0, 0]
        
        recent_data = processed_data.tail(100)
        ax1.plot(recent_data.index, recent_data['close'], 
                label='Historical Price', linewidth=2, color='blue', alpha=0.8)
        
        ax1.plot(predictions['date'], predictions['predicted_price'], 
                label='Future Predictions', linewidth=3, color='red', marker='o', markersize=4)
        
        last_date = recent_data.index[-1]
        ax1.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, label='Today')
        
        ax1.set_title('Historical Price vs Future Predictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('EUR/USD Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = axes[0, 1]
        
        ax2.plot(predictions['date'], predictions['predicted_price'], 
                marker='o', linewidth=2, markersize=6, color='red')
        ax2.fill_between(predictions['date'], predictions['predicted_price'], 
                        alpha=0.3, color='red')
        
        ax2.set_title('30-Day Future Predictions Detail', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Predicted EUR/USD Price', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        pred_price = predictions['predicted_price'].iloc[0]
        current_price = recent_data['close'].iloc[-1]
        change = pred_price - current_price
        change_pct = (change / current_price) * 100
        
        ax2.text(0.05, 0.95, f'Predicted Price: {pred_price:.6f}\n'
                             f'Current Price: {current_price:.6f}\n'
                             f'Change: {change:+.6f} ({change_pct:+.2f}%)',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3 = axes[1, 0]
        
        import joblib
        models_data = joblib.load('models/EURUSD=X_models.joblib')
        
        model_names = list(models_data['performance'].keys())
        r2_scores = [models_data['performance'][name]['r2'] for name in model_names]
        
        bars = ax3.bar(range(len(model_names)), r2_scores, 
                      color=['red' if name == models_data['best_model_name'] else 'lightblue' 
                             for name in model_names])
        
        ax3.set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('R² Score (Higher = Better)', fontsize=12)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels([name.replace('_', ' ') for name in model_names], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        best_idx = model_names.index(models_data['best_model_name'])
        ax3.text(best_idx, r2_scores[best_idx] + 0.02, 'BEST', 
                ha='center', fontweight='bold', color='red')
        
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax4 = axes[1, 1]
        
        ax4.hist(recent_data['close'], bins=30, alpha=0.7, color='blue', 
                label='Historical Prices', density=True)
        
        ax4.axvline(x=pred_price, color='red', linewidth=3, 
                   label=f'Predicted Price: {pred_price:.6f}')
        
        ax4.axvline(x=current_price, color='green', linewidth=2, linestyle='--',
                   label=f'Current Price: {current_price:.6f}')
        
        ax4.set_title('Price Distribution Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('EUR/USD Price', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig('plots/prediction_analysis_complete.png', dpi=300, bbox_inches='tight')
        print(" Comprehensive prediction graph saved as: plots/prediction_analysis_complete.png")
        
        plt.show()
        
        create_simple_prediction_timeline(predictions, recent_data)
        
    except Exception as e:
        print(f" Error creating graphs: {e}")

def create_simple_prediction_timeline(predictions, historical_data):

    plt.figure(figsize=(14, 8))
    
    last_30_days = historical_data.tail(30)
    plt.plot(last_30_days.index, last_30_days['close'], 
             label='Historical Price (Last 30 Days)', linewidth=2, color='blue', marker='o', markersize=3)
    
    plt.plot(predictions['date'], predictions['predicted_price'], 
             label='Future Predictions (Next 30 Days)', linewidth=3, color='red', marker='s', markersize=4)
    
    today = last_30_days.index[-1]
    plt.axvline(x=today, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    
    plt.title('EUR/USD: Historical vs Predicted Prices', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('EUR/USD Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    current_price = last_30_days['close'].iloc[-1]
    predicted_price = predictions['predicted_price'].iloc[0]
    
    plt.annotate(f'Current: {current_price:.6f}', 
                xy=(today, current_price), xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Predicted: {predicted_price:.6f}', 
                xy=(predictions['date'].iloc[0], predicted_price), xytext=(10, -20),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    plt.savefig('plots/prediction_timeline.png', dpi=300, bbox_inches='tight')
    print(" Simple prediction timeline saved as: plots/prediction_timeline.png")
    
    plt.show()

if __name__ == "__main__":
    print(" Creating Prediction Graphs...")
    print("=" * 50)
    create_prediction_graph()
    print("\n Graphs created successfully!")
    print(" Check the 'plots' folder for:")
    print("   - prediction_analysis_complete.png")
    print("   - prediction_timeline.png")
    print("   - EURUSD=X_predictions_vs_actual.png")
    print("   - EURUSD=X_dashboard.png") 