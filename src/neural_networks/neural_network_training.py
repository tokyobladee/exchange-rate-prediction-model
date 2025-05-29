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

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from analytics.currency_prediction_app import CurrencyPredictionApp
from models.pytorch_models import PyTorchCurrencyModels

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NeuralNetworkTrainer:
    
    def __init__(self):
        self.app = CurrencyPredictionApp()
        self.pytorch_models = PyTorchCurrencyModels()
        self.combined_performance = {}
    
    def fetch_and_prepare_data(self, pairs=['EURUSD=X'], period='3y'):
        
        print(f"FETCHING AND PREPARING DATA")
        print("=" * 50)
        
        self.app.fetch_currency_data(pairs, period)
        self.app.engineer_features()
        
        print(f"Data preparation complete for {len(self.app.processed_data)} currency pairs")
        
    def train_traditional_models(self, pair='EURUSD=X'):
        
        print(f"\nTRAINING TRADITIONAL ML MODELS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.app.processed_data:
            print(f"No data available for {pair}")
            return {}
        
        traditional_performance = self.app.train_models(pair)
        
        for model_name, metrics in traditional_performance.items():
            self.combined_performance[f"Traditional_{model_name}"] = metrics
        
        return traditional_performance
    
    def train_neural_networks(self, pair='EURUSD=X', sequence_length=10):
        
        print(f"\nTRAINING NEURAL NETWORKS FOR {pair}")
        print("=" * 60)
        
        if pair not in self.app.processed_data:
            print(f"No data available for {pair}")
            return {}
        
        data = self.app.processed_data[pair]
        
        nn_performance = self.pytorch_models.train_all_models(data, sequence_length=sequence_length)
        
        for model_name, metrics in nn_performance.items():
            self.combined_performance[f"Neural_{model_name}"] = metrics
        
        self.pytorch_models.save_models(f'models/{pair}_pytorch_models.pth')
        
        return nn_performance
    
    def compare_all_models(self):
        
        print(f"\nMODEL PERFORMANCE COMPARISON")
        print("=" * 60)
        
        if not self.combined_performance:
            print("No models trained yet")
            return
        
        comparison_data = []
        for model_name, metrics in self.combined_performance.items():
            comparison_data.append({
                'Model': model_name,
                'Type': 'Traditional' if 'Traditional_' in model_name else 'Neural Network',
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print("\nMODEL RANKINGS (by RMSE):")
        print("-" * 80)
        for i, row in comparison_df.iterrows():
            print(f"{row.name+1:2d}. {row['Model']:25s} | RMSE: {row['RMSE']:.6f} | R²: {row['R²']:7.4f} | Type: {row['Type']}")
        
        best_overall = comparison_df.iloc[0]
        best_traditional = comparison_df[comparison_df['Type'] == 'Traditional'].iloc[0] if len(comparison_df[comparison_df['Type'] == 'Traditional']) > 0 else None
        best_neural = comparison_df[comparison_df['Type'] == 'Neural Network'].iloc[0] if len(comparison_df[comparison_df['Type'] == 'Neural Network']) > 0 else None
        
        print(f"\nBEST OVERALL: {best_overall['Model']} (RMSE: {best_overall['RMSE']:.6f})")
        if best_traditional is not None:
            print(f"BEST TRADITIONAL: {best_traditional['Model']} (RMSE: {best_traditional['RMSE']:.6f})")
        if best_neural is not None:
            print(f"BEST NEURAL NETWORK: {best_neural['Model']} (RMSE: {best_neural['RMSE']:.6f})")
        
        comparison_df.to_csv('reports/model_comparison.csv', index=False)
        
        return comparison_df
    
    def create_performance_visualizations(self, pair='EURUSD=X'):
        
        print(f"\nCREATING PERFORMANCE VISUALIZATIONS")
        print("=" * 50)
        
        if not self.combined_performance:
            print("No performance data available")
            return
        
        models = list(self.combined_performance.keys())
        rmse_values = [self.combined_performance[model]['rmse'] for model in models]
        r2_values = [self.combined_performance[model]['r2'] for model in models]
        model_types = ['Traditional' if 'Traditional_' in model else 'Neural Network' for model in models]
        
        clean_names = [model.replace('Traditional_', '').replace('Neural_', '') for model in models]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Performance Comparison - {pair}', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bars1 = ax1.bar(range(len(models)), rmse_values, color=colors, alpha=0.7)
        ax1.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('RMSE')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(clean_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        bars2 = ax2.bar(range(len(models)), r2_values, color=colors, alpha=0.7)
        ax2.set_title('R-squared Score', fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R²')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(clean_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, r2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        type_counts = pd.Series(model_types).value_counts()
        ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
        ax3.set_title('Model Type Distribution', fontweight='bold')
        
        traditional_mask = [t == 'Traditional' for t in model_types]
        neural_mask = [t == 'Neural Network' for t in model_types]
        
        if any(traditional_mask):
            ax4.scatter([rmse_values[i] for i in range(len(rmse_values)) if traditional_mask[i]], 
                       [r2_values[i] for i in range(len(r2_values)) if traditional_mask[i]], 
                       c='blue', alpha=0.7, s=100, label='Traditional ML')
        
        if any(neural_mask):
            ax4.scatter([rmse_values[i] for i in range(len(rmse_values)) if neural_mask[i]], 
                       [r2_values[i] for i in range(len(r2_values)) if neural_mask[i]], 
                       c='red', alpha=0.7, s=100, label='Neural Networks')
        
        ax4.set_xlabel('RMSE')
        ax4.set_ylabel('R²')
        ax4.set_title('RMSE vs R² Performance', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for i, name in enumerate(clean_names):
            ax4.annotate(name, (rmse_values[i], r2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'plots/{pair}_neural_network_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance visualization saved to plots/{pair}_neural_network_comparison.png")
    
    def generate_neural_predictions(self, pair: str):
        
        print(f"GENERATING NEURAL NETWORK PREDICTIONS")
        print("=" * 50)
        
        if pair not in self.app.processed_data:
            print(f"No data available for {pair}")
            return None
        
        data = self.app.processed_data[pair]
        
        if not self.pytorch_models.models:
            model_file = f'models/{pair}_pytorch_models.pth'
            if os.path.exists(model_file):
                self.pytorch_models.load_models(model_file)
            else:
                print(f"No trained neural networks found for {pair}")
                return None
        
        current_price = data['close'].iloc[-1]
        current_date = data.index[-1]
        
        print(f"Neural Network Prediction for {pair}:")
        print(f"   Model: {self.pytorch_models.best_model_name}")
        print(f"   Current Price: {current_price:.6f}")
        print(f"   Current Date: {current_date.date()}")
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'target']
            last_features = data[feature_cols].iloc[-1:].values
            
            prediction = self.pytorch_models.predict_next(last_features)
            
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 0:
                    next_price = float(prediction)
                else:
                    next_price = float(prediction[0])
            else:
                next_price = float(prediction)
            
            change_percent = ((next_price - current_price) / current_price) * 100
            
            print(f"   Predicted Price: {next_price:.6f}")
            print(f"   Expected Change: {change_percent:+.2f}%")
            
            prediction_data = {
                'pair': pair,
                'current_price': float(current_price),
                'predicted_price': next_price,
                'change_percent': change_percent,
                'model': self.pytorch_models.best_model_name,
                'prediction_date': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs('predictions', exist_ok=True)
            with open(f'predictions/{pair}_neural_prediction.json', 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            print(f"Prediction saved to predictions/{pair}_neural_prediction.json")
            
            return prediction_data
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None
    
    def run_complete_neural_analysis(self, pairs=['EURUSD=X'], period='3y', sequence_length=10):
        
        print("COMPLETE NEURAL NETWORK ANALYSIS")
        print("=" * 60)
        
        self.fetch_and_prepare_data(pairs, period)
        
        for pair in pairs:
            print(f"\nProcessing {pair}...")
            
            traditional_perf = self.train_traditional_models(pair)
            neural_perf = self.train_neural_networks(pair, sequence_length)
            
            comparison_df = self.compare_all_models()
            
            self.create_performance_visualizations(pair)
            
            prediction = self.generate_neural_predictions(pair)
            
            self.create_neural_summary_report(pair, comparison_df)
        
        print("\nComplete neural network analysis finished!")
        return True
    
    def create_neural_summary_report(self, pair, comparison_df):
        
        print(f"CREATING NEURAL ANALYSIS REPORT")
        print("=" * 40)
        
        report_content = "NEURAL NETWORK ANALYSIS REPORT\n"
        report_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"Currency Pair: {pair}\n\n"
        report_content += "MODEL PERFORMANCE SUMMARY:\n"
        report_content += "=" * 50 + "\n"
        
        if comparison_df is not None and len(comparison_df) > 0:
            report_content += f"\nTop 5 Models by RMSE:\n"
            report_content += "-" * 40 + "\n"
            
            for i, row in comparison_df.head().iterrows():
                r2_score = row['R²'] if 'R²' in comparison_df.columns else row.get('R2', 0)
                report_content += f"{i+1}. {row['Model']:25s} | RMSE: {row['RMSE']:.6f} | R2: {r2_score:7.4f}\n"
            
            best_model = comparison_df.iloc[0]
            best_r2 = best_model['R²'] if 'R²' in comparison_df.columns else best_model.get('R2', 0)
            report_content += f"\nBest Overall Model: {best_model['Model']}\n"
            report_content += f"  - RMSE: {best_model['RMSE']:.6f}\n"
            report_content += f"  - R2: {best_r2:.4f}\n"
            report_content += f"  - Type: {best_model['Type']}\n"
        
        if os.path.exists(f'predictions/{pair}_neural_prediction.json'):
            with open(f'predictions/{pair}_neural_prediction.json', 'r') as f:
                pred_data = json.load(f)
            
            report_content += f"\nCURRENT PREDICTION:\n"
            report_content += "-" * 20 + "\n"
            report_content += f"Current Price: {pred_data['current_price']:.6f}\n"
            report_content += f"Predicted Price: {pred_data['predicted_price']:.6f}\n"
            report_content += f"Expected Change: {pred_data['change_percent']:+.2f}%\n"
            report_content += f"Model Used: {pred_data['model']}\n"
        
        report_content += f"\nFILES GENERATED:\n"
        report_content += "-" * 15 + "\n"
        report_content += f"- models/{pair}_pytorch_models.pth - Trained neural networks\n"
        report_content += f"- plots/{pair}_neural_network_comparison.png - Performance charts\n"
        report_content += f"- predictions/{pair}_neural_prediction.json - Latest prediction\n"
        report_content += f"- reports/model_comparison.csv - Model comparison data\n"
        
        os.makedirs('reports', exist_ok=True)
        with open(f'reports/{pair}_neural_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Neural analysis report saved to reports/{pair}_neural_analysis_report.txt")

def main():
    
    trainer = NeuralNetworkTrainer()
    
    pairs = ['EURUSD=X']
    period = '2y'
    sequence_length = 10
    
    trainer.run_complete_neural_analysis(pairs, period, sequence_length)

if __name__ == "__main__":
    main()