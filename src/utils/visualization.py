import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CurrencyVisualization:

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_price_data(self, data: pd.DataFrame, pair: str, 
                       show_volume: bool = True, save_path: str = None) -> None:
        
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=2)
        ax1.fill_between(data.index, data['low'], data['high'], alpha=0.3, label='High-Low Range')
        
        ax1.set_title(f'{pair} Price Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if show_volume and 'volume' in data.columns:
            ax2.bar(data.index, data['volume'], alpha=0.7, color='gray')
            ax2.set_title('Volume', fontsize=14)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame, pair: str, 
                                 indicators: List[str] = None, save_path: str = None) -> None:
        
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'rsi', 'macd']
        
        available_indicators = [ind for ind in indicators if ind in data.columns]
        
        if not available_indicators:
            print("No technical indicators found in data")
            return
        
        n_plots = len(available_indicators) + 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * n_plots // 2))
        
        if n_plots == 1:
            axes = [axes]
        
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=2)
        
        ma_indicators = [ind for ind in available_indicators if 'sma' in ind or 'ema' in ind]
        for i, ma in enumerate(ma_indicators):
            axes[0].plot(data.index, data[ma], label=ma.upper(), alpha=0.8)
        
        axes[0].set_title(f'{pair} Price and Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        plot_idx = 1
        for indicator in available_indicators:
            if 'sma' not in indicator and 'ema' not in indicator:
                axes[plot_idx].plot(data.index, data[indicator], 
                                   label=indicator.upper(), color=self.colors[plot_idx % len(self.colors)])
                axes[plot_idx].set_title(f'{indicator.upper()}', fontsize=12)
                axes[plot_idx].set_ylabel(indicator.upper(), fontsize=10)
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                
                if indicator == 'rsi':
                    axes[plot_idx].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                    axes[plot_idx].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                
                plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, save_path: str = None) -> None:
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        corr_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance(self, performance_dict: Dict[str, Dict], 
                              save_path: str = None) -> None:
        
        models = list(performance_dict.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [performance_dict[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.upper(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str, dates: pd.DatetimeIndex = None,
                                  save_path: str = None) -> None:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        if dates is not None:
            ax1.plot(dates, y_true, label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.8)
        else:
            ax1.plot(y_true, label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
        
        ax1.set_title(f'{model_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])
        
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Values', fontsize=12)
        ax2.set_ylabel('Predicted Values', fontsize=12)
        ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20, save_path: str = None) -> None:
        
        if importance_df.empty:
            print("No feature importance data available")
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0], alpha=0.8)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(value, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_price_distribution(self, data: pd.DataFrame, pair: str, 
                               save_path: str = None) -> None:
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(data['close'], bins=50, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].set_title(f'{pair} Price Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Price', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        if 'returns' in data.columns:
            axes[0, 1].hist(data['returns'].dropna(), bins=50, alpha=0.7, 
                           color=self.colors[1], edgecolor='black')
            axes[0, 1].set_title('Returns Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Returns', fontsize=10)
            axes[0, 1].set_ylabel('Frequency', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(data.index, data['close'], linewidth=1, color=self.colors[2])
        axes[1, 0].set_title('Price Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Date', fontsize=10)
        axes[1, 0].set_ylabel('Price', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'volatility_20' in data.columns:
            axes[1, 1].plot(data.index, data['volatility_20'], linewidth=1, color=self.colors[3])
            axes[1, 1].set_title('Volatility Over Time', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Date', fontsize=10)
            axes[1, 1].set_ylabel('Volatility', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_price_chart(self, data: pd.DataFrame, pair: str, 
                                     indicators: List[str] = None) -> go.Figure:
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{pair} Price Chart', 'Volume', 'Technical Indicators'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        if indicators:
            for indicator in indicators:
                if indicator in data.columns and ('sma' in indicator or 'ema' in indicator):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator.upper(),
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
        
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color='gray',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        if indicators and 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{pair} Interactive Price Chart',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_multiple_pairs_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                                     save_path: str = None) -> None:
        
        plt.figure(figsize=(14, 8))
        
        for i, (pair, data) in enumerate(data_dict.items()):
            normalized_price = (data['close'] / data['close'].iloc[0]) * 100
            plt.plot(data.index, normalized_price, 
                    label=pair.replace('=X', ''), 
                    linewidth=2, 
                    color=self.colors[i % len(self.colors)])
        
        plt.title('Currency Pairs Comparison (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (Base = 100)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard_summary(self, data: pd.DataFrame, pair: str, 
                               performance_dict: Dict[str, Dict] = None,
                               save_path: str = None) -> None:
        
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(data.index, data['close'], linewidth=2, color=self.colors[0])
        ax1.set_title(f'{pair} Price Chart', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2])
        if 'volume' in data.columns:
            ax2.bar(data.index, data['volume'], alpha=0.7, color='gray')
            ax2.set_title('Volume')
            ax2.tick_params(axis='x', rotation=45)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(data['close'], bins=30, alpha=0.7, color=self.colors[1])
        ax3.set_title('Price Distribution')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Frequency')
        
        ax4 = fig.add_subplot(gs[1, 1])
        if 'returns' in data.columns:
            ax4.hist(data['returns'].dropna(), bins=30, alpha=0.7, color=self.colors[2])
            ax4.set_title('Returns Distribution')
            ax4.set_xlabel('Returns')
            ax4.set_ylabel('Frequency')
        
        ax5 = fig.add_subplot(gs[1, 2])
        if 'rsi' in data.columns:
            ax5.plot(data.index, data['rsi'], color=self.colors[3])
            ax5.axhline(y=70, color='r', linestyle='--', alpha=0.7)
            ax5.axhline(y=30, color='g', linestyle='--', alpha=0.7)
            ax5.set_title('RSI')
            ax5.tick_params(axis='x', rotation=45)
        
        if performance_dict:
            ax6 = fig.add_subplot(gs[2, :])
            models = list(performance_dict.keys())
            rmse_values = [performance_dict[model]['rmse'] for model in models]
            
            bars = ax6.bar(models, rmse_values, color=self.colors[:len(models)])
            ax6.set_title('Model Performance (RMSE)', fontsize=14, fontweight='bold')
            ax6.set_ylabel('RMSE')
            ax6.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, rmse_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle(f'{pair} Analysis Dashboard', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 