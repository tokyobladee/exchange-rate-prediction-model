#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

class PredictionValidator:

    def __init__(self):
        self.validation_thresholds = {
            'max_daily_change': 0.05,
            'volatility_multiplier': 3.0,
            'z_score_threshold': 3.0,
            'isolation_contamination': 0.1
        }
        
        self.validation_history = {}
        
    def validate_prediction(self, current_price: float, predicted_price: float, 
                          historical_data: pd.DataFrame, pair: str) -> Dict:
        
        print(f" VALIDATING PREDICTION FOR {pair}")
        print("=" * 50)
        
        validation_results = {
            'pair': pair,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'prediction_change': (predicted_price / current_price - 1) * 100,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_status': 'VALID',
            'warnings': [],
            'recommendations': []
        }
        
        daily_change = abs(predicted_price / current_price - 1)
        max_change = self.validation_thresholds['max_daily_change']
        
        validation_results['checks']['daily_change'] = {
            'value': daily_change * 100,
            'threshold': max_change * 100,
            'status': 'PASS' if daily_change <= max_change else 'FAIL',
            'description': f'Daily change: {daily_change*100:.2f}% (max: {max_change*100:.1f}%)'
        }
        
        if daily_change > max_change:
            validation_results['warnings'].append(f"Large daily change: {daily_change*100:.2f}%")
            validation_results['overall_status'] = 'WARNING'
        
        if 'close' in historical_data.columns:
            returns = historical_data['close'].pct_change().dropna()
            historical_volatility = returns.std()
            predicted_return = predicted_price / current_price - 1
            
            volatility_threshold = historical_volatility * self.validation_thresholds['volatility_multiplier']
            
            validation_results['checks']['volatility'] = {
                'predicted_return': predicted_return * 100,
                'historical_volatility': historical_volatility * 100,
                'threshold': volatility_threshold * 100,
                'status': 'PASS' if abs(predicted_return) <= volatility_threshold else 'FAIL',
                'description': f'Return vs {self.validation_thresholds["volatility_multiplier"]}x historical volatility'
            }
            
            if abs(predicted_return) > volatility_threshold:
                validation_results['warnings'].append(
                    f"Prediction exceeds {self.validation_thresholds['volatility_multiplier']}x historical volatility"
                )
                validation_results['overall_status'] = 'WARNING'
        
        if 'close' in historical_data.columns and len(historical_data) > 30:
            recent_prices = historical_data['close'].tail(30)
            price_mean = recent_prices.mean()
            price_std = recent_prices.std()
            
            z_score = abs(predicted_price - price_mean) / price_std
            z_threshold = self.validation_thresholds['z_score_threshold']
            
            validation_results['checks']['z_score'] = {
                'value': z_score,
                'threshold': z_threshold,
                'status': 'PASS' if z_score <= z_threshold else 'FAIL',
                'description': f'Z-score: {z_score:.2f} (threshold: {z_threshold})'
            }
            
            if z_score > z_threshold:
                validation_results['warnings'].append(f"High Z-score: {z_score:.2f}")
                validation_results['overall_status'] = 'WARNING'
        
        if 'close' in historical_data.columns and len(historical_data) > 5:
            recent_trend = historical_data['close'].tail(5).pct_change().mean()
            predicted_direction = 1 if predicted_price > current_price else -1
            trend_direction = 1 if recent_trend > 0 else -1
            
            trend_consistent = predicted_direction == trend_direction
            
            validation_results['checks']['trend_consistency'] = {
                'recent_trend': recent_trend * 100,
                'predicted_direction': 'UP' if predicted_direction > 0 else 'DOWN',
                'trend_direction': 'UP' if trend_direction > 0 else 'DOWN',
                'consistent': trend_consistent,
                'status': 'PASS' if trend_consistent else 'WARNING',
                'description': 'Prediction direction vs recent trend'
            }
            
            if not trend_consistent:
                validation_results['warnings'].append("Prediction contradicts recent trend")
        
        current_time = datetime.now()
        is_weekend = current_time.weekday() >= 5
        
        validation_results['checks']['market_hours'] = {
            'is_weekend': is_weekend,
            'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'WARNING' if is_weekend else 'PASS',
            'description': 'Market hours validation'
        }
        
        if is_weekend:
            validation_results['warnings'].append("Prediction made during weekend (markets closed)")
        
        self._generate_recommendations(validation_results)
        
        self._print_validation_summary(validation_results)
        
        return validation_results
    
    def detect_prediction_anomalies(self, predictions: List[float], 
                                  timestamps: List[datetime], 
                                  pair: str) -> Dict:
        
        print(f" DETECTING ANOMALIES IN PREDICTIONS FOR {pair}")
        print("=" * 60)
        
        if len(predictions) < 10:
            return {
                'status': 'INSUFFICIENT_DATA',
                'message': 'Need at least 10 predictions for anomaly detection'
            }
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        
        z_scores = np.abs(stats.zscore(predictions))
        z_outliers = np.where(z_scores > self.validation_thresholds['z_score_threshold'])[0]
        
        iso_forest = IsolationForest(
            contamination=self.validation_thresholds['isolation_contamination'],
            random_state=42
        )
        iso_outliers = iso_forest.fit_predict(predictions_array)
        iso_anomaly_indices = np.where(iso_outliers == -1)[0]
        
        Q1 = np.percentile(predictions, 25)
        Q3 = np.percentile(predictions, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = np.where((predictions < lower_bound) | (predictions > upper_bound))[0]
        
        if len(predictions) > 1:
            changes = np.diff(predictions) / predictions[:-1]
            large_jumps = np.where(np.abs(changes) > self.validation_thresholds['max_daily_change'])[0]
        else:
            large_jumps = []
        
        all_anomalies = set(z_outliers) | set(iso_anomaly_indices) | set(iqr_outliers) | set(large_jumps)
        
        anomaly_results = {
            'pair': pair,
            'total_predictions': len(predictions),
            'anomalies_detected': len(all_anomalies),
            'anomaly_percentage': (len(all_anomalies) / len(predictions)) * 100,
            'methods': {
                'z_score': {
                    'outliers': len(z_outliers),
                    'indices': z_outliers.tolist(),
                    'threshold': self.validation_thresholds['z_score_threshold']
                },
                'isolation_forest': {
                    'outliers': len(iso_anomaly_indices),
                    'indices': iso_anomaly_indices.tolist(),
                    'contamination': self.validation_thresholds['isolation_contamination']
                },
                'iqr': {
                    'outliers': len(iqr_outliers),
                    'indices': iqr_outliers.tolist(),
                    'bounds': [lower_bound, upper_bound]
                },
                'large_jumps': {
                    'outliers': len(large_jumps),
                    'indices': large_jumps.tolist(),
                    'threshold': self.validation_thresholds['max_daily_change']
                }
            },
            'anomaly_details': []
        }
        
        for idx in sorted(all_anomalies):
            if idx < len(predictions) and idx < len(timestamps):
                anomaly_details = {
                    'index': idx,
                    'timestamp': timestamps[idx].isoformat() if idx < len(timestamps) else 'Unknown',
                    'value': predictions[idx],
                    'detected_by': []
                }
                
                if idx in z_outliers:
                    anomaly_details['detected_by'].append('z_score')
                if idx in iso_anomaly_indices:
                    anomaly_details['detected_by'].append('isolation_forest')
                if idx in iqr_outliers:
                    anomaly_details['detected_by'].append('iqr')
                if idx in large_jumps:
                    anomaly_details['detected_by'].append('large_jump')
                
                anomaly_results['anomaly_details'].append(anomaly_details)
        
        print(f" Anomaly Detection Summary:")
        print(f"   Total Predictions: {len(predictions)}")
        print(f"   Anomalies Detected: {len(all_anomalies)} ({(len(all_anomalies)/len(predictions)*100):.1f}%)")
        print(f"   Z-Score Outliers: {len(z_outliers)}")
        print(f"   Isolation Forest: {len(iso_anomaly_indices)}")
        print(f"   IQR Outliers: {len(iqr_outliers)}")
        print(f"   Large Jumps: {len(large_jumps)}")
        
        if len(all_anomalies) > 0:
            print(f"\n  Anomalous Predictions:")
            for detail in anomaly_results['anomaly_details'][:5]:
                print(f"   Index {detail['index']}: {detail['value']:.6f} "
                      f"(detected by: {', '.join(detail['detected_by'])})")
        
        return anomaly_results
    
    def create_prediction_validation_plot(self, validation_results: Dict, 
                                        historical_data: pd.DataFrame,
                                        save_path: str = None):
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Prediction Validation - {validation_results["pair"]}', 
                     fontsize=16, fontweight='bold')
        
        if 'close' in historical_data.columns:
            recent_data = historical_data.tail(30)
            ax1.plot(recent_data.index, recent_data['close'], 'b-', label='Historical Prices', linewidth=2)
            
            last_date = recent_data.index[-1]
            next_date = last_date + timedelta(days=1)
            
            ax1.plot([last_date, next_date], 
                    [validation_results['current_price'], validation_results['predicted_price']], 
                    'r--', linewidth=2, label='Prediction')
            ax1.scatter([next_date], [validation_results['predicted_price']], 
                       color='red', s=100, zorder=5, label='Predicted Price')
            
            ax1.set_title('Price History and Prediction')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        checks = validation_results['checks']
        check_names = list(checks.keys())
        check_statuses = [1 if checks[name]['status'] == 'PASS' else 
                         0.5 if checks[name]['status'] == 'WARNING' else 0 
                         for name in check_names]
        
        colors = ['green' if s == 1 else 'orange' if s == 0.5 else 'red' for s in check_statuses]
        
        ax2.barh(check_names, check_statuses, color=colors, alpha=0.7)
        ax2.set_xlim(0, 1)
        ax2.set_title('Validation Checks')
        ax2.set_xlabel('Status (1=Pass, 0.5=Warning, 0=Fail)')
        
        if 'close' in historical_data.columns:
            returns = historical_data['close'].pct_change().dropna()
            predicted_return = validation_results['predicted_price'] / validation_results['current_price'] - 1
            
            ax3.hist(returns, bins=30, alpha=0.7, density=True, label='Historical Returns')
            ax3.axvline(predicted_return, color='red', linestyle='--', linewidth=2, 
                       label=f'Predicted Return: {predicted_return*100:.2f}%')
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Daily Return')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        ax4.axis('off')
        
        summary_text = f
        
        for warning in validation_results['warnings']:
            summary_text += f"• {warning}\n"
        
        if not validation_results['warnings']:
            summary_text += "• No warnings detected\n"
        
        summary_text += "\nRECOMMENDations:\n"
        for rec in validation_results['recommendations']:
            summary_text += f"• {rec}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Validation plot saved to {save_path}")
        
        plt.show()
    
    def _generate_recommendations(self, validation_results: Dict):
        
        recommendations = []
        
        failed_checks = [name for name, check in validation_results['checks'].items() 
                        if check['status'] == 'FAIL']
        
        if 'daily_change' in failed_checks:
            recommendations.append("Consider using ensemble methods to reduce prediction volatility")
            recommendations.append("Review model training data for similar market conditions")
        
        if 'volatility' in failed_checks:
            recommendations.append("Increase model regularization to prevent overfitting")
            recommendations.append("Consider using volatility-adjusted predictions")
        
        if 'z_score' in failed_checks:
            recommendations.append("Review recent market events that might affect predictions")
            recommendations.append("Consider retraining models with more recent data")
        
        if len(validation_results['warnings']) > 2:
            recommendations.append("Multiple warnings detected - consider manual review")
            recommendations.append("Use conservative position sizing for this prediction")
        
        if not recommendations:
            recommendations.append("Prediction passes all validation checks")
            recommendations.append("Consider normal position sizing")
        
        validation_results['recommendations'] = recommendations
    
    def _print_validation_summary(self, validation_results: Dict):
        
        print(f"\n VALIDATION SUMMARY:")
        print(f"   Overall Status: {validation_results['overall_status']}")
        print(f"   Current Price: {validation_results['current_price']:.6f}")
        print(f"   Predicted Price: {validation_results['predicted_price']:.6f}")
        print(f"   Predicted Change: {validation_results['prediction_change']:+.2f}%")
        
        print(f"\n VALIDATION CHECKS:")
        for name, check in validation_results['checks'].items():
            status_icon = "" if check['status'] == 'PASS' else "" if check['status'] == 'WARNING' else ""
            print(f"   {status_icon} {name}: {check['description']}")
        
        if validation_results['warnings']:
            print(f"\n  WARNINGS:")
            for warning in validation_results['warnings']:
                print(f"   • {warning}")
        
        print(f"\n RECOMMENDATIONS:")
        for rec in validation_results['recommendations']:
            print(f"   • {rec}")

def main():
    
    validator = PredictionValidator()
    
    print(" TESTING PREDICTION VALIDATOR")
    print("=" * 50)
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = 1.1 + 0.1 * np.random.randn(len(dates)).cumsum() * 0.01
    historical_data = pd.DataFrame({'close': prices}, index=dates)
    
    current_price = prices[-1]
    predicted_price = current_price * 1.02
    
    results = validator.validate_prediction(
        current_price=current_price,
        predicted_price=predicted_price,
        historical_data=historical_data,
        pair='EURUSD=X'
    )
    
    predictions = [1.1 + 0.01 * np.random.randn() for _ in range(50)]
    predictions[25] = 1.5
    timestamps = [datetime.now() - timedelta(days=i) for i in range(50)]
    
    anomaly_results = validator.detect_prediction_anomalies(
        predictions=predictions,
        timestamps=timestamps,
        pair='EURUSD=X'
    )
    
    print("\n Prediction validator test completed!")

if __name__ == "__main__":
    main() 