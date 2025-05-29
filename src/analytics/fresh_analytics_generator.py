#!/usr/bin/env python3

import os
import glob
import shutil
import sys
from pathlib import Path
from datetime import datetime

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from neural_networks.neural_network_training import NeuralNetworkTrainer

def clear_old_analytics():
    
    print("  CLEARING OLD ANALYTICS")
    print("=" * 40)
    
    clear_patterns = [
        'plots/*.png',
        'reports/*.txt',
        'reports/*.csv',
        'reports/*.json',
        'predictions/*.json',
        'data/*_predictions.csv',
        'data/*_processed.csv',
        'models/*.pth',
        'models/*.pkl'
    ]
    
    cleared_count = 0
    
    for pattern in clear_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"  Deleted: {file_path}")
                cleared_count += 1
            except Exception as e:
                print(f"  Could not delete {file_path}: {e}")
    
    cache_dirs = ['__pycache__', '.pytest_cache']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"  Deleted cache: {cache_dir}")
            except Exception as e:
                print(f"  Could not delete {cache_dir}: {e}")
    
    print(f" Cleared {cleared_count} old files")
    return cleared_count

def generate_fresh_analytics(currency_pair='EURUSD=X', timeframe='2y'):
    
    print(f"\n GENERATING FRESH ANALYTICS")
    print("=" * 40)
    print(f"Currency Pair: {currency_pair}")
    print(f"Timeframe: {timeframe}")
    print(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer = NeuralNetworkTrainer()
        
        print("\n Fetching fresh market data...")
        trainer.fetch_and_prepare_data([currency_pair], timeframe)
        
        print(f"\n Training traditional ML models...")
        traditional_performance = trainer.train_traditional_models(currency_pair)
        
        print(f"\n Training neural networks...")
        nn_performance = trainer.train_neural_networks(currency_pair)
        
        print(f"\n Comparing model performance...")
        comparison_df = trainer.compare_all_models()
        
        print(f"\n Generating fresh predictions...")
        predictions = trainer.generate_neural_predictions(currency_pair)
        
        print(f"\n Creating fresh charts...")
        trainer.create_performance_visualizations(currency_pair)
        
        print(f"\n Creating summary report...")
        trainer.create_neural_summary_report(currency_pair, comparison_df)
        
        return True, predictions
        
    except Exception as e:
        print(f" Error generating fresh analytics: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_fresh_charts(currency_pair='EURUSD=X'):
    
    print(f"\n CREATING FRESH CHARTS")
    print("=" * 30)
    
    try:
        from charts.create_prediction_charts import create_prediction_vs_current_chart
        
        chart1 = create_prediction_vs_current_chart()
        
        charts_created = 1 if chart1 else 0
        print(f" Created {charts_created}/1 fresh charts")
        
        return charts_created > 0
        
    except Exception as e:
        print(f" Error creating fresh charts: {e}")
        return False

def run_complete_fresh_analysis(currency_pair='EURUSD=X', timeframe='2y'):
    
    print(" COMPLETE FRESH ANALYTICS PIPELINE")
    print("=" * 60)
    print("This will delete ALL old results and create fresh ones")
    print("=" * 60)
    
    start_time = datetime.now()
    
    cleared_files = clear_old_analytics()
    
    analytics_success, predictions = generate_fresh_analytics(currency_pair, timeframe)
    
    charts_success = create_fresh_charts(currency_pair)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print(" FRESH ANALYTICS GENERATION SUMMARY")
    print("=" * 60)
    
    print(f"Total Time: {duration:.1f} seconds")
    print(f"  Files Cleared: {cleared_files}")
    print(f" Analytics: {' SUCCESS' if analytics_success else ' FAILED'}")
    print(f" Charts: {' SUCCESS' if charts_success else ' FAILED'}")
    
    if analytics_success and predictions:
        print(f"\n FRESH PREDICTION RESULTS:")
        print(f"   Current Price: {predictions['current_price']:.6f}")
        print(f"   Predicted Price: {predictions['predicted_price']:.6f}")
        print(f"   Expected Change: {predictions['change_percent']:+.2f}%")
        print(f"   Model: {predictions['model']}")
    
    if analytics_success and charts_success:
        print(f"\n FRESH FILES GENERATED:")
        print(f"    plots/prediction_vs_current_analysis.png")
        print(f"    plots/{currency_pair}_neural_network_comparison.png")
        print(f"    reports/{currency_pair}_neural_analysis_report.txt")
        print(f"    predictions/{currency_pair}_neural_prediction.json")
        
        print(f"\n FRESH ANALYTICS COMPLETE!")
        print(f"All old data deleted, new analysis generated with timestamp: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n  Some components failed - check logs above")
    
    return analytics_success and charts_success

if __name__ == "__main__":
    run_complete_fresh_analysis('EURUSD=X', '2y') 