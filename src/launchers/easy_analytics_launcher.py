#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from datetime import datetime

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from analytics.fresh_analytics_generator import run_complete_fresh_analysis

CURRENCY_PAIRS = {
    '1': {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'description': 'Euro vs US Dollar (Most Popular)'},
    '2': {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'description': 'British Pound vs US Dollar'},
    '3': {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'description': 'US Dollar vs Japanese Yen'},
    '4': {'symbol': 'AUDUSD=X', 'name': 'AUD/USD', 'description': 'Australian Dollar vs US Dollar'},
    '5': {'symbol': 'USDCAD=X', 'name': 'USD/CAD', 'description': 'US Dollar vs Canadian Dollar'},
    '6': {'symbol': 'USDCHF=X', 'name': 'USD/CHF', 'description': 'US Dollar vs Swiss Franc'},
    '7': {'symbol': 'NZDUSD=X', 'name': 'NZD/USD', 'description': 'New Zealand Dollar vs US Dollar'},
    '8': {'symbol': 'EURGBP=X', 'name': 'EUR/GBP', 'description': 'Euro vs British Pound'},
    '9': {'symbol': 'EURJPY=X', 'name': 'EUR/JPY', 'description': 'Euro vs Japanese Yen'},
    '10': {'symbol': 'GBPJPY=X', 'name': 'GBP/JPY', 'description': 'British Pound vs Japanese Yen'}
}

TIMEFRAMES = {
    '1': {'period': '1mo', 'name': '1 Month', 'description': 'Short-term analysis (30 days)', 'recommended': 'Day trading'},
    '2': {'period': '3mo', 'name': '3 Months', 'description': 'Short-term analysis (90 days)', 'recommended': 'Swing trading'},
    '3': {'period': '6mo', 'name': '6 Months', 'description': 'Medium-term analysis (180 days)', 'recommended': 'Position trading'},
    '4': {'period': '1y', 'name': '1 Year', 'description': 'Medium-term analysis (365 days)', 'recommended': 'Trend analysis'},
    '5': {'period': '2y', 'name': '2 Years', 'description': 'Long-term analysis (730 days)', 'recommended': 'Investment analysis'},
    '6': {'period': '5y', 'name': '5 Years', 'description': 'Long-term analysis (1825 days)', 'recommended': 'Strategic planning'},
    '7': {'period': 'max', 'name': 'Maximum', 'description': 'All available data', 'recommended': 'Historical analysis'}
}

def display_currency_pairs():
    
    print(" AVAILABLE CURRENCY PAIRS")
    print("=" * 50)
    
    for key, pair in CURRENCY_PAIRS.items():
        print(f"{key:2s}. {pair['name']:8s} - {pair['description']}")
    
    print("\n Tip: EUR/USD is the most liquid and popular pair")

def display_timeframes():
    
    print("\n AVAILABLE TIMEFRAMES")
    print("=" * 50)
    
    for key, timeframe in TIMEFRAMES.items():
        print(f"{key}. {timeframe['name']:10s} - {timeframe['description']:25s} | Best for: {timeframe['recommended']}")
    
    print("\n Tip: 2 Years provides good balance of data and training time")

def get_user_choice(prompt, options, default=None):
    
    while True:
        if default:
            choice = input(f"{prompt} (default: {default}): ").strip()
            if not choice:
                return default
        else:
            choice = input(f"{prompt}: ").strip()
        
        if choice in options:
            return choice
        else:
            print(f" Invalid choice. Please select from: {', '.join(options.keys())}")

def confirm_selection(currency_pair, timeframe):
    
    pair_info = CURRENCY_PAIRS[currency_pair]
    time_info = TIMEFRAMES[timeframe]
    
    print("\n" + "=" * 60)
    print(" ANALYSIS CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Currency Pair: {pair_info['name']} ({pair_info['symbol']})")
    print(f"Description: {pair_info['description']}")
    print(f"Timeframe: {time_info['name']} ({time_info['period']})")
    print(f"Analysis Type: {time_info['description']}")
    print(f"Recommended For: {time_info['recommended']}")
    print(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    confirm = input("\n Start fresh analysis with these settings? (y/n, default: y): ").strip().lower()
    return confirm in ['', 'y', 'yes']

def quick_start_menu():
    
    print(" QUICK START OPTIONS")
    print("=" * 30)
    print("1. EUR/USD - 2 Years (Recommended)")
    print("2. GBP/USD - 1 Year (Popular)")
    print("3. USD/JPY - 6 Months (Active)")
    print("4. Custom Configuration")
    
    choice = get_user_choice("Select quick start option", {'1': '1', '2': '2', '3': '3', '4': '4'}, '1')
    
    if choice == '1':
        return '1', '5'
    elif choice == '2':
        return '2', '4'
    elif choice == '3':
        return '3', '3'
    else:
        return None, None

def main():
    
    print(" CURRENCY PREDICTION ANALYTICS LAUNCHER")
    print("=" * 60)
    print(" This will generate FRESH analytics (deletes old results)")
    print(" Creates new charts, predictions, and reports")
    print("=" * 60)
    
    print("\n Choose your approach:")
    print("1. Quick Start (recommended configurations)")
    print("2. Custom Configuration")
    
    approach = get_user_choice("Select approach", {'1': '1', '2': '2'}, '1')
    
    if approach == '1':
        currency_choice, timeframe_choice = quick_start_menu()
        
        if currency_choice is None:
            approach = '2'
    
    if approach == '2':
        display_currency_pairs()
        currency_choice = get_user_choice("Select currency pair", CURRENCY_PAIRS, '1')
        
        display_timeframes()
        timeframe_choice = get_user_choice("Select timeframe", TIMEFRAMES, '5')
    
    selected_pair = CURRENCY_PAIRS[currency_choice]['symbol']
    selected_timeframe = TIMEFRAMES[timeframe_choice]['period']
    
    if confirm_selection(currency_choice, timeframe_choice):
        print("\n STARTING FRESH ANALYTICS GENERATION...")
        print("This may take a few minutes...")
        
        success = run_complete_fresh_analysis(selected_pair, selected_timeframe)
        
        if success:
            print("\n" + "" * 20)
            print(" FRESH ANALYTICS COMPLETED SUCCESSFULLY!")
            print("" * 20)
            print(f"\n Check these fresh files:")
            print(f"    plots/prediction_vs_current_analysis.png - Main prediction chart")
            print(f"    plots/detailed_price_analysis.png - Detailed analysis")
            print(f"    plots/model_performance_analysis.png - Model comparison")
            print(f"    plots/prediction_confidence_analysis.png - Confidence metrics")
            print(f"    reports/{selected_pair}_neural_analysis_report.txt - Full report")
        else:
            print("\n Analytics generation failed. Check error messages above.")
    else:
        print("\n Analysis cancelled by user.")

def run_with_parameters(pair_number='1', timeframe_number='5'):
    
    selected_pair = CURRENCY_PAIRS[pair_number]['symbol']
    selected_timeframe = TIMEFRAMES[timeframe_number]['period']
    
    print(f" Running automated analysis:")
    print(f"   Pair: {CURRENCY_PAIRS[pair_number]['name']}")
    print(f"   Timeframe: {TIMEFRAMES[timeframe_number]['name']}")
    
    return run_complete_fresh_analysis(selected_pair, selected_timeframe)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Analysis cancelled by user (Ctrl+C)")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc() 