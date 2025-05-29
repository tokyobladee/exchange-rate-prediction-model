#!/usr/bin/env python3

import sys
import os
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from launchers.easy_analytics_launcher import run_with_parameters, CURRENCY_PAIRS, TIMEFRAMES

PAIR_SHORTCUTS = {
    'EURUSD': '1', 'EUR': '1', 'EURUSD=X': '1',
    'GBPUSD': '2', 'GBP': '2', 'GBPUSD=X': '2', 
    'USDJPY': '3', 'JPY': '3', 'USDJPY=X': '3',
    'AUDUSD': '4', 'AUD': '4', 'AUDUSD=X': '4',
    'USDCAD': '5', 'CAD': '5', 'USDCAD=X': '5',
    'USDCHF': '6', 'CHF': '6', 'USDCHF=X': '6',
    'NZDUSD': '7', 'NZD': '7', 'NZDUSD=X': '7',
    'EURGBP': '8', 'EURGBP=X': '8',
    'EURJPY': '9', 'EURJPY=X': '9',
    'GBPJPY': '10', 'GBPJPY=X': '10'
}

TIME_SHORTCUTS = {
    '1mo': '1', '1m': '1', '30d': '1',
    '3mo': '2', '3m': '2', '90d': '2',
    '6mo': '3', '6m': '3', '180d': '3',
    '1y': '4', '1yr': '4', '365d': '4',
    '2y': '5', '2yr': '5', '730d': '5',
    '5y': '6', '5yr': '6',
    'max': '7', 'all': '7'
}

def show_help():
    
    print("QUICK ANALYSIS SCRIPT")
    print("=" * 40)
    print("Usage: python src/launchers/quick_analysis.py [PAIR] [TIMEFRAME]")
    print()
    print("Available Currency Pairs:")
    for shortcut, number in PAIR_SHORTCUTS.items():
        if shortcut in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
            pair_info = CURRENCY_PAIRS[number]
            print(f"  {shortcut:8s} - {pair_info['name']}")
    
    print("\nAvailable Timeframes:")
    for shortcut, number in TIME_SHORTCUTS.items():
        if shortcut in ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']:
            time_info = TIMEFRAMES[number]
            print(f"  {shortcut:8s} - {time_info['name']}")
    
    print("\nExamples:")
    print("  python src/launchers/quick_analysis.py                # EUR/USD, 2 years (default)")
    print("  python src/launchers/quick_analysis.py GBPUSD 1y      # GBP/USD, 1 year")
    print("  python src/launchers/quick_analysis.py JPY 6mo        # USD/JPY, 6 months")
    print("  python src/launchers/quick_analysis.py AUD            # AUD/USD, 2 years (default timeframe)")

def parse_arguments():
    
    args = sys.argv[1:]
    
    if len(args) > 0:
        first_arg = args[0].lower()
        if first_arg in ['-h', '--help', 'help', '?']:
            show_help()
            return None, None
    
    pair_number = '1'
    timeframe_number = '5'
    
    if len(args) >= 1:
        pair_arg = args[0].upper()
        
        if pair_arg in PAIR_SHORTCUTS:
            pair_number = PAIR_SHORTCUTS[pair_arg]
        else:
            print(f"Unknown currency pair: {pair_arg}")
            print("Use 'python src/launchers/quick_analysis.py help' for available options")
            return None, None
    
    if len(args) >= 2:
        time_arg = args[1].lower()
        if time_arg in TIME_SHORTCUTS:
            timeframe_number = TIME_SHORTCUTS[time_arg]
        else:
            print(f"Unknown timeframe: {time_arg}")
            print("Use 'python src/launchers/quick_analysis.py help' for available options")
            return None, None
    
    return pair_number, timeframe_number

def main():
    
    pair_number, timeframe_number = parse_arguments()
    
    if pair_number is None:
        return
    
    pair_info = CURRENCY_PAIRS[pair_number]
    time_info = TIMEFRAMES[timeframe_number]
    
    print("QUICK ANALYSIS STARTING")
    print("=" * 40)
    print(f"Currency Pair: {pair_info['name']} ({pair_info['symbol']})")
    print(f"Timeframe: {time_info['name']} ({time_info['period']})")
    print(f"Description: {pair_info['description']}")
    print("=" * 40)
    print("Generating fresh analytics (deleting old results)...")
    
    success = run_with_parameters(pair_number, timeframe_number)
    
    if success:
        print("\nQUICK ANALYSIS COMPLETED!")
        print("Fresh charts and reports generated")
    else:
        print("\nAnalysis failed - check error messages above")

if __name__ == "__main__":
    main() 