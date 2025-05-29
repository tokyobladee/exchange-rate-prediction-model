#!/usr/bin/env python3

import os
import subprocess
import sys

def open_graphs():

    print(" Opening Prediction Graphs...")
    print("=" * 40)
    
    graphs = [
        'plots/prediction_timeline.png',
        'plots/prediction_analysis_complete.png', 
        'plots/EURUSD=X_predictions_vs_actual.png',
        'plots/EURUSD=X_dashboard.png',
        'plots/EURUSD=X_model_performance.png',
        'plots/EURUSD=X_price_chart.png'
    ]
    
    opened_count = 0
    
    for graph in graphs:
        if os.path.exists(graph):
            try:
                if sys.platform.startswith('win'):
                    os.startfile(graph)
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['open', graph])
                else:
                    subprocess.run(['xdg-open', graph])
                
                print(f" Opened: {graph}")
                opened_count += 1
                
            except Exception as e:
                print(f" Failed to open {graph}: {e}")
        else:
            print(f" File not found: {graph}")
    
    print(f"\n Opened {opened_count} prediction graphs!")
    print("\n Key Graphs Explained:")
    print("=" * 40)
    print(" prediction_timeline.png - Shows historical vs future predictions")
    print(" prediction_analysis_complete.png - Complete 4-panel analysis")
    print(" EURUSD=X_predictions_vs_actual.png - Model accuracy visualization")
    print(" EURUSD=X_dashboard.png - Overview dashboard")
    print(" EURUSD=X_model_performance.png - ML model comparison")
    print(" EURUSD=X_price_chart.png - Price history chart")

def main():
    open_graphs()

if __name__ == "__main__":
    main() 