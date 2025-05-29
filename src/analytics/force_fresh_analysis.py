#!/usr/bin/env python3

import os
import shutil
import glob
from datetime import datetime
from neural_network_training import NeuralNetworkTrainer

def clear_old_results():
    
    print("CLEARING OLD RESULTS")
    print("=" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    if os.path.exists("plots") and os.listdir("plots"):
        print(f" Backing up old plots to {backup_dir}/plots/")
        os.makedirs(f"{backup_dir}/plots", exist_ok=True)
        for file in glob.glob("plots/*"):
            if os.path.isfile(file):
                shutil.copy2(file, f"{backup_dir}/plots/")
    
    if os.path.exists("reports") and os.listdir("reports"):
        print(f" Backing up old reports to {backup_dir}/reports/")
        os.makedirs(f"{backup_dir}/reports", exist_ok=True)
        for file in glob.glob("reports/*"):
            if os.path.isfile(file):
                shutil.copy2(file, f"{backup_dir}/reports/")
    
    print("  Clearing current plots...")
    if os.path.exists("plots"):
        for file in glob.glob("plots/*.png"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
    
    print("  Clearing current reports...")
    if os.path.exists("reports"):
        for file in glob.glob("reports/*.txt"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
        for file in glob.glob("reports/*.csv"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
    
    print("  Clearing old models to force retraining...")
    if os.path.exists("models"):
        for file in glob.glob("models/EURUSD=X_*.pth"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
        for file in glob.glob("models/EURUSD=X_*.joblib"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
    
    print("  Clearing processed data to force fresh fetch...")
    if os.path.exists("data"):
        for file in glob.glob("data/EURUSD=X_*.csv"):
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Warning: Could not remove {file}: {e}")
    
    print(" Old results cleared successfully!")
    print(f" Backup saved in: {backup_dir}/")
    return backup_dir

def run_fresh_analysis():
    
    print("\n RUNNING FRESH ANALYSIS")
    print("=" * 40)
    
    try:
        trainer = NeuralNetworkTrainer()
        
        print(" Starting complete neural network analysis...")
        trainer.run_complete_neural_analysis(
            pairs=['EURUSD=X'],
            period='3y',
            sequence_length=10
        )
        
        print("\n Fresh analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f" Error during fresh analysis: {e}")
        return False

def verify_new_results():
    
    print("\n VERIFYING NEW RESULTS")
    print("=" * 40)
    
    if os.path.exists("plots"):
        plot_files = [f for f in os.listdir("plots") if f.endswith('.png')]
        if plot_files:
            print(f" Generated {len(plot_files)} new plots:")
            for file in plot_files:
                file_path = os.path.join("plots", file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"    {file} (created: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(" No new plots generated")
    
    if os.path.exists("reports"):
        report_files = [f for f in os.listdir("reports") if f.endswith('.txt') or f.endswith('.csv')]
        if report_files:
            print(f" Generated {len(report_files)} new reports:")
            for file in report_files:
                file_path = os.path.join("reports", file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"    {file} (created: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(" No new reports generated")
    
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if 'EURUSD=X' in f and (f.endswith('.pth') or f.endswith('.joblib'))]
        if model_files:
            print(f" Generated {len(model_files)} new models:")
            for file in model_files:
                file_path = os.path.join("models", file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"   models/{file} (created: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(" No new models generated")

def main():
    
    print(" FORCE FRESH ANALYSIS")
    print("=" * 50)
    print("This will clear all old results and generate fresh analysis")
    print("=" * 50)
    
    backup_dir = clear_old_results()
    
    success = run_fresh_analysis()
    
    verify_new_results()
    
    if success:
        print("\n" + "=" * 50)
        print(" FRESH ANALYSIS COMPLETE!")
        print("=" * 50)
        print(" Check the following for new results:")
        print("    plots/ - New visualizations")
        print("    reports/ - New analysis reports")
        print("    models/ - New trained models")
        print(f"    {backup_dir}/ - Backup of old results")
    else:
        print("\n" + "=" * 50)
        print(" FRESH ANALYSIS FAILED!")
        print("=" * 50)
        print("Check the error messages above for details")

if __name__ == "__main__":
    main() 