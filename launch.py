#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from datetime import datetime

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def fetch_fresh_data():
    print("INITIALIZING CURRENCY PREDICTION SYSTEM")
    print("=" * 50)
    print("Fetching fresh market data...")
    
    try:
        from utils.data_fetcher import CurrencyDataFetcher
        
        fetcher = CurrencyDataFetcher()
        
        main_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        
        for pair in main_pairs:
            print(f"  Updating {pair}...")
            data = fetcher.get_currency_data(pair, period='5d')
            if data is not None and len(data) > 0:
                print(f"    {len(data)} records fetched")
            else:
                print(f"    Failed to fetch data")
        
        print("Market data updated successfully")
        print("=" * 50)
        
    except Exception as e:
        print(f"Warning: Could not fetch fresh data: {e}")
        print("System will continue with existing data")
        print("=" * 50)

def show_main_menu():
    print("\nCURRENCY PREDICTION ANALYTICS SYSTEM")
    print("=" * 60)
    print("Fresh Analytics Generation (Deletes Old Results)")
    print("Neural Networks + Traditional ML Models")
    print("Comprehensive Charts and Predictions")
    print("=" * 60)
    
    print("\nMAIN OPTIONS:")
    print("1. Quick Analysis (EUR/USD, 2 years)")
    print("2. Command Line Interface")
    print("3. Interactive Menu (Choose pair & timeframe)")
    print("4. Advanced Analytics")
    print("5. Utilities & Tools")
    print("6. View Current Predictions")
    print("7. Open Charts")
    print("8. Help & Documentation")
    print("9. Exit")

def quick_analysis():
    print("\nQUICK ANALYSIS - EUR/USD, 2 YEARS")
    print("=" * 45)
    
    try:
        from launchers.quick_analysis import main as quick_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['quick_analysis.py']
        
        quick_main()
        
        sys.argv = original_argv
        
        print("\nQuick analysis completed!")
        print("Check the plots/ folder for fresh charts")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def command_line_interface():
    print("\nCOMMAND LINE INTERFACE")
    print("=" * 30)
    print("Usage examples:")
    print("  python src/launchers/quick_analysis.py")
    print("  python src/launchers/quick_analysis.py GBPUSD 1y")
    print("  python src/launchers/quick_analysis.py JPY 6mo")
    print("\nFor help: python src/launchers/quick_analysis.py help")

def interactive_menu():
    print("\nLAUNCHING INTERACTIVE MENU...")
    
    try:
        from launchers.easy_analytics_launcher import main
        main()
    except Exception as e:
        print(f"Error launching interactive menu: {e}")
        import traceback
        traceback.print_exc()

def advanced_analytics():
    print("\nADVANCED ANALYTICS")
    print("=" * 25)
    print("1. Fresh Analytics Generator")
    print("2. Neural Network Training")
    print("3. Custom Chart Creation")
    print("4. Prediction Validation")
    print("5. Data Management")
    print("6. Model Retraining")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == '1':
        try:
            from analytics.fresh_analytics_generator import run_complete_fresh_analysis
            run_complete_fresh_analysis()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '2':
        try:
            from neural_networks.neural_network_training import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '3':
        try:
            from charts.create_prediction_charts import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '4':
        try:
            from utils.prediction_validator import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '5':
        try:
            from utils.data_manager import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '6':
        try:
            from utils.retrain_models import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")

def utilities_menu():
    print("\nUTILITIES & TOOLS")
    print("=" * 20)
    print("1. Show Current Predictions")
    print("2. Diagnose System Issues")
    print("3. Test System Components")
    print("4. Currency Manager")
    print("5. View Results")
    print("6. System Status")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == '1':
        show_current_predictions()
    elif choice == '2':
        diagnose_system_issues()
    elif choice == '3':
        test_system_components()
    elif choice == '4':
        try:
            from utils.currency_manager import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '5':
        try:
            from utils.show_results import main
            main()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '6':
        show_system_status()
    else:
        print("Invalid choice")

def show_current_predictions():
    print("\nCURRENT PREDICTIONS")
    print("=" * 25)
    
    try:
        from utils.show_prediction_summary import main
        main()
    except Exception as e:
        print(f"Error showing predictions: {e}")
        
        prediction_files = []
        if os.path.exists('predictions'):
            for file in os.listdir('predictions'):
                if file.endswith('.json'):
                    prediction_files.append(file)
        
        if prediction_files:
            print(f"Found {len(prediction_files)} prediction files:")
            for file in prediction_files:
                print(f"  - {file}")
        else:
            print("No prediction files found")

def diagnose_system_issues():
    print("\nSYSTEM DIAGNOSTICS")
    print("=" * 25)
    
    try:
        from tests.diagnose_prediction_issues import main
        main()
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        
        print("\nBasic system check:")
        
        directories = ['src', 'data', 'models', 'plots', 'predictions', 'reports']
        for directory in directories:
            if os.path.exists(directory):
                print(f"  {directory}/ - OK")
            else:
                print(f"  {directory}/ - MISSING")
        
        key_files = ['src/utils/data_fetcher.py', 'src/utils/feature_engineering.py', 'src/models/ml_models.py']
        for file in key_files:
            if os.path.exists(file):
                print(f"  {file} - OK")
            else:
                print(f"  {file} - MISSING")

def test_system_components():
    print("\nTESTING SYSTEM COMPONENTS")
    print("=" * 35)
    
    try:
        import test_system
        test_system.main()
    except Exception as e:
        print(f"Error running comprehensive test: {e}")
        
        print("\nBasic component test:")
        
        try:
            from utils.data_fetcher import CurrencyDataFetcher
            print("  Data Fetcher - OK")
        except Exception as e:
            print(f"  Data Fetcher - ERROR: {e}")
        
        try:
            from utils.feature_engineering import FeatureEngineer
            print("  Feature Engineering - OK")
        except Exception as e:
            print(f"  Feature Engineering - ERROR: {e}")
        
        try:
            from models.ml_models import MLModels
            print("  ML Models - OK")
        except Exception as e:
            print(f"  ML Models - ERROR: {e}")

def show_system_status():
    print("\nSYSTEM STATUS")
    print("=" * 20)
    
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        print(f"Data Files: {len(data_files)}")
        if data_files:
            latest_file = max([os.path.join('data', f) for f in data_files], key=os.path.getmtime)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
            print(f"Latest Data: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith(('.pkl', '.pth'))]
        print(f"Model Files: {len(model_files)}")
    
    if os.path.exists('plots'):
        plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
        print(f"Chart Files: {len(plot_files)}")

def view_predictions():
    print("Opening prediction viewer...")
    open_charts()

def open_charts():
    print("\nOPENING CHARTS")
    print("=" * 20)
    
    try:
        from charts.open_graphs import main
        main()
    except Exception as e:
        print(f"Error opening charts: {e}")
        
        if os.path.exists('plots'):
            chart_files = [f for f in os.listdir('plots') if f.endswith('.png')]
            if chart_files:
                print(f"Available charts in plots/ folder:")
                for chart in chart_files:
                    print(f"  - {chart}")
                print("\nOpen the plots/ folder to view charts manually")
            else:
                print("No chart files found in plots/ folder")
        else:
            print("plots/ folder not found")

def show_help():
    print("\nHELP & DOCUMENTATION")
    print("=" * 30)
    print("1. Quick Start Guide")
    print("2. Command Line Usage")
    print("3. System Requirements")
    print("4. Troubleshooting")
    print("5. About")
    
    choice = input("\nSelect help topic (1-5): ").strip()
    
    if choice == '1':
        print("\nQUICK START GUIDE:")
        print("1. Run option 1 for quick EUR/USD analysis")
        print("2. Check plots/ folder for generated charts")
        print("3. View predictions in predictions/ folder")
        print("4. Use option 5 for system diagnostics if needed")
    elif choice == '2':
        command_line_interface()
    elif choice == '3':
        print("\nSYSTEM REQUIREMENTS:")
        print("- Python 3.7+")
        print("- Required packages: pandas, numpy, matplotlib, scikit-learn")
        print("- Internet connection for data fetching")
    elif choice == '4':
        print("\nTROUBLESHOOTING:")
        print("- Use option 5 → 2 for system diagnostics")
        print("- Use option 5 → 3 for component testing")
        print("- Check internet connection if data fetching fails")
    elif choice == '5':
        print("\nCURRENCY PREDICTION SYSTEM")
        print("Advanced ML-based currency prediction and analysis")
        print("Supports traditional ML and neural network models")

def main():
    fetch_fresh_data()
    
    while True:
        show_main_menu()
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == '1':
            quick_analysis()
        elif choice == '2':
            command_line_interface()
        elif choice == '3':
            interactive_menu()
        elif choice == '4':
            advanced_analytics()
        elif choice == '5':
            utilities_menu()
        elif choice == '6':
            view_predictions()
        elif choice == '7':
            open_charts()
        elif choice == '8':
            show_help()
        elif choice == '9':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
