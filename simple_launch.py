#!/usr/bin/env python3

import sys
import os
from pathlib import Path

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_main_menu():
    print("\nCURRENCY PREDICTION SYSTEM")
    print("=" * 40)
    print("Simplified launcher for maximum compatibility")
    print("=" * 40)
    
    print("\nAVAILABLE OPTIONS:")
    print("1. Quick Analysis (EUR/USD)")
    print("2. System Test")
    print("3. Check System Status")
    print("4. Exit")

def quick_analysis():
    clear_screen()
    print("\nQUICK ANALYSIS - EUR/USD")
    print("=" * 30)
    
    try:
        from launchers.quick_analysis import main as quick_main
        quick_main()
        print("\nAnalysis completed!")
    except Exception as e:
        print(f"Error: {e}")
        print("Try running: python -m pip install -r requirements.txt")

def system_test():
    clear_screen()
    print("\nSYSTEM TEST")
    print("=" * 20)
    
    try:
        import test_system
        test_system.main()
    except Exception as e:
        print(f"Test system error: {e}")
        
        print("\nBasic import test:")
        
        imports_to_test = [
            ("pandas", "pd"),
            ("numpy", "np"),
            ("matplotlib.pyplot", "plt"),
            ("sklearn", "sklearn"),
            ("utils.data_fetcher", "CurrencyDataFetcher")
        ]
        
        for module_name, alias in imports_to_test:
            try:
                __import__(module_name)
                print(f"✓ {module_name}")
            except ImportError:
                print(f"✗ {module_name} - Not installed")

def check_status():
    clear_screen()
    print("\nSYSTEM STATUS")
    print("=" * 20)
    
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    directories = ['src', 'data', 'models', 'plots', 'predictions', 'reports']
    for directory in directories:
        if os.path.exists(directory):
            files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
            print(f"✓ {directory}/ ({files} files)")
        else:
            print(f"✗ {directory}/ - Missing")

def main():
    clear_screen()
    print("CURRENCY PREDICTION SYSTEM - SIMPLE LAUNCHER")
    print("=" * 50)
    print("This launcher uses only core functionality")
    print("=" * 50)
    
    while True:
        show_main_menu()
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            quick_analysis()
        elif choice == '2':
            system_test()
        elif choice == '3':
            check_status()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")
        
        input("\nPress Enter to continue...")
        clear_screen()

if __name__ == "__main__":
    main() 