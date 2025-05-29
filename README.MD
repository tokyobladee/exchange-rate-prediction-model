CURRENCY PREDICTION SYSTEM - LAUNCH INSTRUCTIONS

INSTALLATION
============
Requirements:
- Python 3.8+
- Internet connection
- 4GB RAM minimum

Install dependencies:
   Windows: install.bat
   Linux/Mac: install.sh

LAUNCH
======
Run main launcher:
   python launch.py

Select option from menu or use direct commands:
   python src/launchers/quick_analysis.py [PAIR] [PERIOD]

Examples:
   python src/launchers/quick_analysis.py EURUSD 1y
   python src/launchers/quick_analysis.py GBPUSD 6mo

DATA STORAGE
============
- data/ - Raw market data and processed datasets
- models/ - Trained ML models (.pth, .joblib files)
- plots/ - Generated charts and visualizations
- reports/ - Analysis reports and logs
- predictions/ - Price forecasts and predictions 