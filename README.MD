## Installation

### Windows
```bash
install.bat
```

### Linux/Mac
```bash
./install.sh
```

## Running the System

### Option 1: Full Launcher (Recommended)
For all features and maximum functionality:
```bash
python launch.py
```

### Option 2: Simple Launcher
For basic functionality and maximum compatibility:
```bash
python simple_launch.py
```

## Troubleshooting

If you encounter any issues:

1. **Check system status:**
   ```bash
   python launch.py
   # Select option 5 → 6: System Status
   ```

2. **Run system test:**
   ```bash
   python test_system.py
   ```

3. **Verify dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

- `data/` - Raw and processed market data
- `models/` - Trained ML models (.pth, .joblib files)
- `plots/` - Generated charts and visualizations
- `reports/` - Analysis reports and logs
- `predictions/` - Price forecasts and predictions