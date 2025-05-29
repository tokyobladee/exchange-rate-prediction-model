#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================"
echo "CURRENCY PREDICTION SYSTEM INSTALLATION"
echo -e "================================================================${NC}"
echo

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo -e "${BLUE}[1/7] Checking Python...${NC}"
if check_command python3; then
    PYTHON_CMD="python3"
elif check_command python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python not found!${NC}"
    echo "Install Python 3.8+ from your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${YELLOW}WARNING: Python 3.8+ recommended${NC}"
fi

echo -e "${BLUE}[2/7] Checking pip...${NC}"
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo -e "${RED}ERROR: pip not found!${NC}"
    echo "Install pip:"
    echo "  Ubuntu/Debian: sudo apt install python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3-pip"
    echo "  macOS: $PYTHON_CMD -m ensurepip --upgrade"
    exit 1
fi

echo -e "${BLUE}[3/7] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

echo -e "${BLUE}[4/7] Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${BLUE}[5/7] Upgrading pip...${NC}"
python -m pip install --upgrade pip

echo -e "${BLUE}[6/7] Installing dependencies...${NC}"
echo "This may take several minutes..."

if python -m pip install -r requirements.txt; then
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
else
    echo -e "${RED}ERROR: Failed to install dependencies!${NC}"
    echo "Try installing core packages manually:"
    echo "  pip install pandas numpy matplotlib seaborn scikit-learn"
    exit 1
fi

echo -e "${BLUE}[7/7] Verifying installation...${NC}"
if python -c "import pandas, numpy, sklearn, matplotlib, seaborn, yfinance, plotly, tensorflow, torch; print('Core libraries installed successfully')" 2>/dev/null; then
    echo -e "${GREEN}All libraries installed successfully!${NC}"
else
    echo -e "${YELLOW}WARNING: Some libraries may not work correctly${NC}"
    echo "Run system test for detailed diagnostics"
fi

echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p data models plots reports predictions

echo
echo -e "${GREEN}================================================================"
echo "INSTALLATION COMPLETED!"
echo -e "================================================================${NC}"
echo
echo -e "${BLUE}To run the system:${NC}"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Launch system: python launch.py"
echo
echo -e "${BLUE}Quick start (with activated venv):${NC}"
echo "  python launch.py"
echo
echo -e "${BLUE}System test:${NC}"
echo "  python test_system.py"
echo
echo -e "${BLUE}Deactivate virtual environment:${NC}"
echo "  deactivate"
echo
echo -e "${GREEN}================================================================${NC}" 