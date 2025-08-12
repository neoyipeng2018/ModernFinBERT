#!/bin/bash
# Setup script for ModernFinBERT development environment

echo "🚀 Setting up ModernFinBERT environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run this from the ModernFinBERT directory."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "🔬 Python version: $(python --version)"

# Test core functionality
echo "🧪 Testing environment..."
python test_environment.py

echo ""
echo "🎉 Environment is ready!"
echo ""
echo "📚 To start Jupyter:"
echo "  jupyter notebook"
echo ""
echo "📊 To run Data.ipynb:"
echo "  1. Start Jupyter: jupyter notebook"
echo "  2. Open Data.ipynb"
echo "  3. Select kernel: Kernel → Change Kernel → ModernFinBERT"
echo "  4. Run all cells"
echo ""
echo "🔄 To deactivate environment:"
echo "  deactivate"