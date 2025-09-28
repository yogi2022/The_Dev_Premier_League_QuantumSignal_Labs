#!/bin/bash

# AI Financial Signal Extraction System - Linux/Mac Launch Script
# This script activates the virtual environment and launches the Streamlit dashboard

echo "🚀 Starting AI Financial Signal Extraction & Trading Strategy Validation System"
echo "=================================================================="

# Check if virtual environment exists
if [ ! -d "financial_signals_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run 'python setup.py' first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source financial_signals_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "📝 Some features may not work without proper configuration"
fi

# Check if main dashboard file exists
if [ ! -f "main_dashboard.py" ]; then
    echo "❌ main_dashboard.py not found in current directory"
    echo "Please ensure you're running this script from the project root"
    exit 1
fi

echo "🌐 Launching Streamlit dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit with optimal settings
streamlit run main_dashboard.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=false \
    --server.runOnSave=true \
    --browser.gatherUsageStats=false

echo ""
echo "👋 Dashboard stopped. Thank you for using AI Financial Signal Extraction System!"