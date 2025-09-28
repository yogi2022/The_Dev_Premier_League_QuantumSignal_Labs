"""
Setup script for creating virtual environment and installing dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(e.stderr)
        return False

def main():
    """Main setup function"""
    print("🏗️  Setting up AI Financial Signal Extraction System")
    print("=" * 60)
    
    # Check if Python 3.8+ is available
    try:
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            sys.exit(1)
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        sys.exit(1)
    
    # Create virtual environment
    venv_name = "financial_signals_env"
    if not run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
        print("❌ Failed to create virtual environment")
        return False
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = f"{venv_name}\\Scripts\\activate"
        pip_path = f"{venv_name}\\Scripts\\pip"
        python_path = f"{venv_name}\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        activate_script = f"source {venv_name}/bin/activate"
        pip_path = f"{venv_name}/bin/pip"
        python_path = f"{venv_name}/bin/python"
    
    # Upgrade pip
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: Failed to upgrade pip, continuing anyway...")
    
    # Install requirements
    if not run_command(f"{pip_path} install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        return False
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("\n⚠️  .env file not found!")
        print("📝 Creating sample .env file...")
        
        sample_env = """# Snowflake Configuration
SNOWFLAKE_USER=PRASAD123
SNOWFLAKE_PASSWORD=Prasad@123####
SNOWFLAKE_ACCOUNT=app.snowflake.com
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=FINANCIAL_SIGNALS_DB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=ACCOUNTADMIN

# API Keys
ALPHA_VANTAGE_API_KEY=FCUH0PN1D771S5KL
NEWS_API_KEY=4085c792763b4c4ab6b10c5357d3e9f7

# Application Settings
DEBUG_MODE=True
REFRESH_INTERVAL=300
MAX_SIGNALS_PER_STOCK=10
BACKTESTING_PERIOD_DAYS=90
"""
        
        with open(".env", "w") as f:
            f.write(sample_env)
        
        print("✅ Sample .env file created with your provided credentials!")
    else:
        print("✅ .env file found")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "streamlit",
        "pandas",
        "numpy", 
        "plotly",
        "snowflake.snowpark",
        "alpha_vantage",
        "newsapi",
        "scikit-learn"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {failed_imports}")
        print("You may need to install additional dependencies manually.")
    
    # Create launch script
    if os.name == 'nt':  # Windows
        launch_script = f"""@echo off
echo Starting AI Financial Signal Extraction System...
{activate_script}
streamlit run main_dashboard.py --server.port=8501 --server.address=localhost
pause
"""
        script_name = "run_dashboard.bat"
    else:  # Unix/Linux/MacOS
        launch_script = f"""#!/bin/bash
echo "Starting AI Financial Signal Extraction System..."
{activate_script}
streamlit run main_dashboard.py --server.port=8501 --server.address=localhost
"""
        script_name = "run_dashboard.sh"
    
    with open(script_name, "w") as f:
        f.write(launch_script)
    
    if os.name != 'nt':
        os.chmod(script_name, 0o755)
    
    print(f"\n✅ Launch script created: {script_name}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate virtual environment: {activate_script}")
    print("2. Verify your .env file contains correct credentials")
    print("3. Run the dashboard:")
    print(f"   - Using launch script: ./{script_name}")
    print(f"   - Or manually: {python_path} -m streamlit run main_dashboard.py")
    print("\n🌐 The dashboard will be available at: http://localhost:8501")
    
    print("\n🔧 System Components:")
    print("• Data Ingestion: Alpha Vantage API + News API")
    print("• AI Processing: Snowflake Cortex (if configured)")
    print("• Backtesting: Advanced risk-adjusted framework")
    print("• Visualization: Interactive Streamlit dashboard")
    
    print("\n⚡ Features:")
    print("• Real-time market data ingestion")
    print("• AI-powered sentiment analysis")
    print("• Multi-factor signal generation")
    print("• Comprehensive backtesting")
    print("• Risk management & performance analytics")
    print("• Explainable AI for signal attribution")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\n🚀 Ready to launch your AI Financial Signal Extraction System!")