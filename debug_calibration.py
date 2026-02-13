import logging
import sys
import os

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add src to path so imports work
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.physics_calibration import train_thermal_equilibrium_model
    
    print("Starting calibration debug run...")
    model = train_thermal_equilibrium_model()
    
    if model:
        print("Calibration successful!")
    else:
        print("Calibration failed.")
        
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Execution Error: {e}")
