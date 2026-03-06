
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from thermal_config import ThermalParameterConfig

def check_bounds():
    print("Checking bounds for outlet_effectiveness...")
    bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
    print(f"Current bounds: {bounds}")
    
    # Calculate required ratio for the sample case
    # Outlet 45.3, Outdoor -5.3, Indoor 21.3
    # UA_rad * (45.3 - 21.3) = UA_loss * (21.3 - (-5.3))
    # UA_rad * 24.0 = UA_loss * 26.6
    # UA_rad / UA_loss = 1.108
    
    print(f"Required ratio (UA_rad/UA_loss) for sample case: 1.108")
    
    if bounds[1] < 1.11:
        print("FAIL: Upper bound is too low to support this physical scenario if UA_loss >= 1.0")
    else:
        print("PASS: Bounds are sufficient")

if __name__ == "__main__":
    check_bounds()
