#!/usr/bin/env python3
"""
Test Phase 0 House-Specific Calibration

This script tests the Phase 0 calibration process using TRAINING_LOOKBACK_HOURS
from the .env configuration file.
"""

import sys
import os
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from physics_calibration import run_phase0_calibration
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the ml_heating directory")
    sys.exit(1)

def main():
    """Test Phase 0 calibration"""
    print("🧪 TESTING PHASE 0 HOUSE-SPECIFIC CALIBRATION")
    print("=" * 60)
    
    # Set up logging to see all details
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run Phase 0 calibration
        success = run_phase0_calibration()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 PHASE 0 CALIBRATION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nFiles created:")
            print("  • calibrated_baseline.json - House-specific parameters")
            print("\nNext steps:")
            print("  1. Review the calibrated_baseline.json file")
            print("  2. Update your .env with optimized parameters (optional)")
            print("  3. Restart ML heating system to use new baseline")
            print("  4. Run Phase 1 adaptive learning for real-time corrections")
            return True
        else:
            print("\n" + "=" * 60) 
            print("❌ PHASE 0 CALIBRATION TEST FAILED")
            print("=" * 60)
            print("\nCommon issues:")
            print("  • Insufficient historical data (need 28+ days)")
            print("  • InfluxDB connection problems")
            print("  • Missing required sensor data")
            print("  • scipy optimization library not installed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.exception("Full error details:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
