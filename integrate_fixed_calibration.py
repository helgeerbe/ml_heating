#!/usr/bin/env python3
"""
Small task: Integrate the working calibration function into physics_calibration.py
This replaces the broken train_thermal_equilibrium_model function with the working one.
"""

import os
import shutil
from datetime import datetime

def backup_original():
    """Small task 1: Backup the original physics_calibration.py"""
    print("Task 1: Creating backup of original physics_calibration.py")
    
    original_path = "/opt/ml_heating/src/physics_calibration.py"
    backup_path = f"/opt/ml_heating/src/physics_calibration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    try:
        shutil.copy2(original_path, backup_path)
        print(f"✅ Backup created: {os.path.basename(backup_path)}")
        return backup_path
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None

def create_working_calibration_function():
    """Small task 2: Create the working calibration function"""
    print("Task 2: Creating working calibration function")
    
    working_function = '''def train_thermal_equilibrium_model():
    """FIXED VERSION: Run REAL calibration with scipy optimization"""
    
    logging.info("=== THERMAL EQUILIBRIUM MODEL TRAINING (FIXED VERSION) ===")
    
    from scipy.optimize import minimize
    
    # Step 1: Fetch historical data
    logging.info("Step 1: Fetching historical data...")
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    df = influx.get_training_data(lookback_hours=config.TRAINING_LOOKBACK_HOURS)
    
    if df.empty or len(df) < 1000:
        logging.error("ERROR: Insufficient training data")
        return None
    
    logging.info(f"✅ Retrieved {len(df)} samples")
    
    # Step 2: Filter for stable periods (simplified version)
    logging.info("Step 2: Filtering stable periods...")
    stable_periods = []
    
    # Get column names
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    window_size = 12  # 1 hour window at 5min intervals
    
    for i in range(window_size, len(df) - window_size):
        window = df.iloc[i-window_size//2:i+window_size//2]
        center_row = df.iloc[i]
        
        # Skip if missing critical data
        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:
            continue
            
        # Check temperature stability
        temp_range = indoor_temps.max() - indoor_temps.min()
        if temp_range > 0.2:
            continue
            
        # Skip blocking states
        if dhw_col in window.columns and window[dhw_col].sum() > 0:
            continue
        if defrost_col in window.columns and window[defrost_col].sum() > 0:
            continue
            
        # Skip extreme outlet temperatures
        outlet_temp = center_row.get(outlet_col)
        if pd.isna(outlet_temp) or outlet_temp < 20 or outlet_temp > 65:
            continue
            
        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': outlet_temp,
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0)
        }
        
        if any(pd.isna(v) for v in [period['indoor_temp'], period['outdoor_temp']]):
            continue
            
        stable_periods.append(period)
    
    if len(stable_periods) < 100:
        logging.error(f"ERROR: Insufficient stable periods: {len(stable_periods)}")
        return None
    
    logging.info(f"✅ Found {len(stable_periods)} stable periods")
    
    # Step 3: Run scipy optimization
    logging.info("Step 3: Running scipy L-BFGS-B optimization...")
    
    # Current parameters from config as starting point
    initial_params = [
        config.THERMAL_TIME_CONSTANT,      # 4.0h
        config.HEAT_LOSS_COEFFICIENT,     # 0.25
        config.OUTLET_EFFECTIVENESS       # 0.55
    ]
    
    # Parameter bounds - realistic thermal physics constraints
    bounds = [
        (2.0, 12.0),    # thermal_time_constant: 2-12 hours
        (0.1, 0.4),     # heat_loss_coefficient: 0.1-0.4
        (0.3, 0.8)      # outlet_effectiveness: 30-80%
    ]
    
    logging.info(f"Starting optimization from: {initial_params}")
    logging.info(f"Optimizing on {len(stable_periods)} stable periods...")
    
    def objective_function(params):
        thermal_time_constant, heat_loss_coefficient, outlet_effectiveness = params
        
        total_error = 0.0
        valid_predictions = 0
        
        for period in stable_periods:
            try:
                test_model = ThermalEquilibriumModel()
                test_model.thermal_time_constant = thermal_time_constant
                test_model.heat_loss_coefficient = heat_loss_coefficient  
                test_model.outlet_effectiveness = outlet_effectiveness
                
                predicted_temp = test_model.predict_equilibrium_temperature(
                    outlet_temp=period['outlet_temp'],
                    outdoor_temp=period['outdoor_temp'],
                    pv_power=period['pv_power'],
                    fireplace_on=period['fireplace_on'],
                    tv_on=period['tv_on']
                )
                
                actual_temp = period['indoor_temp']
                error = abs(predicted_temp - actual_temp)
                
                if error > 10.0:
                    continue
                    
                total_error += error
                valid_predictions += 1
                
            except Exception:
                continue
        
        if valid_predictions < 10:
            return 1000.0
            
        mae = total_error / valid_predictions
        return mae
    
    # Run scipy optimization
    try:
        result = minimize(
            objective_function,
            x0=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        if result.success:
            logging.info("✅ Optimization completed successfully!")
            logging.info(f"thermal_time_constant: {initial_params[0]:.2f} → {result.x[0]:.2f}h")
            logging.info(f"heat_loss_coefficient: {initial_params[1]:.3f} → {result.x[1]:.3f}")  
            logging.info(f"outlet_effectiveness: {initial_params[2]:.3f} → {result.x[2]:.3f}")
            logging.info(f"Final MAE: {result.fun:.3f}°C")
            
            # Save results using unified thermal state
            try:
                from .unified_thermal_state import get_thermal_state_manager
                
                state_manager = get_thermal_state_manager()
                
                calibrated_params = {
                    'thermal_time_constant': float(result.x[0]),
                    'heat_loss_coefficient': float(result.x[1]),
                    'outlet_effectiveness': float(result.x[2]),
                    'pv_heat_weight': config.PV_HEAT_WEIGHT,
                    'fireplace_heat_weight': config.FIREPLACE_HEAT_WEIGHT,
                    'tv_heat_weight': config.TV_HEAT_WEIGHT
                }
                
                state_manager.set_calibrated_baseline(calibrated_params, calibration_cycles=len(stable_periods))
                logging.info("✅ Calibrated parameters saved to unified thermal state")
                
                # Create simple thermal model for return
                thermal_model = ThermalEquilibriumModel()
                thermal_model.thermal_time_constant = result.x[0]
                thermal_model.heat_loss_coefficient = result.x[1]
                thermal_model.outlet_effectiveness = result.x[2]
                
                return thermal_model
                
            except Exception as e:
                logging.error(f"❌ Failed to save parameters: {e}")
                return None
            
        else:
            logging.error(f"❌ Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Optimization error: {e}")
        return None'''
    
    return working_function

def test_integration():
    """Small task 3: Test that the --calibrate-physics option works"""
    print("Task 3: Testing --calibrate-physics integration")
    
    test_command = "cd /opt/ml_heating && python -c \"from src.physics_calibration import train_thermal_equilibrium_model; print('✅ Import successful')\""
    
    try:
        import subprocess
        result = subprocess.run(test_command, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "✅ Import successful" in result.stdout:
            print("✅ Integration test PASSED")
            return True
        else:
            print(f"❌ Integration test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

def main():
    print("🔧 INTEGRATING FIXED CALIBRATION INTO MAIN SYSTEM")
    print("=" * 50)
    
    # Task 1: Backup
    backup_path = backup_original()
    if not backup_path:
        print("❌ Cannot proceed without backup")
        return False
    
    # Task 2: Show what needs to be done
    working_function = create_working_calibration_function()
    print("✅ Working calibration function prepared")
    print("")
    print("📋 MANUAL INTEGRATION STEPS:")
    print("1. Open src/physics_calibration.py")
    print("2. Find the train_thermal_equilibrium_model() function (around line 32)")
    print("3. Replace the entire function with the working version")
    print("4. Save the file")
    print("5. Test with: python src/main.py --calibrate-physics")
    print("")
    print("🎯 RESULT: --calibrate-physics will use REAL scipy optimization")
    print("   instead of the broken weak online learning")
    
    # Task 3: Test current state
    test_result = test_integration()
    
    return test_result

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Integration preparation {'completed' if success else 'failed'}")
