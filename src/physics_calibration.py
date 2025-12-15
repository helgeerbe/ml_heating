"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import pandas as pd
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
    logging.warning("scipy not available - optimization will be disabled")

# Support both package-relative and direct import for notebooks/scripts
try:
    from . import config
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .state_manager import save_state
    from .influx_service import InfluxService
except ImportError:
    # Direct import fallback for standalone execution
    import config
    from thermal_equilibrium_model import ThermalEquilibriumModel
    from state_manager import save_state
    from influx_service import InfluxService


def train_thermal_equilibrium_model():
    """Train the Thermal Equilibrium Model with historical data for optimal thermal parameters using scipy optimization"""
    
    logging.info("=== THERMAL EQUILIBRIUM MODEL TRAINING (SCIPY OPTIMIZATION) ===")
    
    # Step 1: Fetch historical data
    logging.info("Step 1: Fetching historical data...")
    df = fetch_historical_data_for_calibration(lookback_hours=config.TRAINING_LOOKBACK_HOURS)
    
    if df is None or df.empty:
        logging.error("❌ Failed to fetch historical data")
        return None
    
    logging.info(f"✅ Retrieved {len(df)} samples ({len(df)/12:.1f} hours)")
    
    # Step 2: Filter for stable periods
    logging.info("Step 2: Filtering for stable thermal equilibrium periods...")
    stable_periods = filter_stable_periods(df)
    
    if len(stable_periods) < 50:
        logging.error(f"❌ Insufficient stable periods: {len(stable_periods)} (need at least 50)")
        return None
    
    logging.info(f"✅ Found {len(stable_periods)} stable periods for calibration")
    
    # Step 3: Optimize thermal parameters using scipy
    logging.info("Step 3: Optimizing thermal parameters using scipy.optimize...")
    optimized_params = optimize_thermal_parameters(stable_periods)
    
    if optimized_params is None or not optimized_params.get('optimization_success', False):
        logging.error("❌ Parameter optimization failed")
        return None
    
    logging.info(f"✅ Optimization completed - MAE: {optimized_params['mae']:.4f}°C")
    
    # Step 4: Create thermal model with optimized parameters
    logging.info("Step 4: Creating thermal model with optimized parameters...")
    thermal_model = ThermalEquilibriumModel()
    
    # Apply optimized parameters to thermal model
    thermal_model.outlet_effectiveness = optimized_params['outlet_effectiveness']
    thermal_model.heat_loss_coefficient = optimized_params['heat_loss_coefficient']
    thermal_model.thermal_time_constant = optimized_params['thermal_time_constant']
    
    # Apply heat source weights
    thermal_model.external_source_weights['pv'] = optimized_params.get('pv_heat_weight', config.PV_HEAT_WEIGHT)
    thermal_model.external_source_weights['fireplace'] = optimized_params.get('fireplace_heat_weight', config.FIREPLACE_HEAT_WEIGHT)
    thermal_model.external_source_weights['tv'] = optimized_params.get('tv_heat_weight', config.TV_HEAT_WEIGHT)
    
    # Set reasonable learning confidence based on optimization success
    thermal_model.learning_confidence = 0.8  # High confidence from scipy optimization
    
    logging.info("\n=== OPTIMIZED THERMAL PARAMETERS ===")
    logging.info(f"Thermal time constant: {thermal_model.thermal_time_constant:.2f}h")
    logging.info(f"Heat loss coefficient: {thermal_model.heat_loss_coefficient:.4f}")
    logging.info(f"Outlet effectiveness: {thermal_model.outlet_effectiveness:.3f}")
    logging.info(f"PV heat weight: {thermal_model.external_source_weights.get('pv', 0):.4f}")
    logging.info(f"Fireplace heat weight: {thermal_model.external_source_weights.get('fireplace', 0):.2f}")
    logging.info(f"TV heat weight: {thermal_model.external_source_weights.get('tv', 0):.2f}")
    logging.info(f"Optimization MAE: {optimized_params['mae']:.4f}°C")
    logging.info(f"Learning confidence: {thermal_model.learning_confidence:.3f}")
    
    # Step 5: Save thermal learning state to unified thermal state manager
    logging.info("Step 5: Saving calibrated parameters to unified thermal state...")
    try:
        from .unified_thermal_state import get_thermal_state_manager
        
        state_manager = get_thermal_state_manager()
        
        # Use optimized parameters as calibrated baseline
        calibrated_params = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'pv_heat_weight': thermal_model.external_source_weights.get('pv', config.PV_HEAT_WEIGHT),
            'fireplace_heat_weight': thermal_model.external_source_weights.get('fireplace', config.FIREPLACE_HEAT_WEIGHT),
            'tv_heat_weight': thermal_model.external_source_weights.get('tv', config.TV_HEAT_WEIGHT)
        }
        
        # Set as calibrated baseline (this updates the parameters source to "calibrated")
        state_manager.set_calibrated_baseline(calibrated_params, calibration_cycles=len(stable_periods))
        
        logging.info("✅ Calibrated parameters (scipy-optimized) saved to unified thermal state")
        logging.info("✅ Parameters will be automatically loaded on next restart")
        logging.info("🔄 Restart ml_heating service to use calibrated thermal model")
        
    except Exception as e:
        logging.error(f"❌ Failed to save calibrated parameters: {e}")
        # Fallback to old method
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'learning_confidence': thermal_model.learning_confidence,
        }
        save_state(thermal_learning_state=thermal_learning_state)
        logging.warning("⚠️ Used fallback save method - parameters may not persist")
    
    return thermal_model


def validate_thermal_model():
    """Validate thermal equilibrium model behavior across temperature ranges"""
    
    logging.info("=== THERMAL MODEL VALIDATION ===")
    
    try:
        # Import centralized thermal configuration
        from .thermal_config import ThermalParameterConfig
        
        # Initialize thermal model (will use default parameters or restored state)
        thermal_model = ThermalEquilibriumModel()
        
        logging.info("Testing thermal equilibrium physics compliance:")
        print("\nOUTLET TEMP → EQUILIBRIUM TEMP")
        print("=" * 35)
        
        # Test monotonicity - higher outlet should mean higher equilibrium
        monotonic_check = []
        outdoor_temp_test = 5.0  # Test outdoor temperature
        for outlet_temp in [25, 30, 35, 40, 45, 50, 55, 60]:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp_test,
                current_indoor=21.0,  # Test indoor temperature
                pv_power=0,
                fireplace_on=0,
                tv_on=0,
                _suppress_logging=True
            )
            monotonic_check.append(equilibrium)
            print(f"{outlet_temp:3d}°C       → {equilibrium:.2f}°C")
        
        # Check monotonicity
        is_monotonic = all(monotonic_check[i] <= monotonic_check[i+1] 
                          for i in range(len(monotonic_check)-1))
        
        print(f"\n{'✅' if is_monotonic else '❌'} Physics compliance: "
              f"{'PASSED' if is_monotonic else 'FAILED'}")
        print(f"Range: {min(monotonic_check):.2f}°C to "
              f"{max(monotonic_check):.2f}°C")
        
        # Test parameter bounds
        logging.info("\n=== THERMAL PARAMETER BOUNDS TEST ===")
        params_ok = True
        
        # Get bounds from centralized config
        thermal_bounds = ThermalParameterConfig.get_bounds('thermal_time_constant')
        if not (thermal_bounds[0] <= thermal_model.thermal_time_constant <= thermal_bounds[1]):
            logging.error(f"Thermal time constant out of bounds: "
                         f"{thermal_model.thermal_time_constant:.2f}h (bounds: {thermal_bounds})")
            params_ok = False
            
        # Get bounds from centralized config
        heat_loss_bounds = ThermalParameterConfig.get_bounds('heat_loss_coefficient')
        if not (heat_loss_bounds[0] <= thermal_model.heat_loss_coefficient <= heat_loss_bounds[1]):
            logging.error(f"Heat loss coefficient out of bounds: "
                         f"{thermal_model.heat_loss_coefficient:.4f} (bounds: {heat_loss_bounds})")
            params_ok = False
            
        if not (0.2 <= thermal_model.outlet_effectiveness <= 1.5):
            logging.error(f"Outlet effectiveness out of bounds: "
                         f"{thermal_model.outlet_effectiveness:.3f}")
            params_ok = False
            
        if params_ok:
            logging.info("✅ All thermal parameters within physical bounds")
        else:
            logging.error("❌ Some thermal parameters out of bounds")
        
        # Test adaptive learning system
        logging.info("\n=== ADAPTIVE LEARNING SYSTEM TEST ===")
        
        # Simulate some prediction feedback
        test_context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0
        }
        
        initial_confidence = thermal_model.learning_confidence
        
        # Simulate good predictions (should boost confidence)
        for _ in range(5):
            predicted = thermal_model.predict_equilibrium_temperature(
                current_indoor=21.0,  # Default indoor temp for test
                **test_context,
                _suppress_logging=True
            )
            actual = predicted + 0.1  # Small error
            thermal_model.update_prediction_feedback(
                predicted_temp=predicted,
                actual_temp=actual,
                prediction_context=test_context
            )
            
        final_confidence = thermal_model.learning_confidence
        learning_works = final_confidence != initial_confidence
        
        logging.info(f"Initial confidence: {initial_confidence:.3f}")
        logging.info(f"Final confidence: {final_confidence:.3f}")
        logging.info(f"{'✅' if learning_works else '❌'} "
                    f"Adaptive learning: {'WORKING' if learning_works else 'NOT WORKING'}")
        
        overall_success = is_monotonic and params_ok and learning_works
        logging.info(f"\n{'✅' if overall_success else '❌'} "
                    f"Overall validation: {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success
        
    except Exception as e:
        logging.error("Thermal model validation error: %s", e, exc_info=True)
        return False


def fetch_historical_data_for_calibration(lookback_hours=672):
    """
    Step 2: Fetch 672 hours of historical data for Phase 0 calibration
    
    Args:
        lookback_hours: Hours of historical data (default 672 = 28 days from .env)
    
    Returns:
        pandas.DataFrame: Historical data with required columns
    """
    logging.info(f"=== FETCHING {lookback_hours} HOURS OF HISTORICAL DATA ===")
    
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    # Fetch the data
    df = influx.get_training_data(lookback_hours=lookback_hours)
    
    if df.empty:
        logging.error("❌ No historical data available")
        return None
        
    logging.info(f"✅ Fetched {len(df)} samples ({len(df)/12:.1f} hours)")
    
    # Validate required columns exist
    required_columns = [
        config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1], 
        config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.PV_POWER_ENTITY_ID.split(".", 1)[-1],
        config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return None
    
    # Check optional columns with fallback
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    if fireplace_col not in df.columns:
        logging.info(f"⚠️  Optional fireplace column '{fireplace_col}' not found - will use 0")
        
    logging.info("✅ All required columns present")
    return df


def main():
    """Main function to run thermal equilibrium model training and validation"""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    try:
        # Train thermal model with real data
        print("🚀 Starting thermal equilibrium model training...")
        thermal_model = train_thermal_equilibrium_model()
        
        if thermal_model:
            print("✅ Thermal training completed successfully!")
            
            # Validate the trained model
            print("\n🧪 Running thermal model validation...")
            validation_passed = validate_thermal_model()
            
            if validation_passed:
                print("✅ Thermal model validation PASSED!")
            else:
                print("❌ Thermal model validation FAILED!")
                
        else:
            print("❌ Thermal training failed!")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Main execution error: {e}", exc_info=True)
        return False


def filter_stable_periods(df, temp_change_threshold=0.1, min_duration=30):
    """
    Step 3: Filter historical data for stable equilibrium periods with blocking state detection
    
    Enhanced filtering that respects blocking states and grace periods like core ml_heating system:
    - Excludes periods during DHW heating, defrosting, disinfection, boost heater
    - Applies grace periods after blocking states end (using GRACE_PERIOD_MAX_MINUTES)
    - Validates thermal stability and outlet temperature normalization
    
    Args:
        df: Historical data DataFrame
        temp_change_threshold: Max temp change per 30min (°C)
        min_duration: Minimum stable period duration (minutes)
    
    Returns:
        list: Stable periods suitable for calibration
    """
    logging.info("=== FILTERING FOR STABLE PERIODS WITH BLOCKING STATE DETECTION ===")
    
    # Column mappings
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    # Blocking state columns
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfect_col = config.DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    boost_col = config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    # Grace period configuration
    grace_period_minutes = config.GRACE_PERIOD_MAX_MINUTES
    grace_period_samples = grace_period_minutes // 5  # Convert to 5-min samples
    
    stable_periods = []
    window_size = min_duration // 5  # Convert to 5-min samples
    
    logging.info(f"Looking for periods with <{temp_change_threshold}°C change"
                f" over {min_duration} min")
    logging.info(f"Using {grace_period_minutes}min grace periods after blocking states")
    
    # Filter statistics
    filter_stats = {
        'total_checked': 0,
        'missing_data': 0,
        'temp_unstable': 0,
        'blocking_active': 0,
        'grace_period': 0,
        'fireplace_changed': 0,
        'outlet_unstable': 0,
        'passed': 0
    }
    
    for i in range(window_size + grace_period_samples, len(df) - window_size):
        filter_stats['total_checked'] += 1
        
        # Get window of data for stability analysis
        window_start = i - window_size // 2
        window_end = i + window_size // 2
        window = df.iloc[window_start:window_end]
        
        # Get extended window for blocking state analysis (includes grace period)
        grace_start = i - grace_period_samples
        grace_end = i + window_size // 2
        grace_window = df.iloc[grace_start:grace_end]
        
        # Skip if missing critical data
        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:  # Need 80% data coverage
            filter_stats['missing_data'] += 1
            continue
            
        # Calculate temperature stability
        temp_range = indoor_temps.max() - indoor_temps.min()
        temp_std = indoor_temps.std()
        
        if (temp_range > temp_change_threshold or 
            temp_std > temp_change_threshold / 2):
            filter_stats['temp_unstable'] += 1
            continue
        
        # Check for blocking states in extended window (including grace period)
        blocking_detected = False
        blocking_reasons = []
        
        # DHW heating check
        if dhw_col in grace_window.columns:
            if grace_window[dhw_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('dhw_heating')
        
        # Defrosting check  
        if defrost_col in grace_window.columns:
            if grace_window[defrost_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('defrosting')
        
        # Disinfection check
        if disinfect_col in grace_window.columns:
            if grace_window[disinfect_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('disinfection')
                
        # DHW boost heater check
        if boost_col in grace_window.columns:
            if grace_window[boost_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('boost_heater')
        
        if blocking_detected:
            # Check if blocking was during grace period vs current window
            current_window_blocking = False
            if dhw_col in window.columns and window[dhw_col].sum() > 0:
                current_window_blocking = True
            if defrost_col in window.columns and window[defrost_col].sum() > 0:
                current_window_blocking = True
            if disinfect_col in window.columns and window[disinfect_col].sum() > 0:
                current_window_blocking = True
            if boost_col in window.columns and window[boost_col].sum() > 0:
                current_window_blocking = True
                
            if current_window_blocking:
                filter_stats['blocking_active'] += 1
            else:
                filter_stats['grace_period'] += 1
            continue
        
        # Check for fireplace state changes
        fireplace_changed = False
        if fireplace_col in window.columns:
            fireplace_values = window[fireplace_col]
            if fireplace_values.nunique() > 1:
                fireplace_changed = True
                filter_stats['fireplace_changed'] += 1
                continue
        
        # Outlet temperature stability validation
        if outlet_col in window.columns:
            outlet_temps = window[outlet_col].dropna()
            if len(outlet_temps) >= window_size * 0.8:
                outlet_std = outlet_temps.std()
                # Require outlet temperature stability (< 2°C std dev)
                if outlet_std > 2.0:
                    filter_stats['outlet_unstable'] += 1
                    continue
        
        # Period passed all filters - extract data
        center_row = df.iloc[i]
        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': center_row[outlet_col], 
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0),
            'timestamp': center_row['_time'],
            'stability_score': 1.0 / (temp_std + 0.01),
            'outlet_stability': 1.0 / (outlet_temps.std() + 0.01) if outlet_col in window.columns else 1.0
        }
        stable_periods.append(period)
        filter_stats['passed'] += 1
    
    # Log filtering statistics
    logging.info("=== BLOCKING STATE FILTERING RESULTS ===")
    logging.info(f"Total periods checked: {filter_stats['total_checked']}")
    logging.info(f"Stable periods found: {filter_stats['passed']}")
    logging.info(f"Filter exclusions:")
    logging.info(f"  Missing data: {filter_stats['missing_data']}")
    logging.info(f"  Temperature unstable: {filter_stats['temp_unstable']}")
    logging.info(f"  Blocking states active: {filter_stats['blocking_active']}")
    logging.info(f"  Grace period recovery: {filter_stats['grace_period']}")
    logging.info(f"  Fireplace state changes: {filter_stats['fireplace_changed']}")
    logging.info(f"  Outlet temperature unstable: {filter_stats['outlet_unstable']}")
    
    retention_rate = (filter_stats['passed'] / filter_stats['total_checked']) * 100 if filter_stats['total_checked'] > 0 else 0
    logging.info(f"Data retention rate: {retention_rate:.1f}%")
    
    logging.info(f"✅ Found {len(stable_periods)} stable periods with blocking state filtering")
    return stable_periods


def debug_thermal_predictions(stable_periods, sample_size=5):
    """
    Debug thermal model predictions on sample data
    """
    logging.info("=== DEBUGGING THERMAL PREDICTIONS ===")
    
    # Create test model with current parameters
    test_model = ThermalEquilibriumModel()
    
    logging.info("Testing thermal model on sample periods:")
    for i, period in enumerate(stable_periods[:sample_size]):
        predicted_temp = test_model.predict_equilibrium_temperature(
            outlet_temp=period['outlet_temp'],
            outdoor_temp=period['outdoor_temp'],
            current_indoor=period.get('indoor_temp', period['outdoor_temp'] + 10.0),
            pv_power=period['pv_power'],
            fireplace_on=period['fireplace_on'],
            tv_on=period['tv_on'],
            _suppress_logging=True
        )
        
        actual_temp = period['indoor_temp']
        error = abs(predicted_temp - actual_temp)
        
        logging.info(f"Sample {i+1}:")
        logging.info(f"  Outlet: {period['outlet_temp']:.1f}°C, Outdoor: {period['outdoor_temp']:.1f}°C")
        logging.info(f"  PV: {period['pv_power']:.1f}W, Fireplace: {period['fireplace_on']:.0f}, TV: {period['tv_on']:.0f}")
        logging.info(f"  Predicted: {predicted_temp:.1f}°C, Actual: {actual_temp:.1f}°C")
        logging.info(f"  Error: {error:.1f}°C")
        logging.info("")


def optimize_thermal_parameters(stable_periods):
    """
    Step 4: Multi-parameter optimization using scipy.optimize with data availability checks
    
    Args:
        stable_periods: List of stable equilibrium periods
    
    Returns:
        dict: Optimized thermal parameters
    """
    logging.info("=== MULTI-PARAMETER OPTIMIZATION WITH DATA AVAILABILITY CHECKS ===")
    
    if minimize is None:
        logging.error("❌ scipy not available - cannot optimize parameters")
        return None
    
    # Debug thermal predictions before optimization
    debug_thermal_predictions(stable_periods)
    
    # Check data availability for each heat source
    logging.info("=== CHECKING DATA AVAILABILITY ===")
    
    # Analyze actual data availability in stable periods
    total_periods = len(stable_periods)
    data_stats = {
        'pv_power': sum(1 for p in stable_periods if p.get('pv_power', 0) > 0),
        'fireplace_on': sum(1 for p in stable_periods if p.get('fireplace_on', 0) > 0), 
        'tv_on': sum(1 for p in stable_periods if p.get('tv_on', 0) > 0)
    }
    
    # Calculate usage percentages
    data_availability = {}
    for source, count in data_stats.items():
        percentage = (count / total_periods) * 100
        data_availability[source] = percentage
        logging.info(f"  {source}: {count}/{total_periods} periods ({percentage:.1f}%)")
    
    # Determine which parameters to exclude from optimization
    excluded_params = []
    min_usage_threshold = 1.0  # Require >1% usage to optimize
    
    if data_availability['fireplace_on'] <= min_usage_threshold:
        excluded_params.append('fireplace_heat_weight')
        logging.info(f"  🚫 Excluding fireplace_heat_weight (only {data_availability['fireplace_on']:.1f}% usage)")
    
    if data_availability['tv_on'] <= min_usage_threshold:
        excluded_params.append('tv_heat_weight') 
        logging.info(f"  🚫 Excluding tv_heat_weight (only {data_availability['tv_on']:.1f}% usage)")
        
    if data_availability['pv_power'] <= min_usage_threshold:
        excluded_params.append('pv_heat_weight')
        logging.info(f"  🚫 Excluding pv_heat_weight (only {data_availability['pv_power']:.1f}% usage)")
        
    # Current parameter values from config
    current_params = {
        'outlet_effectiveness': config.OUTLET_EFFECTIVENESS,
        'heat_loss_coefficient': config.HEAT_LOSS_COEFFICIENT, 
        'thermal_time_constant': config.THERMAL_TIME_CONSTANT,
        'pv_heat_weight': config.PV_HEAT_WEIGHT,
        'fireplace_heat_weight': config.FIREPLACE_HEAT_WEIGHT,
        'tv_heat_weight': config.TV_HEAT_WEIGHT
    }
    
    logging.info("=== PARAMETERS FOR OPTIMIZATION ===")
    for param, value in current_params.items():
        if param in excluded_params:
            logging.info(f"  {param}: {value} (FIXED - insufficient data)")
        else:
            logging.info(f"  {param}: {value} (OPTIMIZE)")
    
    # Build optimization parameter list and bounds (excluding parameters without data)
    param_names = []
    param_values = []
    param_bounds = []
    
    # Always optimize core thermal parameters (exclude thermal_time_constant due to parameter coupling)
    param_names.extend(['outlet_effectiveness', 'heat_loss_coefficient'])
    param_values.extend([
        current_params['outlet_effectiveness'],
        current_params['heat_loss_coefficient']
    ])
    # Import centralized thermal configuration
    from .thermal_config import ThermalParameterConfig
    
    param_bounds.extend([
        ThermalParameterConfig.get_bounds('outlet_effectiveness'),
        ThermalParameterConfig.get_bounds('heat_loss_coefficient')
    ])
    
    # Conditionally add heat source parameters based on data availability
    if 'pv_heat_weight' not in excluded_params:
        param_names.append('pv_heat_weight')
        param_values.append(current_params['pv_heat_weight'])
        param_bounds.append((0.0005, 0.005))
        
    if 'fireplace_heat_weight' not in excluded_params:
        param_names.append('fireplace_heat_weight') 
        param_values.append(current_params['fireplace_heat_weight'])
        param_bounds.append((1.0, 6.0))
        
    if 'tv_heat_weight' not in excluded_params:
        param_names.append('tv_heat_weight')
        param_values.append(current_params['tv_heat_weight'])
        param_bounds.append((0.1, 1.5))
    
    logging.info(f"Optimizing {len(param_names)} parameters: {param_names}")
    
    def objective_function(params):
        """Calculate MAE for given parameters with dynamic parameter mapping and debugging"""
        total_error = 0.0
        valid_predictions = 0
        
        # Map parameters to their names dynamically
        param_dict = dict(zip(param_names, params))
        
        # Debug: Log parameter values being tested
        debug_str = ", ".join([f"{name}={val:.4f}" for name, val in param_dict.items()])
        # Only log every 10th call to avoid spam
        if hasattr(objective_function, 'call_count'):
            objective_function.call_count += 1
        else:
            objective_function.call_count = 1
            
        if objective_function.call_count % 10 == 1:
            logging.debug(f"Testing params: {debug_str}")
        
        # Test on subset for speed during optimization (use all data for final evaluation)
        test_periods = stable_periods[::5]  # Use every 5th period for speed
        
        for period in test_periods:
            try:
                # Create temporary thermal model with test parameters
                test_model = ThermalEquilibriumModel()
                test_model.outlet_effectiveness = param_dict['outlet_effectiveness']
                test_model.heat_loss_coefficient = param_dict['heat_loss_coefficient']
                test_model.thermal_time_constant = current_params['thermal_time_constant']  # Use current value since not optimized
                
                # Set heat source weights (use original values for excluded params)
                test_model.external_source_weights['pv'] = param_dict.get(
                    'pv_heat_weight', current_params['pv_heat_weight'])
                test_model.external_source_weights['fireplace'] = param_dict.get(
                    'fireplace_heat_weight', current_params['fireplace_heat_weight'])
                test_model.external_source_weights['tv'] = param_dict.get(
                    'tv_heat_weight', current_params['tv_heat_weight'])
                
                # Predict equilibrium temperature
                predicted_temp = test_model.predict_equilibrium_temperature(
                    outlet_temp=period['outlet_temp'],
                    outdoor_temp=period['outdoor_temp'],
                    current_indoor=period.get('indoor_temp', period['outdoor_temp'] + 10.0),
                    pv_power=period['pv_power'],
                    fireplace_on=period['fireplace_on'],
                    tv_on=period['tv_on'],
                    _suppress_logging=True
                )
                
                # Calculate error
                error = abs(predicted_temp - period['indoor_temp'])
                
                # Skip unrealistic errors that indicate bad parameters
                if error > 50.0:  # Skip extreme errors
                    continue
                    
                total_error += error
                valid_predictions += 1
                
            except Exception:
                # Skip problematic periods
                continue
                
        if valid_predictions == 0:
            return 1000.0  # High error for invalid parameter sets
            
        mae = total_error / valid_predictions
        
        # Debug: Log MAE for this parameter set
        if objective_function.call_count % 10 == 1:
            logging.debug(f"MAE for {debug_str}: {mae:.4f}")
            
        return mae
    
    logging.info(f"Starting optimization with {len(stable_periods)} periods...")
    logging.info("This may take a few minutes...")
    
    # Enable DEBUG logging to see parameter exploration
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Run optimization with improved settings for better exploration
    try:
        result = minimize(
            objective_function,
            x0=param_values,
            bounds=param_bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 500,      # More iterations for better exploration
                'ftol': 1e-3,        # More appropriate tolerance for MAE ~1.5°C
                'disp': True,        # Show optimization progress
                'iprint': 2          # Print status every 2 iterations
            }
        )
        
        # Log detailed optimization results regardless of success
        logging.info(f"🔍 SCIPY OPTIMIZATION RESULTS:")
        logging.info(f"  Success: {result.success}")
        logging.info(f"  Message: {result.message}")
        logging.info(f"  Function evaluations: {result.nfev}")
        logging.info(f"  Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
        logging.info(f"  Final function value: {result.fun:.6f}")
        
        # Log initial vs final parameter values
        logging.info(f"  Parameter changes:")
        for i, param_name in enumerate(param_names):
            initial_val = param_values[i]
            final_val = result.x[i]
            change = final_val - initial_val
            logging.info(f"    {param_name}: {initial_val:.6f} → {final_val:.6f} (Δ{change:+.6f})")
        
        if result.success:
            # Build optimized parameters dict with both optimized and excluded params
            optimized_params = dict(current_params)  # Start with all original
            
            # Update with optimized values
            for i, param_name in enumerate(param_names):
                optimized_params[param_name] = result.x[i]
            
            optimized_params['mae'] = result.fun
            optimized_params['optimization_success'] = True
            optimized_params['excluded_parameters'] = excluded_params
            
            logging.info("✅ Optimization completed successfully!")
            logging.info("Optimized parameters:")
            for param, value in optimized_params.items():
                if param not in ['mae', 'optimization_success', 'excluded_parameters']:
                    old_value = current_params[param]
                    if param in excluded_params:
                        logging.info(f"  {param}: {value:.4f} (FIXED - no data)")
                    else:
                        change_pct = ((value - old_value) / old_value) * 100
                        logging.info(f"  {param}: {value:.4f} "
                                   f"(was {old_value:.4f}, {change_pct:+.1f}%)")
            
            logging.info(f"Final MAE: {result.fun:.4f}°C")
            return optimized_params
            
        else:
            logging.error(f"❌ Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Optimization error: {e}")
        return None


def backup_existing_calibration():
    """
    Create backup of existing thermal calibration before new calibration
    
    Returns:
        str: Path to backup file, or None if no existing calibration
    """
    logging.info("Creating backup of existing thermal calibration...")
    
    try:
        from .unified_thermal_state import get_thermal_state_manager
        import json
        import os
        from datetime import datetime
        
        state_manager = get_thermal_state_manager()
        
        # Get current thermal state from the state manager
        # Use the thermal_state property or the file directly
        if hasattr(state_manager, 'thermal_state'):
            current_state = state_manager.thermal_state
        else:
            # Fallback: read the thermal state file directly
            state_file_path = "/opt/ml_heating/thermal_state.json"
            if os.path.exists(state_file_path):
                with open(state_file_path, 'r') as f:
                    current_state = json.load(f)
            else:
                logging.info("No existing thermal state file found - skipping backup")
                return None
        
        # Create backup filename with timestamp (use pre_calibration prefix to match expected format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"pre_calibration_{timestamp}.json"
        backup_path = os.path.join("/opt/ml_heating", backup_filename)
        
        # Save current state to backup file
        with open(backup_path, 'w') as f:
            json.dump(current_state, f, indent=2, default=str)
            
        logging.info(f"✅ Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        logging.warning(f"Failed to create calibration backup: {e}")
        return None


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
