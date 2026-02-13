"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import json
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
    from .thermal_config import ThermalParameterConfig
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    # Direct import fallback for standalone execution
    import config
    from thermal_equilibrium_model import ThermalEquilibriumModel
    from state_manager import save_state
    from influx_service import InfluxService
    from thermal_config import ThermalParameterConfig
    from unified_thermal_state import get_thermal_state_manager


def train_thermal_equilibrium_model():
    """Train the Thermal Equilibrium Model with historical data for optimal
    thermal parameters using scipy optimization"""

    logging.info(
        "=== THERMAL EQUILIBRIUM MODEL TRAINING (SCIPY OPTIMIZATION) ==="
    )

    # Step 0: Backup existing calibration
    backup_existing_calibration()

    # Step 1: Fetch historical data
    logging.info("Step 1: Fetching historical data...")
    df = fetch_historical_data_for_calibration(
        lookback_hours=config.TRAINING_LOOKBACK_HOURS
    )

    if df is None or df.empty:
        logging.error("❌ Failed to fetch historical data")
        return None

    logging.info(f"✅ Retrieved {len(df)} samples ({len(df)/12:.1f} hours)")

    # Step 2: Filter for stable periods
    logging.info("Step 2: Filtering for stable thermal equilibrium periods...")
    stable_periods = filter_stable_periods(df)

    if len(stable_periods) < 50:
        logging.error(
            f"❌ Insufficient stable periods: {len(stable_periods)} "
            "(need at least 50)"
        )
        return None

    logging.info(
        f"✅ Found {len(stable_periods)} stable periods for calibration"
    )

    # Step 3: Optimize thermal parameters using scipy
    logging.info(
        "Step 3: Optimizing thermal parameters using scipy.optimize..."
    )
    optimized_params = optimize_thermal_parameters(stable_periods)

    if not optimized_params or not optimized_params.get(
        'optimization_success'
    ):
        logging.error("❌ Parameter optimization failed")
        return None

    logging.info(
        f"✅ Optimization completed - MAE: {optimized_params['mae']:.4f}°C"
    )

    # Step 3b: Optimize dynamic parameters (transient)
    logging.info("Step 3b: Optimizing dynamic parameters (thermal_time_constant)...")

    # Create a temporary model with optimized static params for transient calibration
    temp_model = ThermalEquilibriumModel()
    temp_model.heat_loss_coefficient = optimized_params['heat_loss_coefficient']
    temp_model.outlet_effectiveness = optimized_params['outlet_effectiveness']

    # Apply weights
    temp_model.external_source_weights['pv'] = optimized_params.get(
        'pv_heat_weight', ThermalParameterConfig.get_default('pv_heat_weight')
    )
    temp_model.external_source_weights['fireplace'] = optimized_params.get(
        'fireplace_heat_weight',
        ThermalParameterConfig.get_default('fireplace_heat_weight')
    )
    temp_model.external_source_weights['tv'] = optimized_params.get(
        'tv_heat_weight', ThermalParameterConfig.get_default('tv_heat_weight')
    )

    transient_samples = filter_transient_periods(df)
    if transient_samples:
        best_tau = calibrate_transient_parameters(temp_model, transient_samples)
        if best_tau:
            optimized_params['thermal_time_constant'] = best_tau
            logging.info(f"✅ Optimized thermal_time_constant: {best_tau:.2f}h")
    else:
        logging.warning("⚠️ No transient periods found for dynamic calibration")

    # Step 3c: Validate with cooling time constant (DHW periods)
    logging.info("Step 3c: Validating with cooling time constant...")
    cooling_tau, cooling_r2 = calculate_cooling_time_constant(df)

    if cooling_tau:
        heating_tau = optimized_params['thermal_time_constant']
        diff_pct = abs(heating_tau - cooling_tau) / heating_tau * 100

        logging.info(
            f"Cooling time constant: {cooling_tau:.2f}h (R²={cooling_r2:.2f})"
        )
        logging.info(f"Heating time constant: {heating_tau:.2f}h")
        logging.info(f"Difference: {diff_pct:.1f}%")

        if diff_pct > 30.0:
            logging.warning(
                "⚠️ Significant difference between heating and cooling "
                "time constants (>30%)"
            )
            logging.warning(
                "This may indicate unmodeled heat gains or sensor issues."
            )

        # If heating calibration failed (default value) or has low confidence,
        # consider using cooling tau or a weighted average.
        default_tau = ThermalParameterConfig.get_default('thermal_time_constant')
        if abs(heating_tau - default_tau) < 0.1 and cooling_r2 > 0.9:
            logging.info(
                "Heating tau is default. Using high-confidence cooling tau."
            )
            optimized_params['thermal_time_constant'] = cooling_tau

    # Step 4: Create thermal model with optimized parameters
    logging.info("Step 4: Creating thermal model with optimized parameters...")
    thermal_model = ThermalEquilibriumModel()

    # Apply optimized parameters to thermal model
    thermal_model.thermal_time_constant = optimized_params[
        'thermal_time_constant'
    ]
    thermal_model.heat_loss_coefficient = optimized_params[
        'heat_loss_coefficient'
    ]
    thermal_model.outlet_effectiveness = optimized_params[
        'outlet_effectiveness'
    ]

    # Apply heat source weights
    pv_weight = optimized_params.get(
        'pv_heat_weight',
        ThermalParameterConfig.get_default('pv_heat_weight')
    )
    fireplace_weight = optimized_params.get(
        'fireplace_heat_weight',
        ThermalParameterConfig.get_default('fireplace_heat_weight')
    )
    tv_weight = optimized_params.get(
        'tv_heat_weight',
        ThermalParameterConfig.get_default('tv_heat_weight')
    )

    thermal_model.external_source_weights['pv'] = pv_weight
    thermal_model.external_source_weights['fireplace'] = fireplace_weight
    thermal_model.external_source_weights['tv'] = tv_weight

    # Set reasonable learning confidence based on optimization success
    thermal_model.learning_confidence = 3.0  # High confidence from scipy

    logging.info("\n=== OPTIMIZED THERMAL PARAMETERS ===")
    logging.info(
        f"Thermal time constant: {thermal_model.thermal_time_constant:.2f}h"
    )
    logging.info(
        f"Heat loss coefficient: {thermal_model.heat_loss_coefficient:.3f}"
    )
    logging.info(
        f"Outlet effectiveness: {thermal_model.outlet_effectiveness:.3f}"
    )
    logging.info(
        "PV heat weight: "
        f"{thermal_model.external_source_weights.get('pv', 0):.4f}"
    )
    logging.info(
        "Fireplace heat weight: "
        f"{thermal_model.external_source_weights.get('fireplace', 0):.2f}"
    )
    logging.info(
        "TV heat weight: "
        f"{thermal_model.external_source_weights.get('tv', 0):.2f}"
    )
    logging.info(f"Optimization MAE: {optimized_params['mae']:.4f}°C")
    logging.info(
        f"Learning confidence: {thermal_model.learning_confidence:.3f}"
    )

    # Step 5: Save thermal learning state to unified thermal state manager
    logging.info(
        "Step 5: Saving calibrated parameters to unified thermal state..."
    )
    try:
        state_manager = get_thermal_state_manager()

        # Use optimized parameters as calibrated baseline
        calibrated_params = {
            'thermal_time_constant': optimized_params['thermal_time_constant'],
            'heat_loss_coefficient': optimized_params['heat_loss_coefficient'],
            'outlet_effectiveness': optimized_params['outlet_effectiveness'],
            'pv_heat_weight': pv_weight,
            'fireplace_heat_weight': fireplace_weight,
            'tv_heat_weight': tv_weight
        }

        # Set as calibrated baseline
        state_manager.set_calibrated_baseline(
            calibrated_params, calibration_cycles=len(stable_periods)
        )

        # Explicitly set confidence to 3.0 after calibration
        state_manager.update_learning_state(learning_confidence=3.0)

        logging.info(
            "✅ Calibrated parameters (scipy-optimized) saved to unified "
            "thermal state"
        )
        logging.info(
            "✅ Parameters will be automatically loaded on next restart"
        )
        logging.info(
            "🔄 Restart ml_heating service to use calibrated thermal model"
        )

    except Exception as e:
        logging.error(f"❌ Failed to save calibrated parameters: {e}")
        # Fallback to old method
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'learning_confidence': thermal_model.learning_confidence,
        }
        save_state(thermal_learning_state=thermal_learning_state)
        logging.warning(
            "⚠️ Used fallback save method - parameters may not persist"
        )

    return thermal_model


def validate_thermal_model():
    """Validate thermal equilibrium model behavior across temperature ranges"""

    logging.info("=== THERMAL MODEL VALIDATION ===")

    try:
        # Import centralized thermal configuration

        # Initialize thermal model
        thermal_model = ThermalEquilibriumModel()

        logging.info("Testing thermal equilibrium physics compliance:")
        print("\nOUTLET TEMP → EQUILIBRIUM TEMP")
        print("=" * 35)

        # Test monotonicity
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

        is_monotonic = all(monotonic_check[i] <= monotonic_check[i+1]
                           for i in range(len(monotonic_check)-1))

        print(f"\n{'✅' if is_monotonic else '❌'} Physics compliance: "
              f"{'PASSED' if is_monotonic else 'FAILED'}")
        print(f"Range: {min(monotonic_check):.2f}°C to "
              f"{max(monotonic_check):.2f}°C")

        # Test parameter bounds
        logging.info("\n=== THERMAL PARAMETER BOUNDS TEST ===")
        params_ok = True

        thermal_bounds = ThermalParameterConfig.get_bounds(
            'thermal_time_constant'
        )
        if not (thermal_bounds[0] <=
                thermal_model.thermal_time_constant <= thermal_bounds[1]):
            logging.error(
                "Thermal time constant out of bounds: "
                f"{thermal_model.thermal_time_constant:.2f}h "
                f"(bounds: {thermal_bounds})"
            )
            params_ok = False

        if params_ok:
            logging.info("✅ All thermal parameters within physical bounds")
        else:
            logging.error("❌ Some thermal parameters out of bounds")

        # Test adaptive learning system
        logging.info("\n=== ADAPTIVE LEARNING SYSTEM TEST ===")

        test_context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0,
            'current_indoor': 21.0
        }

        initial_confidence = thermal_model.learning_confidence

        # Simulate good predictions
        for _ in range(10):
            predicted = thermal_model.predict_equilibrium_temperature(
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
        logging.info(
            f"{'✅' if learning_works else '❌'} "
            "Adaptive learning: "
            f"{'WORKING' if learning_works else 'NOT WORKING'}"
        )

        overall_success = is_monotonic and params_ok and learning_works
        logging.info(
            f"\n{'✅' if overall_success else '❌'} "
            f"Overall validation: {'PASSED' if overall_success else 'FAILED'}"
        )

        return overall_success

    except Exception as e:
        logging.error("Thermal model validation error: %s", e, exc_info=True)
        return False


def fetch_historical_data_for_calibration(lookback_hours=672):
    """Fetch historical data for calibration."""
    logging.info(f"=== FETCHING {lookback_hours} HOURS OF HISTORICAL DATA ===")

    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )

    df = influx.get_training_data(lookback_hours=lookback_hours)

    if df.empty:
        logging.error("❌ No historical data available")
        return None

    logging.info(f"✅ Fetched {len(df)} samples ({len(df)/12:.1f} hours)")

    required_columns = [
        config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.PV_POWER_ENTITY_ID.split(".", 1)[-1],
        config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    ]

    # Optional new sensors (don't fail if missing to maintain backward
    # compatibility)
    # optional_columns = [
    #     config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1],
    #     config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1],
    #     config.POWER_CONSUMPTION_ENTITY_ID.split(".", 1)[-1]
    # ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return None

    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    if fireplace_col not in df.columns:
        logging.info(
            f"⚠️  Optional fireplace column '{fireplace_col}' not found "
            "- will use 0"
        )

    logging.info("✅ All required columns present")
    return df


def main():
    """Main function to run training and validation."""
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s - %(message)s'
    )

    try:
        print("🚀 Starting thermal equilibrium model training...")
        thermal_model = train_thermal_equilibrium_model()

        if thermal_model:
            print("✅ Thermal training completed successfully!")

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
        logging.error("Main execution error: %s", e, exc_info=True)
        return False


def filter_transient_periods(df):
    """
    Filter for transient periods (heating up or cooling down) for dynamic calibration.
    Returns a list of samples, each containing current state and next state.
    """
    logging.info("=== FILTERING FOR TRANSIENT PERIODS ===")

    if df is None or df.empty:
        return []

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]

    transient_samples = []

    # Ensure sorted by time
    df = df.sort_values('_time')

    # Calculate time diff in minutes
    # Note: We need to handle the case where _time might not be datetime or might be missing
    try:
        time_diffs = df['_time'].diff().dt.total_seconds() / 60.0
    except Exception as e:
        logging.warning(f"Could not calculate time differences: {e}")
        return []

    # Calculate temp change
    df['temp_change'] = df[indoor_col].diff()

    count = 0
    # Iterate through rows
    # We use iloc, so we need integer indices
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Check if consecutive (approx 5 mins)
        t_diff = time_diffs.iloc[i]
        if np.isnan(t_diff) or not (4.0 <= t_diff <= 6.0):
            continue

        # Heuristic for transient/active periods:
        # 1. Heating is ON (outlet > indoor + 5) OR
        # 2. Temperature is changing significantly (abs(delta) > 0.05)
        # We want to capture the dynamics.

        is_heating = (prev_row[outlet_col] > prev_row[indoor_col] + 5.0)
        is_changing = abs(row['temp_change']) > 0.05

        if is_heating or is_changing:
            sample = {
                'current_indoor': prev_row[indoor_col],
                'next_indoor': row[indoor_col],
                'outlet_temp': prev_row[outlet_col],
                'outdoor_temp': prev_row[outdoor_col],
                'pv_power': prev_row.get(pv_col, 0),
                'fireplace_on': prev_row.get(fireplace_col, 0),
                'tv_on': prev_row.get(tv_col, 0),
                'time_step_hours': t_diff / 60.0
            }
            transient_samples.append(sample)
            count += 1

    logging.info(f"✅ Found {len(transient_samples)} transient samples")
    return transient_samples


def calculate_cooling_time_constant(df):
    """
    Estimate thermal time constant from cooling periods (e.g. during DHW cycles).
    Uses log-linear regression on (T_indoor - T_outdoor).
    Returns (tau, r_squared).
    """
    logging.info("=== CALCULATING COOLING TIME CONSTANT (DHW PERIODS) ===")

    if df is None or df.empty:
        return None, 0.0

    # Identify columns
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]

    if dhw_col not in df.columns:
        logging.warning(
            "⚠️ DHW status column not found - cannot estimate cooling constant"
        )
        return None, 0.0

    # Work on a copy to avoid modifying the original dataframe
    df_cooling = df.copy()
    df_cooling = df_cooling.sort_values('_time')

    # Find continuous DHW blocks
    # Create a group identifier that changes when DHW status changes
    df_cooling['dhw_group'] = (
        df_cooling[dhw_col] != df_cooling[dhw_col].shift()
    ).cumsum()

    cooling_taus = []

    for _, group in df_cooling.groupby('dhw_group'):
        if group[dhw_col].iloc[0] == 0:
            continue  # Not a DHW period

        if len(group) < 4:  # Need some points (assuming 5 min intervals)
            continue

        # Calculate time in hours from start of block
        times = (
            group['_time'] - group['_time'].iloc[0]
        ).dt.total_seconds() / 3600.0

        # Calculate log diff
        # T(t) - T_out = Delta * exp(-t/tau)
        # ln(T(t) - T_out) = ln(Delta) - t/tau
        # y = c + m*x
        # m = -1/tau => tau = -1/m

        # Use average outdoor temp for the block to simplify
        avg_outdoor = group[outdoor_col].mean()
        temp_diffs = group[indoor_col] - avg_outdoor

        # Filter for valid log inputs
        # Ensure indoor is at least 2 degrees above outdoor
        valid_indices = temp_diffs > 2.0
        if valid_indices.sum() < 4:
            continue

        y = np.log(temp_diffs[valid_indices])
        x = times[valid_indices]

        if len(x) < 4:
            continue

        # Linear regression
        # slope = cov(x, y) / var(x)
        # intercept = mean(y) - slope * mean(x)

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R^2
        mean_y = np.mean(y)
        ss_tot = np.sum((y - mean_y)**2)
        ss_res = np.sum((y - (slope * x + (sum_y - slope * sum_x)/n))**2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Expecting cooling, so slope should be negative
        if slope >= -0.001:  # Too flat or heating up
            continue

        tau = -1.0 / slope

        # Sanity check tau (1h to 200h)
        if 1.0 < tau < 200.0 and r_squared > 0.8:
            cooling_taus.append({
                'tau': tau,
                'r2': r_squared,
                'duration': times.max()
            })

    if not cooling_taus:
        logging.info("No valid cooling periods found")
        return None, 0.0

    # Weighted average by R^2 * duration
    total_weight = sum(item['r2'] * item['duration'] for item in cooling_taus)
    weighted_tau = sum(
        item['tau'] * item['r2'] * item['duration'] for item in cooling_taus
    ) / total_weight

    avg_r2 = sum(item['r2'] for item in cooling_taus) / len(cooling_taus)

    logging.info(
        f"✅ Estimated cooling time constant: {weighted_tau:.2f}h "
        f"(from {len(cooling_taus)} periods, avg R²={avg_r2:.2f})"
    )

    return weighted_tau, avg_r2


def calibrate_transient_parameters(thermal_model, transient_samples):
    """
    Calibrate thermal time constant using transient samples.
    """
    logging.info("=== CALIBRATING TRANSIENT PARAMETERS ===")

    if not transient_samples:
        return None

    if minimize is None:
        logging.error("scipy.optimize.minimize not available")
        return None

    def objective(params):
        tau = params[0]
        mse = 0.0

        for s in transient_samples:
            # Predict equilibrium
            eq_temp = thermal_model.predict_equilibrium_temperature(
                outlet_temp=s['outlet_temp'],
                outdoor_temp=s['outdoor_temp'],
                current_indoor=s['current_indoor'],
                pv_power=s['pv_power'],
                fireplace_on=s['fireplace_on'],
                tv_on=s['tv_on'],
                _suppress_logging=True
            )

            # Predict next step
            # T_next = T_curr + (T_eq - T_curr) * (1 - exp(-dt/tau))
            dt = s['time_step_hours']
            approach = 1 - np.exp(-dt / tau)
            pred_next = s['current_indoor'] + (eq_temp - s['current_indoor']) * approach

            mse += (pred_next - s['next_indoor']) ** 2

        return mse / len(transient_samples)

    # Initial guess
    initial_tau = thermal_model.thermal_time_constant
    # Use wider bounds to allow for faster/slower thermal responses
    # 1.0h = very fast (tent), 100.0h = very slow (bunker)
    bounds = [(1.0, 100.0)]

    try:
        result = minimize(
            objective,
            [initial_tau],
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result.success:
            best_tau = result.x[0]
            logging.info(
                f"✅ Optimized thermal_time_constant: {best_tau:.2f}h "
                f"(MSE: {result.fun:.6f})"
            )
            return best_tau
        else:
            logging.error(f"❌ Transient optimization failed: {result.message}")
            return None
    except Exception as e:
        logging.error(f"❌ Transient optimization error: {e}")
        return None


def filter_stable_periods(df, temp_change_threshold=0.2, min_duration=20):
    """Filter for stable periods with blocking state detection."""
    logging.info(
        "=== FILTERING FOR STABLE PERIODS WITH BLOCKING STATE DETECTION ==="
    )

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]

    # New sensors
    flow_rate_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfect_col = config.DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    boost_col = config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]

    grace_period_minutes = config.GRACE_PERIOD_MAX_MINUTES
    grace_period_samples = grace_period_minutes // 5

    stable_periods = []
    window_size = min_duration // 5

    logging.info(
        f"Looking for periods with <{temp_change_threshold}°C change"
        f" over {min_duration} min"
    )
    logging.info(
        f"Using {grace_period_minutes}min grace periods after blocking states"
    )

    filter_stats = {
        'total_checked': 0, 'missing_data': 0, 'temp_unstable': 0,
        'blocking_active': 0, 'grace_period': 0, 'fireplace_changed': 0,
        'outlet_unstable': 0, 'low_flow': 0, 'passed': 0
    }

    for i in range(window_size + grace_period_samples, len(df) - window_size):
        filter_stats['total_checked'] += 1

        window_start = i - window_size // 2
        window_end = i + window_size // 2
        window = df.iloc[window_start:window_end]

        grace_start = i - grace_period_samples
        grace_end = i + window_size // 2
        grace_window = df.iloc[grace_start:grace_end]

        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:
            filter_stats['missing_data'] += 1
            continue

        temp_range = indoor_temps.max() - indoor_temps.min()
        temp_std = indoor_temps.std()

        if (temp_range > temp_change_threshold or
                temp_std > temp_change_threshold / 2):
            filter_stats['temp_unstable'] += 1
            continue

        blocking_detected, _ = check_blocking_states(
            grace_window, dhw_col, defrost_col, disinfect_col, boost_col
        )

        if blocking_detected:
            current_window_blocking, _ = check_blocking_states(
                window, dhw_col, defrost_col, disinfect_col, boost_col
            )
            if current_window_blocking:
                filter_stats['blocking_active'] += 1
            else:
                filter_stats['grace_period'] += 1
            continue

        if fireplace_col in window.columns:
            if window[fireplace_col].nunique() > 1:
                filter_stats['fireplace_changed'] += 1
                continue

        if outlet_col in window.columns:
            outlet_temps = window[outlet_col].dropna()
            if len(outlet_temps) >= window_size * 0.8:
                if outlet_temps.std() > 2.0:
                    filter_stats['outlet_unstable'] += 1
                    continue

        # Filter by flow rate if available (ensure pump is running)
        if flow_rate_col in window.columns:
            flow_rates = window[flow_rate_col].dropna()
            if not flow_rates.empty:
                avg_flow = flow_rates.mean()
                # If flow is too low (< 10 L/h), heat pump is likely off/idle
                if avg_flow < 10.0:
                    filter_stats['low_flow'] = (
                        filter_stats.get('low_flow', 0) + 1
                    )
                    continue

        center_row = df.iloc[i]

        # Calculate thermal power if available
        thermal_power_kw = 0.0
        if (inlet_col in df.columns and flow_rate_col in df.columns and
                outlet_col in df.columns):
            inlet_val = center_row.get(inlet_col)
            flow_val = center_row.get(flow_rate_col)
            outlet_val = center_row.get(outlet_col)

            if (inlet_val is not None and flow_val is not None and
                    outlet_val is not None):
                delta_t = outlet_val - inlet_val
                # Q = m * c * dt (flow in L/h -> kg/s: / 3600)
                thermal_power_kw = (
                    (flow_val / 3600.0) *
                    config.SPECIFIC_HEAT_CAPACITY *
                    delta_t
                )

        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': center_row[outlet_col],
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0),
            'thermal_power_kw': thermal_power_kw,
            'timestamp': center_row['_time'],
            'stability_score': 1.0 / (temp_std + 0.01),
            'outlet_stability': 1.0 / (outlet_temps.std() + 0.01)
            if outlet_col in window.columns else 1.0
        }
        stable_periods.append(period)
        filter_stats['passed'] += 1

    log_filtering_stats(filter_stats)

    logging.info(
        f"✅ Found {len(stable_periods)} stable periods with blocking "
        "state filtering"
    )

    with open("/opt/ml_heating/stable_periods.json", "w") as f:
        json.dump(stable_periods, f, indent=2, default=str)

    return stable_periods


def check_blocking_states(df, dhw_col, defrost_col, disinfect_col, boost_col):
    """Check for blocking states in a DataFrame."""
    blocking_detected = False
    blocking_reasons = []
    if dhw_col in df.columns and df[dhw_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('dhw_heating')
    if defrost_col in df.columns and df[defrost_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('defrosting')
    if disinfect_col in df.columns and df[disinfect_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('disinfection')
    if boost_col in df.columns and df[boost_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('boost_heater')
    return blocking_detected, blocking_reasons


def log_filtering_stats(stats):
    """Log statistics from the filtering process."""
    logging.info("=== BLOCKING STATE FILTERING RESULTS ===")
    logging.info(f"Total periods checked: {stats['total_checked']}")
    logging.info(f"Stable periods found: {stats['passed']}")
    logging.info("Filter exclusions:")
    logging.info(f"  Missing data: {stats['missing_data']}")
    logging.info(f"  Temperature unstable: {stats['temp_unstable']}")
    logging.info(f"  Blocking states active: {stats['blocking_active']}")
    logging.info(f"  Grace period recovery: {stats['grace_period']}")
    logging.info(f"  Fireplace state changes: {stats['fireplace_changed']}")
    logging.info(f"  Outlet temperature unstable: {stats['outlet_unstable']}")
    logging.info(f"  Low flow rate: {stats.get('low_flow', 0)}")

    retention = (stats['passed'] / stats['total_checked']) * 100 \
        if stats['total_checked'] > 0 else 0
    logging.info(f"Data retention rate: {retention:.1f}%")


def debug_thermal_predictions(stable_periods, sample_size=5):
    """Debug thermal model predictions on sample data."""
    logging.info("=== DEBUGGING THERMAL PREDICTIONS ===")

    test_model = ThermalEquilibriumModel()

    logging.info("Testing thermal model on sample periods:")
    for i, period in enumerate(stable_periods[:sample_size]):
        predicted_temp = test_model.predict_equilibrium_temperature(
            outlet_temp=period['outlet_temp'],
            outdoor_temp=period['outdoor_temp'],
            current_indoor=period.get(
                'indoor_temp', period['outdoor_temp'] + 10.0
            ),
            pv_power=period['pv_power'],
            fireplace_on=period['fireplace_on'],
            tv_on=period['tv_on'],
            thermal_power=None,  # Force temperature-based path for consistency
            _suppress_logging=True
        )

        actual_temp = period['indoor_temp']
        error = abs(predicted_temp - actual_temp)

        logging.info(f"Sample {i+1}:")
        logging.info(
            f"  Outlet: {period['outlet_temp']:.1f}°C, "
            f"Outdoor: {period['outdoor_temp']:.1f}°C"
        )
        logging.info(
            "  PV: %.1fW, Fireplace: %.0f, "
            "TV: %.0f",
            period['pv_power'], period['fireplace_on'], period['tv_on']
        )
        logging.info(
            f"  Predicted: {predicted_temp:.1f}°C, "
            f"Actual: {actual_temp:.1f}°C"
        )
        logging.info(f"  Error: {error:.1f}°C")
        logging.info("")


def calculate_direct_heat_loss(stable_periods):
    """
    Calculate heat loss coefficient directly from periods with known thermal power
    and minimal external heat sources.

    U = P_thermal / (T_indoor - T_outdoor)
    """
    logging.info("=== CALCULATING DIRECT HEAT LOSS COEFFICIENT ===")

    u_values = []

    for p in stable_periods:
        # Check if we have thermal power data
        if 'thermal_power_kw' not in p or not p['thermal_power_kw']:
            continue

        # Filter for pure heating periods (minimal external sources)
        # Allow small PV/TV but ensure they are minor compared to heating
        if (p.get('pv_power', 0) > 100 or
                p.get('fireplace_on', 0) > 0 or
                p.get('tv_on', 0) > 0):
            continue

        # Ensure significant temperature difference and heating power
        # to avoid division by zero or noise amplification
        delta_t = p['indoor_temp'] - p['outdoor_temp']
        if delta_t < 5.0 or p['thermal_power_kw'] < 0.5:
            continue

        # Calculate U = P / dT
        u = p['thermal_power_kw'] / delta_t

        # Sanity check (0.05 to 1.0 kW/K are reasonable values for houses)
        if 0.05 <= u <= 1.0:
            u_values.append(u)

    if not u_values:
        logging.warning(
            "⚠️ No suitable periods found for direct heat loss calculation"
        )
        return None

    avg_u = sum(u_values) / len(u_values)
    std_u = (sum((x - avg_u) ** 2 for x in u_values) / len(u_values)) ** 0.5

    logging.info(f"✅ Calculated direct heat loss from {len(u_values)} periods")
    logging.info(f"  Mean U: {avg_u:.4f} kW/K")
    logging.info(f"  Std Dev: {std_u:.4f}")

    return avg_u


def optimize_thermal_parameters(stable_periods):
    """Multi-parameter optimization with data availability checks."""
    logging.info(
        "=== MULTI-PARAMETER OPTIMIZATION WITH DATA AVAILABILITY CHECKS ==="
    )

    if minimize is None:
        logging.error("❌ scipy not available - cannot optimize parameters")
        return None

    debug_thermal_predictions(stable_periods)

    logging.info("=== CHECKING DATA AVAILABILITY ===")

    total_periods = len(stable_periods)
    data_stats = {
        'pv_power': sum(
            1 for p in stable_periods if p.get('pv_power', 0) > 0
        ),
        'fireplace_on': sum(
            1 for p in stable_periods if p.get('fireplace_on', 0) > 0
        ),
        'tv_on': sum(1 for p in stable_periods if p.get('tv_on', 0) > 0)
    }

    data_availability = {}
    for source, count in data_stats.items():
        percentage = (count / total_periods) * 100
        data_availability[source] = percentage
        logging.info(
            f"  {source}: {count}/{total_periods} periods ({percentage:.1f}%)"
        )

    excluded_params = []
    min_usage_threshold = 1.0

    if data_availability['fireplace_on'] <= min_usage_threshold:
        excluded_params.append('fireplace_heat_weight')
        logging.info(
            "  🚫 Excluding fireplace_heat_weight "
            f"(only {data_availability['fireplace_on']:.1f}% usage)"
        )

    if data_availability['tv_on'] <= min_usage_threshold:
        excluded_params.append('tv_heat_weight')
        logging.info(
            "  🚫 Excluding tv_heat_weight "
            f"(only {data_availability['tv_on']:.1f}% usage)"
        )

    if data_availability['pv_power'] <= min_usage_threshold:
        excluded_params.append('pv_heat_weight')
        logging.info(
            "  🚫 Excluding pv_heat_weight "
            f"(only {data_availability['pv_power']:.1f}% usage)"
        )

    # Try to calculate direct heat loss first
    direct_u = calculate_direct_heat_loss(stable_periods)

    initial_heat_loss = ThermalParameterConfig.get_default(
        'heat_loss_coefficient'
    )
    if direct_u is not None:
        initial_heat_loss = direct_u
        logging.info(
            f"Using calculated heat loss {direct_u:.4f} as initial guess"
        )

    current_params = {
        'thermal_time_constant':
            ThermalParameterConfig.get_default('thermal_time_constant'),
        'heat_loss_coefficient': initial_heat_loss,
        'outlet_effectiveness':
            ThermalParameterConfig.get_default('outlet_effectiveness'),
        'pv_heat_weight':
            ThermalParameterConfig.get_default('pv_heat_weight'),
        'fireplace_heat_weight':
            ThermalParameterConfig.get_default('fireplace_heat_weight'),
        'tv_heat_weight': ThermalParameterConfig.get_default('tv_heat_weight')
    }

    logging.info("=== PARAMETERS FOR OPTIMIZATION ===")
    for param, value in current_params.items():
        if param in excluded_params:
            logging.info(f"  {param}: {value} (FIXED - insufficient data)")
        else:
            logging.info(f"  {param}: {value} (OPTIMIZE)")

    param_names, param_values, param_bounds = build_optimization_params(
        current_params, excluded_params, heat_loss_center=direct_u
    )

    logging.info(f"Optimizing {len(param_names)} parameters: {param_names}")

    def objective_function(params):
        """Calculate MAE for given parameters."""
        return calculate_mae_for_params(
            params, param_names, stable_periods, current_params
        )

    logging.info(
        f"Starting optimization with {len(stable_periods)} periods..."
    )
    logging.info("This may take a few minutes...")

    logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = minimize(
            objective_function,
            x0=param_values,
            bounds=param_bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-3,
                'disp': True,
                'iprint': 2
            }
        )

        log_optimization_results(result, param_names, param_values)

        if result.success:
            optimized_params = build_optimized_params(
                result, current_params, param_names, excluded_params
            )

            log_optimized_parameters(
                optimized_params, current_params, excluded_params
            )
            return optimized_params

        else:
            logging.error(f"❌ Optimization failed: {result.message}")
            return None

    except Exception as e:
        logging.error(f"❌ Optimization error: {e}")
        return None


def build_optimization_params(
    current_params, excluded_params, heat_loss_center=None
):
    """Build lists of parameters for optimization."""
    param_names = []
    param_values = []
    param_bounds = []

    core_params = [
        'heat_loss_coefficient',
        'outlet_effectiveness'
    ]

    # Note: thermal_time_constant cannot be calibrated using stable (equilibrium)
    # periods because the time-dependent exponential term decays to zero.
    # It requires dynamic/transient data for calibration.

    for p in core_params:
        param_names.append(p)
        param_values.append(current_params[p])

        # Special handling for heat loss if we have a direct calculation
        if p == 'heat_loss_coefficient' and heat_loss_center is not None:
            # Constrain to +/- 30% of calculated value
            lower = heat_loss_center * 0.7
            upper = heat_loss_center * 1.3
            # But keep within absolute physical bounds
            abs_bounds = ThermalParameterConfig.get_bounds(p)
            final_lower = max(lower, abs_bounds[0])
            final_upper = min(upper, abs_bounds[1])
            param_bounds.append((final_lower, final_upper))
            logging.info(
                f"  Constraining heat_loss_coefficient to {final_lower:.4f}-"
                f"{final_upper:.4f} based on direct calculation"
            )
        elif p == 'heat_loss_coefficient':
            # Expand bounds for poorly insulated buildings if no direct calculation
            # Default bounds might be too restrictive (e.g. 0.5 max)
            default_bounds = ThermalParameterConfig.get_bounds(p)
            expanded_upper = max(default_bounds[1], 2.0)  # Allow up to 2.0
            param_bounds.append((default_bounds[0], expanded_upper))
            logging.info(
                f"  Using expanded bounds for heat_loss_coefficient: "
                f"{default_bounds[0]}-{expanded_upper}"
            )
        else:
            param_bounds.append(ThermalParameterConfig.get_bounds(p))

    heat_source_params = {
        'pv_heat_weight': (0.0005, 0.005),
        'fireplace_heat_weight': (1.0, 6.0),
        'tv_heat_weight': (0.1, 1.5)
    }

    for param, bounds in heat_source_params.items():
        if param not in excluded_params:
            param_names.append(param)
            param_values.append(current_params[param])
            param_bounds.append(bounds)

    return param_names, param_values, param_bounds


def calculate_mae_for_params(
    params, param_names, stable_periods, current_params
):
    """Calculate MAE for a given set of parameters."""
    total_error = 0.0
    valid_predictions = 0

    param_dict = dict(zip(param_names, params))

    debug_str = ", ".join(
        [f"{name}={val:.4f}" for name, val in param_dict.items()]
    )
    if not hasattr(calculate_mae_for_params, 'call_count'):
        calculate_mae_for_params.call_count = 1
    else:
        calculate_mae_for_params.call_count += 1

    if calculate_mae_for_params.call_count % 10 == 1:
        logging.debug("Testing params: %s", debug_str)

    test_periods = stable_periods[::5]

    # Suppress INFO logs from ThermalEquilibriumModel initialization
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    for period in test_periods:
        try:
            test_model = ThermalEquilibriumModel()
            test_model.thermal_time_constant = param_dict.get(
                'thermal_time_constant',
                current_params['thermal_time_constant']
            )
            test_model.heat_loss_coefficient = param_dict[
                'heat_loss_coefficient'
            ]
            test_model.outlet_effectiveness = param_dict[
                'outlet_effectiveness'
            ]

            test_model.external_source_weights['pv'] = param_dict.get(
                'pv_heat_weight', current_params['pv_heat_weight']
            )
            test_model.external_source_weights['fireplace'] = param_dict.get(
                'fireplace_heat_weight',
                current_params['fireplace_heat_weight']
            )
            test_model.external_source_weights['tv'] = param_dict.get(
                'tv_heat_weight', current_params['tv_heat_weight']
            )

            # FORCE TEMPERATURE-BASED PHYSICS:
            # We intentionally pass thermal_power=None here even if available.
            # Why?
            # 1. If we use thermal_power, the model uses the energy balance equation
            #    which ignores 'outlet_effectiveness'. This causes outlet_effectiveness
            #    calibration to stagnate (zero gradient).
            # 2. The primary goal of this calibration is to tune the parameters
            #    (heat_loss_coefficient, outlet_effectiveness) so that the
            #    TEMPERATURE-BASED model (used for control planning) accurately
            #    predicts equilibrium.
            # 3. We want the controller's "Required Outlet Temp" calculation to be
            #    accurate, and that calculation relies on the relationship between
            #    outlet_temp and equilibrium established by these two parameters.

            predicted_temp = test_model.predict_equilibrium_temperature(
                outlet_temp=period['outlet_temp'],
                outdoor_temp=period['outdoor_temp'],
                current_indoor=period.get(
                    'indoor_temp', period['outdoor_temp'] + 10.0
                ),
                pv_power=period['pv_power'],
                fireplace_on=period['fireplace_on'],
                tv_on=period['tv_on'],
                thermal_power=None,  # Force temperature-based path
                _suppress_logging=True
            )

            error = abs(predicted_temp - period['indoor_temp'])

            if error > 50.0:
                if calculate_mae_for_params.call_count <= 1 and valid_predictions == 0:
                    logging.debug(
                        f"Skipping large error: {error:.1f} "
                        f"(Pred: {predicted_temp:.1f}, "
                        f"Actual: {period['indoor_temp']:.1f})"
                    )
                continue

            total_error += error
            valid_predictions += 1

        except Exception as e:
            if calculate_mae_for_params.call_count <= 1:
                logging.debug(f"Prediction exception: {e}")
            continue

    root_logger.setLevel(original_level)

    if valid_predictions == 0:
        if calculate_mae_for_params.call_count <= 1:
            logging.warning("No valid predictions found in MAE calculation!")
        return 1000.0

    mae = total_error / valid_predictions

    if calculate_mae_for_params.call_count % 10 == 1:
        logging.debug("MAE for %s: %.4f", debug_str, mae)

    return mae


def log_optimization_results(result, param_names, param_values):
    """Log the results of the optimization."""
    logging.info("🔍 SCIPY OPTIMIZATION RESULTS:")
    logging.info(f"  Success: {result.success}")
    logging.info(f"  Message: {result.message}")
    logging.info(f"  Function evaluations: {result.nfev}")
    logging.info(
        f"  Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}"
    )
    logging.info(f"  Final function value: {result.fun:.6f}")

    logging.info("  Parameter changes:")
    for i, param_name in enumerate(param_names):
        initial_val = param_values[i]
        final_val = result.x[i]
        change = final_val - initial_val
        logging.info(
            f"    {param_name}: {initial_val:.6f} → {final_val:.6f} "
            f"(Δ{change:+.6f})"
        )


def build_optimized_params(
    result, current_params, param_names, excluded_params
):
    """Build the dictionary of optimized parameters."""
    optimized_params = dict(current_params)

    for i, param_name in enumerate(param_names):
        optimized_params[param_name] = result.x[i]

    optimized_params['mae'] = result.fun
    optimized_params['optimization_success'] = True
    optimized_params['excluded_parameters'] = excluded_params
    return optimized_params


def log_optimized_parameters(
    optimized_params, current_params, excluded_params
):
    """Log the final optimized parameters."""
    logging.info("✅ Optimization completed successfully!")
    logging.info("Optimized parameters:")
    for param, value in optimized_params.items():
        if param not in ['mae', 'optimization_success', 'excluded_parameters']:
            old_value = current_params[param]
            if param in excluded_params:
                logging.info(f"  {param}: {value:.4f} (FIXED - no data)")
            else:
                change_pct = (
                    ((value - old_value) / old_value) * 100 if old_value else 0
                )
                logging.info(
                    f"  {param}: {value:.4f} "
                    f"(was {old_value:.4f}, {change_pct:+.1f}%)"
                )

    logging.info(f"Final MAE: {optimized_params['mae']:.4f}°C")


def backup_existing_calibration():
    """Create a backup of the existing thermal calibration."""
    logging.info("Creating backup of existing thermal calibration...")

    try:
        from datetime import datetime

        state_manager = get_thermal_state_manager()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pre_calibration_{timestamp}.json"

        success, result = state_manager.create_backup(backup_name)

        if success:
            logging.info(f"✅ Backup created: {result}")
            return result
        else:
            logging.warning(f"Failed to create calibration backup: {result}")
            return None

    except Exception as e:
        logging.warning(f"Failed to create calibration backup: {e}")
        return None


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
