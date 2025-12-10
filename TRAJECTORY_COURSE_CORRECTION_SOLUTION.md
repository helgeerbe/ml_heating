# Trajectory Course Correction Solution

## Problem Analysis
The overnight indoor temperature was dropping while outlet temperature was fixed at 26°C, despite ML heating being designed to guarantee stable indoor temperature.

## Root Cause Investigation
1. **Missing Trajectory Verification**: Binary search found equilibrium outlet temperature but didn't verify it would actually reach target within reasonable time
2. **No Forecast Integration**: System used current outdoor temp (9°C) and PV (500W) for predictions, but overnight reality was 5°C and 0W
3. **Insufficient Correction Aggressiveness**: 1°C drop from 21°C to 20°C only received mild 15% correction, but represents significant comfort loss

## Complete Solution Implemented

### 1. Enhanced Control Flow
```
Binary Search → Trajectory Verification → Course Correction → Final Outlet Temp
```

**Before**: Binary search only (equilibrium-based)
**After**: Three-stage verification with real-time trajectory checking

### 2. Gentle Additive Trajectory Correction (FINAL - User Approved)
**Inspired by heat curve logic but scaled for direct outlet temperature adjustment**
- **Gentle (≤0.5°C)**: +5°C per degree of trajectory error
- **Moderate (0.5-1.0°C)**: +8°C per degree of trajectory error  
- **Aggressive (>1.0°C)**: +12°C per degree of trajectory error

**Impact**: 
- 0.5°C error gets +2.5°C correction (32.1°C → 34.6°C, reasonable adjustment)
- 0.8°C error gets +6.4°C correction (moderate push without doubling outlet temp)
- Much gentler than multiplicative approach that would double outlet temperatures
- Based on user's heat curve automation using 15°C per degree for shift values

### 3. Forecast Integration
Enhanced trajectory prediction to use:
- **Temperature forecasts**: `temp_forecast_1h` through `temp_forecast_4h`
- **PV forecasts**: `pv_forecast_1h` through `pv_forecast_4h`
- **Fallback**: Uses forecast averages when detailed method unavailable

### 4. Implementation Details

#### Binary Search Context
- No forecast data available during iterative search
- Uses empty features dict as fallback
- Applies course correction at convergence

#### Full Features Context  
- Complete forecast arrays available from `physics_features.py`
- Enhanced trajectory prediction with time-varying conditions
- Accounts for overnight temperature drops and PV shutdown

## Technical Implementation

### File: `src/model_wrapper.py`
```python
def _verify_trajectory_and_correct(self, outlet_temp, current_indoor, target_indoor, 
                                   outdoor_temp, thermal_features, features=None):
    """
    Verify outlet temp will reach target using forecasts and apply course correction.
    """
    if features:
        # Use forecast data for accurate trajectory prediction
        outdoor_forecast = [features.get(f'temp_forecast_{i}h', outdoor_temp) for i in range(1,5)]
        pv_forecast = [features.get(f'pv_forecast_{i}h', thermal_features.get('pv_power', 0.0)) for i in range(1,5)]
    else:
        # Fallback to current values
        outdoor_forecast = [outdoor_temp] * 4
        pv_forecast = [thermal_features.get('pv_power', 0.0)] * 4
    
    trajectory = self.thermal_model.predict_thermal_trajectory(...)
    
    if trajectory['reaches_target_at'] is None:
        # Apply gentle additive course correction based on error
        temp_error = target_indoor - trajectory['equilibrium_temp']
        if temp_error > 0.1:
            if temp_error <= 0.5:
                correction_amount = temp_error * 5.0  # +5°C per degree - gentle
            elif temp_error <= 1.0:
                correction_amount = temp_error * 8.0  # +8°C per degree - moderate
            else:
                correction_amount = temp_error * 12.0  # +12°C per degree - aggressive
            return outlet_temp + correction_amount
```

### Configuration Integration
- **Forecast entities**: Already configured in `.env_sample`
- **Features building**: `physics_features.py` fetches and processes forecasts
- **Trajectory control**: `TRAJECTORY_PREDICTION_ENABLED=true` enables verification

## Results & Validation

### All Tests Passing
- **10/10 trajectory course correction tests**: ✅
- **Enhanced model wrapper tests**: ✅
- **No regressions detected**: ✅

### RESOLVED: Forecast Integration Issue
**Problem**: Initial logs showed identical forecast and current values.
**Root Cause**: PV forecast parsing looked for `forecast` attribute instead of `watts` attribute.
**Solution**: Fixed PV forecast parsing to use correct `watts` attribute with ISO timestamp keys.

**Results After Fix**:
- **Temperature forecasts**: Working correctly (0.0°C to 0.3°C variations)
- **PV forecasts**: **MAJOR improvement** - up to 644W variations vs current values
- **Trajectory corrections**: Now using realistic changing conditions instead of static assumptions

### Production Validation
1. **Enhanced forecast integration**: ✅ PV forecasts [1180W, 877W, 409W, 87W] vs current 731W
2. **Gentle additive corrections**: ✅ Applied +2.5°C correction (32.1°C→34.6°C) for 0.5°C error instead of aggressive 65°C  
3. **Realistic overnight predictions**: ✅ Accounts for PV shutdown (87W in 4 hours)
4. **Heat curve alignment**: ✅ Additive approach (5°C/8°C/12°C per degree) prevents outlet temperature spikes

### Expected Impact
1. **Overnight stability**: System anticipates temperature drops and PV shutdown with real forecast data
2. **Gentle response**: Additive correction (+5°C per degree for ≤0.5°C, +12°C per degree for >1.0°C)
3. **Forecast-aware**: Uses both temperature and PV forecasts for accurate trajectory planning
4. **Reasonable adjustments**: No more doubling of outlet temperatures - modest increases only

## Usage
The enhanced system is automatically active when:
- `TRAJECTORY_PREDICTION_ENABLED=true` (default)
- Forecast entities configured in `.env`
- ML heating system running normally

No additional configuration required - the improvements are integrated into the existing control loop.
