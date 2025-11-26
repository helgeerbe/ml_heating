# System Architecture & Patterns - ML Heating Control

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Home Assistant │◄────►│  ML Heating      │◄────►│   InfluxDB      │
│                 │      │  Controller      │      │                 │
│ - Sensors       │      │                  │      │ - Historical    │
│ - Controls      │      │ - Physics Model  │      │   Data          │
│ - Metrics       │      │ - Online Learning│      │ - Features      │
└─────────────────┘      │ - Optimization   │      └─────────────────┘
                         └──────────────────┘
```

### Core Components

**1. Main Controller (`main.py`)**
- Orchestrates the entire learning/prediction cycle
- Handles blocking detection and grace periods
- Manages state persistence and error handling
- Implements the main control loop with 30-min cycles

**2. Physics Model (`physics_model.py`)**
- `RealisticPhysicsModel` class with thermodynamic principles
- Multi-lag learning for time-delayed effects
- Seasonal adaptation via cos/sin modulation
- External heat source tracking and learning

**3. Model Wrapper (`model_wrapper.py`)**
- 7-stage prediction pipeline optimization
- Monotonic enforcement for physics compliance
- Smart rounding and gradual temperature control
- Feature importance analysis

**4. Feature Engineering (`physics_features.py`)**
- Builds comprehensive feature vectors from sensor data
- Historical data integration from InfluxDB
- Time-based features (hour, month, cyclical)
- Statistical aggregations and trend analysis

**5. Home Assistant Client (`ha_client.py`)**
- Bidirectional API communication
- Sensor reading and state management
- Metrics publishing and diagnostics
- Error handling and retry logic

**6. InfluxDB Service (`influx_service.py`)**
- Historical data queries for calibration
- Feature importance export
- Learning metrics tracking
- Time-series data management

**7. State Manager (`state_manager.py`)**
- Persistent state between cycles
- Model and metrics serialization
- Configuration and history tracking

## Key Design Patterns

### 1. Physics-Based Machine Learning Pattern

**Principle**: Combine domain knowledge with data-driven learning
```python
# Core physics calculation
base_heating = outlet_effect * self.base_heating_rate
target_boost = temp_gap * self.target_influence  
weather_adjustment = base_heating * outdoor_penalty * self.outdoor_factor

# Data-driven external sources
pv_contribution = self._calculate_pv_lagged(month_cos, month_sin)
fireplace_contribution = self._calculate_fireplace_lagged()

# Total prediction
total_effect = (base_heating + target_boost + weather_adjustment + 
               pv_contribution + fireplace_contribution)
```

**Benefits**:
- Interpretable predictions respecting thermodynamics
- Faster convergence with less training data
- Bounds checking prevents unrealistic outputs
- Domain expertise encoded in model structure

### 2. Online Learning Pattern

**Principle**: Learn from every operational cycle
```python
# At cycle start: learn from previous cycle results
if last_run_features and last_indoor_temp:
    actual_applied_temp = ha_client.get_state(ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID)
    actual_indoor_change = current_indoor - last_indoor_temp
    
    # Update model with actual outcome
    model.learn_one(learning_features, actual_indoor_change)
```

**Benefits**:
- Continuous adaptation to changing conditions
- No separate training/inference phases
- Handles concept drift automatically
- Learns from real operational data

### 3. Multi-Modal Operation Pattern

**Principle**: Support both active control and passive observation

**Active Mode**: ML controls heating directly
```python
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur  # Same entity
```

**Shadow Mode**: ML observes heat curve decisions
```python
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur        # ML calculation
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1  # Heat curve
```

**Benefits**:
- Risk-free testing and validation
- Quantitative comparison between approaches
- Smooth transition from testing to production
- Continuous learning regardless of control mode

### 4. Safety-First Design Pattern

**Principle**: Multiple layers of protection and validation

**Layer 1: Absolute Bounds**
```python
# Hard limits on outlet temperature
CLAMP_MIN_ABS = 14.0°C
CLAMP_MAX_ABS = 65.0°C
```

**Layer 2: Gradual Changes**
```python
# Prevent abrupt temperature jumps
MAX_TEMP_CHANGE_PER_CYCLE = 2°C  # Maximum change per 30-min cycle
```

**Layer 3: Blocking Detection**
```python
# Pause during system blocking events
blocking_entities = [DHW_STATUS, DEFROST_STATUS, DISINFECTION_STATUS]
if any_blocking_active:
    skip_cycle_and_wait()
```

**Layer 4: Grace Periods**
```python
# Stabilization time after blocking ends
if blocking_just_ended:
    restore_previous_target()
    wait_for_outlet_temperature_stabilization()
```

**Layer 5: Physics Validation**
```python
# Monotonic enforcement: higher outlet → higher indoor
def enforce_monotonic(candidates, baseline_outlet):
    # Ensure predictions respect thermodynamic reality
```

### 5. Multi-Lag Learning Pattern

**Principle**: Capture time-delayed thermal effects

**PV Solar (4 lags: 30, 60, 90, 120 minutes)**
```python
# Thermal mass stores and releases solar heat slowly
pv_contribution = (
    pv_history[-2] * pv_coeffs['lag_1'] +  # 30min ago
    pv_history[-3] * pv_coeffs['lag_2'] +  # 60min ago (often peak)
    pv_history[-4] * pv_coeffs['lag_3'] +  # 90min ago
    pv_history[-5] * pv_coeffs['lag_4']    # 120min ago
) * seasonal_multiplier
```

**Fireplace (4 lags: 0, 30, 60, 90 minutes)**
```python
# Immediate radiant + sustained convective heating
fireplace_contribution = (
    fireplace_history[-1] * fireplace_coeffs['immediate'] +  # Direct radiant
    fireplace_history[-2] * fireplace_coeffs['lag_1'] +     # Peak convective
    fireplace_history[-3] * fireplace_coeffs['lag_2'] +     # Sustained
    fireplace_history[-4] * fireplace_coeffs['lag_3']       # Declining
)
```

**Benefits**:
- Realistic modeling of thermal mass effects
- Better prediction accuracy for external heat sources
- Captures complex timing relationships
- Automatic learning of lag coefficients

### 6. Seasonal Adaptation Pattern

**Principle**: Automatic learning of seasonal variations

```python
# Cos/sin modulation for seasonal effects
month_rad = 2 * π * current_month / 12
pv_seasonal_multiplier = 1.0 + (
    pv_seasonal_cos * cos(month_rad) + 
    pv_seasonal_sin * sin(month_rad)
)

# Apply seasonal modulation
pv_effect = base_pv_effect * pv_seasonal_multiplier
```

**Learning from Summer Data**:
```python
# Clean signal when HVAC is off
if not heating_active:
    hvac_off_tracking.append({
        'pv': pv_power,
        'actual_change': temperature_change,
        'month_cos': month_cos,
        'month_sin': month_sin
    })
```

**Benefits**:
- Eliminates manual seasonal recalibration
- Learns realistic seasonal variation (±30-50%)
- Uses clean summer data for baseline learning
- Automatic adaptation to climate patterns

## Critical Implementation Patterns

### 7-Stage Prediction Pipeline

**Stage 1: Optimization Search**
```python
# Test temperature range in 0.5°C steps
for candidate_temp in range(CLAMP_MIN_ABS, CLAMP_MAX_ABS, 0.5):
    predicted_indoor = model.predict_outcome(candidate_temp, features)
    error = abs(predicted_indoor - target_temp)
    if error < best_error:
        best_temp = candidate_temp
```

**Stage 2: Monotonic Enforcement**
```python
# Ensure higher outlet → higher indoor temperature
def ensure_monotonic(predictions, outlet_temps):
    # Correct any violations of thermodynamic principles
```

**Stage 3: Prediction Smoothing**
```python
# Exponential moving average prevents erratic changes
smoothed_temp = SMOOTHING_ALPHA * raw_temp + (1 - SMOOTHING_ALPHA) * previous_temp
```

**Stage 4: Dynamic Boost**
```python
# React to current temperature error
error = target_indoor - current_indoor
if error > 0.5:  # Room too cold
    boost = min(5.0, error * boost_factor)
    final_temp += boost
```

**Stage 5: Smart Rounding**
```python
# Heat pumps need integer temperatures - test both options
floor_temp = int(suggested_temp)
ceil_temp = floor_temp + 1
# Test both and choose the one closest to target
```

**Stage 6: Gradual Control**
```python
# Limit maximum change per cycle
max_change = MAX_TEMP_CHANGE_PER_CYCLE
delta = final_temp - last_outlet_temp
if abs(delta) > max_change:
    final_temp = last_outlet_temp + sign(delta) * max_change
```

### Blocking and Grace Period Handling

**Blocking Detection**:
```python
blocking_entities = [DHW_STATUS, DEFROST_STATUS, DISINFECTION_STATUS, DHW_BOOST_HEATER]
blocking_reasons = [e for e in blocking_entities if ha_client.get_state(e, is_binary=True)]
is_blocking = bool(blocking_reasons)
```

**Grace Period Logic**:
```python
# After blocking ends, restore target and wait for stabilization
if last_is_blocking and not is_blocking:
    grace_target = determine_grace_target(last_final_temp, current_outlet, blocking_reasons)
    set_target_temperature(grace_target)
    wait_for_temperature_stabilization(grace_target, wait_condition)
```

**DHW vs Defrost Handling**:
- **DHW heating**: Outlet gets hot → cool-down target with aggressive restoration
- **Defrost cycle**: Outlet gets cold → exact restoration target for full recovery

### Error Handling and Resilience

**Network Error Recovery**:
```python
try:
    all_states = ha_client.get_all_states()
except Exception:
    log_network_error()
    publish_error_state(code=3)  # NETWORK_ERROR
    wait_and_retry()
```

**Missing Sensor Handling**:
```python
critical_sensors = {
    'target_indoor': target_indoor_temp,
    'actual_indoor': actual_indoor_temp,
    'outdoor': outdoor_temp,
    'outlet': actual_outlet_temp
}
missing = [name for name, value in critical_sensors.items() if value is None]
if missing:
    publish_error_state(code=4, missing_sensors=missing)  # NO_DATA
```

**Model Error Recovery**:
```python
try:
    prediction = model.predict_one(features)
except Exception as e:
    log_model_error(e)
    publish_error_state(code=7, last_error=str(e))  # MODEL_ERROR
    # Fallback to last known good temperature
```

## System State Management

### State Codes and Diagnostics

**ML State Sensor** (`sensor.ml_heating_state`):
- **Code 0**: OK - Prediction completed successfully
- **Code 1**: LOW_CONFIDENCE - Model uncertainty high
- **Code 2**: BLOCKED - DHW/defrost/disinfection active
- **Code 3**: NETWORK_ERROR - HA communication failed
- **Code 4**: NO_DATA - Missing critical sensors
- **Code 5**: TRAINING - Initial calibration running
- **Code 6**: HEATING_OFF - Climate not in heat/auto mode
- **Code 7**: MODEL_ERROR - Exception during prediction

### Performance Monitoring

**Real-time Metrics**:
- **Confidence**: `1.0 / (1.0 + sigma)` where sigma is prediction uncertainty
- **MAE**: Mean Absolute Error between predictions and actual outcomes
- **RMSE**: Root Mean Square Error for large error penalty
- **Shadow Metrics**: Comparison between ML and heat curve performance

**Learning Progress Tracking**:
- Training cycle count and learning milestones
- Sample counts for external heat sources
- Multi-lag feature activation status
- Seasonal adaptation readiness

This architecture provides a robust, safe, and continuously improving heating control system that combines physics knowledge with machine learning adaptation while maintaining comprehensive monitoring and safety mechanisms.
