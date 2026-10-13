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
│ - Notebooks     │      │ - Optimization   │      │ - Analytics     │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

### Notebook Data Access Architecture

The system includes 7 Jupyter notebooks for analysis and monitoring, each requiring access to historical data stored in InfluxDB. A critical design pattern has emerged for reliable data access:

**Proven Data Access Pattern** (Used by notebooks 00-06 and fixed 07):
```python
# CORRECT: Use influx_service.fetch_history() method
influx = influx_service.InfluxService(url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG)
ml_confidence_data = influx.fetch_history('sensor.ml_model_confidence', steps, 0.0, agg_fn='mean')
```

**Anti-Pattern** (Broken approach found in original notebook 07):
```python
# INCORRECT: Direct InfluxDB Flux queries fail in notebook environment
result = influx.query_api.query_data_frame(flux_query)  # Returns empty results
```

**Why the Pattern Works**:
- **Entity ID Handling**: `fetch_history()` automatically strips domain prefixes (`sensor.name` → `name`)
- **Aggregation**: Built-in aggregation functions (`mean`, `last`, etc.) handle data properly
- **Error Handling**: Graceful fallback to default values when data unavailable
- **Consistency**: Same method used across all working notebooks and core system

**Pattern Implementation**:
```python
def get_real_data_correctly():
    try:
        # Calculate data points needed
        steps = int((hours_back * 60) / config.HISTORY_STEP_MINUTES)
        
        # Use proven fetch_history method
        confidence_data = influx.fetch_history('sensor.ml_model_confidence', steps, 0.0, agg_fn='mean')
        temperature_data = influx.fetch_history(config.INDOOR_TEMP_ENTITY_ID, steps, 21.0, agg_fn='mean')
        
        if not any([confidence_data, temperature_data]):
            print("⚠️ No data found - falling back to demo")
            return generate_demo_data()
            
        # Create time-indexed DataFrame
        time_index = pd.date_range(start=start_time, end=end_time, periods=steps)
        return pd.DataFrame({'confidence': confidence_data, 'temperature': temperature_data}, index=time_index)
        
    except Exception as e:
        print(f"Error: {e} - using demo data")
        return generate_demo_data()
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

**Grace Period Logic (Intelligent Recovery)**:
```python
# After blocking ends, use the model to calculate a new target for recovery
if last_is_blocking and not is_blocking:
    # Fetch current state (indoor, target, outdoor temps)
    sensor_data = get_current_sensor_data()

    if sensor_data_is_complete:
        # Use the model to determine the precise outlet temperature needed for recovery
        grace_target = model.calculate_required_outlet_temp(
            sensor_data.current_indoor,
            sensor_data.target_indoor,
            sensor_data.outdoor_temp,
            # ... other features
        )
    else:
        # Fallback to simple restoration if sensor data is missing
        grace_target = state.get("last_final_temp")

    set_target_temperature(grace_target)
    
    # Dynamic Wait Loop:
    # Monitor indoor temperature during the grace period.
    # If it drops significantly, recalculate the target immediately.
    wait_for_temperature_stabilization(grace_target, dynamic_update=True)
```

**Recovery Strategy**:
- **Model-Driven**: Instead of static restoration, the system calculates the optimal temperature to recover from the thermal deficit caused by the blocking event (DHW, defrost).
- **Dynamic Updates**: The system monitors indoor temperature *during* the grace period wait. If the temperature drops further, it recalculates and updates the target immediately, preventing "cold night drift".
- **Resilient**: This makes the system more robust against heat loss and prevents prediction accuracy from dropping.
- **Fallback**: A safe fallback to the previous temperature is retained for sensor data issues.

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

### 7. Adaptive Source Attribution Pattern

**Principle**: Isolate and learn weights for specific heat sources (TV, PV, Fireplace)

**Mechanism**:
- **Selective Learning**: Only update a source's weight when that source is active and dominant.
- **Gradient Descent**: Use prediction error to calculate the gradient for the specific weight.
- **Cross-Validation**: Ensure that improving one weight doesn't degrade overall model performance.

**Implementation**:
```python
# In ThermalEquilibriumModel._adapt_parameters_from_recent_errors
if self.tv_power > 50:  # TV is active
    tv_gradient = self._calculate_tv_heat_weight_gradient(error)
    self.tv_heat_weight -= learning_rate * tv_gradient

if self.pv_power > 100: # PV is active
    pv_gradient = self._calculate_pv_heat_weight_gradient(error)
    self.pv_heat_weight -= learning_rate * pv_gradient
```

**Benefits**:
- Eliminates manual guessing of heat source contributions
- Adapts to device upgrades (e.g., new TV) or degradation (e.g., dirty solar panels)
- Improves prediction accuracy during multi-source events

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

**Stage 3: Heat Balance Controller** 🆕
```python
# Intelligent 3-phase control system replaces simple smoothing
temperature_error = abs(target_temp - current_temp)

if temperature_error > CHARGING_MODE_THRESHOLD:
    mode = "CHARGING"  # Aggressive heating (>0.5°C error)
    outlet_temp = find_outlet_for_target(target_temp)
elif temperature_error > MAINTENANCE_MODE_THRESHOLD:
    mode = "BALANCING"  # Trajectory optimization (0.2-0.5°C error)
    outlet_temp = find_stable_trajectory_outlet()
else:
    mode = "MAINTENANCE"  # Minimal adjustments (<0.2°C error)
    outlet_temp = current_outlet_temp + sign(temperature_error) * 0.5
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

**Grace Period Logic (Intelligent Recovery)**:
```python
# After blocking ends, use the model to calculate a new target for recovery
if last_is_blocking and not is_blocking:
    # Fetch current state (indoor, target, outdoor temps)
    sensor_data = get_current_sensor_data()

    if sensor_data_is_complete:
        # Use the model to determine the precise outlet temperature needed for recovery
        grace_target = model.calculate_required_outlet_temp(
            sensor_data.current_indoor,
            sensor_data.target_indoor,
            sensor_data.outdoor_temp,
            # ... other features
        )
    else:
        # Fallback to simple restoration if sensor data is missing
        grace_target = state.get("last_final_temp")

    set_target_temperature(grace_target)
    wait_for_temperature_stabilization(grace_target, wait_condition)
```

**Recovery Strategy**:
- **Model-Driven**: Instead of static restoration, the system calculates the optimal temperature to recover from the thermal deficit caused by the blocking event (DHW, defrost).
- **Resilient**: This makes the system more robust against heat loss and prevents prediction accuracy from dropping.
- **Fallback**: A safe fallback to the previous temperature is retained for sensor data issues.

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

### Trust Gate & Control Modes (T0.7 + OE-recalibration)

The `sensor.ml_heating_state` **value** answers *"is ML actually commanding the heat pump this cycle?"* (`1`=active, `0`=not). The `ml_control_mode` attribute disambiguates the cases:

| `ml_control_mode` | state | meaning |
|-------------------|-------|---------|
| `"active"` | 1 | trusted by the self-calibration gate; ML drives the outlet |
| `"passthrough"` | 0 | learning / recovering accuracy; heat-curve-anchored outlet |
| `"shadow"` | 0 | shadow mode (heat curve drives); ML observes only |
| `"unknown"` | 0/1 | rare fallback when the trust gate itself could not be evaluated |

**Trust gates** (both must pass for `active`):
- `confidence >= config.CONFIDENCE_THRESHOLD` (code default 0.2; `.env` override 2.0)
- `total_predictions >= MIN_PREDICTIONS_FOR_TRUST` (default 50; `.env` override 5 for post-reset re-engagement). Effective count = `len(prediction_metrics.predictions)`.
- `_is_model_calibrated()` additionally requires `baseline_source == "calibrated"` for the *aggressive* outlet path — this is a separate, stricter cap from the trust gate, so `config`-baseline deployments stay conservatively capped even when the trust threshold is lowered.

**Passthrough safety cap**: the passthrough guard caps the outlet at `config.PASSTHROUGH_MAX_SAFE_OUTLET` (default 50.0 °C) instead of `_MAX_OUTLET_TEMP` (65 °C). This rejects heat-curve feedback-loop contamination (the 53 °C incident) and keeps a hard safety margin between passthrough and the absolute model ceiling.

**Demotion / re-promotion hysteresis** (Oct 13, 2026 death-spiral fix):
- **Demotion floor**: `good_control_pct < AUTO_CAL_GOOD_CONTROL_PCT / DEMOTION_FACTOR` (40% with `DEMOTION_FACTOR=2.0`).
- **Re-promotion threshold**: now the *same scaled* value `AUTO_CAL_GOOD_CONTROL_PCT / DEMOTION_FACTOR` (40%), **not** the unscaled 80%. This makes hysteresis symmetric so a demoted model can actually recover. The previous asymmetry (53.3% floor vs unreachable 80% re-promotion) trapped the model in passthrough forever once a false demotion occurred.
- **Minimum-predictions guard** (`MIN_PREDICTIONS_FOR_DEMOTION`, default 12): `_check_accuracy_demotion()` skips the **entire** gate (resets `_demotion_streak`, returns) when `prediction_metrics.get_24h_prediction_count() < MIN_PREDICTIONS_FOR_DEMOTION`. This prevents two failure modes: (a) stale passthrough-era predictions re-demoting `accuracy_state` on every restart while the 24h window is sparse, and (b) premature re-promotion from a handful of lucky predictions. Both demotion and re-promotion are suppressed on sparse windows.

### Passive Learning (`heating_active` flag)

When the heat pump is off, the model still learns the building's *passive* dynamics without corrupting control-quality metrics:

- **Passive equilibrium** (`thermal_equilibrium_model.py`): with `heating_active=False`, the radiator/outlet term is dropped entirely — equilibrium reduces to `T_out + (PV + fireplace + TV) / HLC` (no outlet-effectiveness in the denominator). Parameter gradients (`_calculate_parameter_gradient`, `_calculate_equilibrium_based_oe_gradient`) become per-record `heating_active`-aware; the OE gradient is forced to 0 when all recent records are passive.
- **Trust-gate isolation** (`model_wrapper.py`): `learn_from_prediction_feedback` always calls `update_prediction_feedback` (so passive parameters keep adapting) but only calls `prediction_metrics.add_prediction` when `heating_active=True`. Passive observations never count toward `good_control_pct` / accuracy metrics that drive the trust gate.
- **Main loop contract** (`main.py`): reads `was_heating_active` from `last_run_features`; uses passive mode (outlet=0, passive trajectory) when False. The heating-off `continue` path now saves passive state (fresh `last_indoor_temp`, `heating_active=False` in features) so the next cycle has a valid baseline.
- **Backward compatibility**: `heating_active` defaults to `True` everywhere, so no existing caller breaks.

### Outlet-Effectiveness (OE) Gradient Learning

OE self-corrects from two gradient signals, combined in the main OE update:

1. **Trajectory gradient** (`_calculate_outlet_effectiveness_gradient` → `_calculate_parameter_gradient`) — uses the 30-minute prediction error. **Weak** at long time constants: with τ≈29 h and 30-min cycles the approach factor is `1 - exp(-0.5/29) ≈ 0.017`, so a 4 °C equilibrium error yields only ~0.07 °C trajectory error → gradient ~0.0003/cycle → 40+ day recovery. This is the *fine-tuning* signal.
2. **Equilibrium-based gradient** (`_calculate_equilibrium_based_oe_gradient`) — fires when the room is thermally stable (indoor gradient < 0.15 °C/cycle). When stable, the actual indoor temperature *is* the equilibrium, so the method computes the predicted equilibrium directly and uses `actual − predicted equilibrium` as the error. **~50× stronger** than the trajectory gradient; this is the *convergence* signal that lets OE recover in hours/days rather than weeks.

> **Cold-weather damping is symmetric**: all OE updates are damped equally during cold weather (no asymmetric ratchet), preventing the 1.107→0.358 seasonal drift that caused the 53 °C incident. See `docs/BUGFIX_OE_DRIFT_AND_PASSTHROUGH_CAP.md`.

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

## Multi-Add-on Deployment Architecture

### Dual Channel Strategy

**Stable Channel** (`ml_heating`):
- Version tags: `v*` (e.g., `v0.1.0`, `v0.2.0`) - excludes alpha releases
- Add-on slug: `ml_heating`
- Container: `ghcr.io/helgeerbe/ml_heating:{version}`
- Auto-updates: ✅ Enabled for production reliability
- Log level: INFO (production optimized)
- Development API: ❌ Disabled for security

**Alpha Channel** (`ml_heating_dev`):
- Version tags: `v*-alpha.*` (e.g., `v0.1.0-alpha.1`, `v0.1.0-alpha.8`)
- Add-on slug: `ml_heating_dev`
- Container: `ghcr.io/helgeerbe/ml_heating:{alpha-version}`
- Auto-updates: ❌ Disabled (manual updates for safety)
- Log level: DEBUG (detailed diagnostics)
- Development API: ✅ Enabled for Jupyter notebooks

### Smart Unified Workflow Architecture

**Single Smart Workflow** (`.github/workflows/build-ml-heating.yml`):
```yaml
on:
  push:
    tags: ['v*']  # Trigger on ANY version tag
  workflow_dispatch:

jobs:
  detect-addon-type:    # Smart tag detection (alpha vs stable)
  validate:            # HA linter validation for detected addon
  build-addon:         # Copy shared components + build
  release:             # Release with smart detection
```

**Smart Tag Detection**:
- `v*-alpha.*` patterns → Development addon
- `v*` patterns (excluding alpha) → Stable addon
- Dynamic configuration updates during build
- Shared component copying from `ml_heating_addons/shared/`

### Dynamic Version Management

**Alpha Development Process**:
```bash
# Development workflow dynamically updates config during build
yq eval ".version = \"$VERSION\"" -i ml_heating_addons/ml_heating_dev/config.yaml
yq eval ".name = \"ML Heating Control (Alpha $VERSION)\"" -i ml_heating_addons/ml_heating_dev/config.yaml
```

**Stable Release Process**:
```bash
# Stable workflow updates version in production config
sed -i "s/^version: .*/version: \"$VERSION\"/" ml_heating_addons/ml_heating/config.yaml
```

### Multi-Platform Container Building

Both channels support all Home Assistant platforms:
- **linux/amd64** - Standard x86_64 systems
- **linux/aarch64** - Raspberry Pi 4, newer ARM64 systems  
- **linux/arm/v7** - Raspberry Pi 3, older ARM systems

Home Assistant Builder handles all platform compilation automatically.

### Add-on Configuration Differences

**File Structure**:
```
ml_heating_addons/
├── ml_heating/          # Stable channel
│   └── config.yaml      # Production config only
├── ml_heating_dev/      # Alpha channel
│   └── config.yaml      # Development config only
└── shared/              # All shared components
    ├── build.json       # Shared build metadata
    ├── config_adapter.py
    ├── config.yaml      # Base configuration template
    ├── Dockerfile       # Container build instructions
    ├── README.md
    ├── requirements.txt # Python dependencies
    ├── run.sh          # Container startup script
    ├── supervisord.conf
    └── dashboard/       # Advanced web interface
        ├── app.py       # Main Streamlit app
        ├── health.py    # Health check endpoint
        └── components/  # Modular dashboard components
            ├── __init__.py
            ├── backup.py    # Backup/restore system
            ├── control.py   # ML control interface
            ├── overview.py  # System overview
            └── performance.py # Analytics & metrics
```

**Key Configuration Differences**:
- **Version**: Stable uses semantic versions, Dev uses alpha versions
- **Auto-updates**: Stable enabled, Dev disabled  
- **Logging**: Stable INFO, Dev DEBUG
- **Dev API**: Stable disabled, Dev enabled
- **Add-on names**: Include channel identification

### Release Channel Benefits

**Users can**:
- Install both channels simultaneously for A/B testing
- Choose appropriate risk level (stable vs cutting-edge)
- Access latest features without affecting production systems
- Maintain separate configurations for different use cases

**Developers get**:
- Safe testing environment with alpha releases
- Comprehensive CI/CD with multi-platform builds
- Automatic version management and release notes
- Clear separation between experimental and production code

This architecture provides a robust, safe, and continuously improving heating control system that combines physics knowledge with machine learning adaptation while maintaining comprehensive monitoring and safety mechanisms, delivered through professional dual-channel deployment infrastructure.
