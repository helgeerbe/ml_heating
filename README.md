# ML Heating Control for Home Assistant

> **Warning**
> This project is an initial test and proof of concept heating controller. However, heating systems are critical infrastructure - always monitor its behavior and ensure you have safety mechanisms in place. Use at your own risk.

This project implements a **physics-based machine learning heating control system** that integrates with Home Assistant. It uses a RealisticPhysicsModel to predict the optimal water outlet temperature for a heat pump, continuously learning from real-world results to efficiently maintain your target indoor temperature.

## Overview

Traditional heating curves are static: they map outdoor temperature to a fixed outlet temperature. This system is **dynamic and adaptive** - it learns your house's unique thermal characteristics and continuously improves its predictions based on what actually happens.

## Goal

The primary goal is to improve upon traditional heat curves by creating a **self-learning system** that:

-   **Increases Efficiency:** Minimizes energy waste by precisely matching heating output to actual need
-   **Improves Comfort:** Maintains stable indoor temperature with less overshoot/undershoot
-   **Adapts Continuously:** Learns from every cycle, adjusting to seasons, weather patterns, house changes, and occupancy
-   **Anticipates Conditions:** Uses weather and PV forecasts to proactively adjust heating
-   **Operates Safely:** Includes comprehensive blocking logic, gradual temperature changes, and health monitoring

## Key Features

### Core Capabilities

-   **Physics-Based Machine Learning:** Uses RealisticPhysicsModel that understands thermodynamic principles while learning your house's unique characteristics from real data
-   **Online Learning:** Continuously adapts after every heating cycle - learns from what actually happened vs what was predicted
-   **Live Performance Tracking:** 🆕 **Real-time confidence and accuracy monitoring that adapts to actual prediction performance**
-   **Multi-Lag Learning:** Captures time-delayed effects of heat sources (PV: 4 lags up to 120min, Fireplace: 4 lags up to 90min, TV: 2 lags up to 30min)
-   **Seasonal Adaptation:** Automatic cos/sin modulation learns seasonal variations (±30-50%) without manual recalibration
-   **Summer Learning:** Collects clean baseline data during HVAC-off periods for improved seasonal accuracy
-   **Active & Shadow Modes:** Can run in active mode (controlling heating) or shadow mode (calculating but not applying, for safe testing and comparison)
-   **Home Assistant Integration:** Seamless bi-directional integration - reads sensors, writes control temperatures, publishes metrics and diagnostics
-   **InfluxDB Historical Data:** Leverages your existing Home Assistant/InfluxDB setup for initial calibration and historical feature engineering

### Intelligent Control

-   **Smart Rounding:** Tests both floor and ceiling temperatures, predicts outcomes, chooses the one that gets closest to target
-   **Prediction Smoothing:** Exponential moving average prevents erratic temperature jumps
-   **Dynamic Boost:** Reacts to current temperature error to accelerate correction when needed
-   **Gradual Temperature Control:** Limits maximum temperature change per cycle to protect heat pump from abrupt setpoint jumps
-   **Monotonic Enforcement:** Ensures predictions respect physical reality (higher outlet temp → higher indoor temp)
### Safety & Robustness

-   **Blocking Event Handling:** Automatically pauses and waits during DHW heating, defrosting, disinfection, and DHW boost heater operation
-   **Grace Period after Blocking:** After blocking events end, intelligently waits for outlet temperature to stabilize before resuming ML control:
    
    - **After DHW heating** (outlet hotter): Sets aggressive cool-down target (`last_target + MAX_TEMP_CHANGE_PER_CYCLE`) and waits for outlet to cool, speeding recovery
    - **After defrost** (outlet colder): Restores exact pre-defrost target and waits for outlet to fully recover, preventing sawtooth patterns and maintaining heat pump efficiency
    
    - Intelligently determines whether to wait for cooling or warming based on measured outlet vs target
    - Enforces maximum timeout (`GRACE_PERIOD_MAX_MINUTES`) to prevent indefinite stalling
    - Skips one cycle after grace completes to allow system to fully stabilize
    - Prevents inefficient temperature oscillations and protects model learning quality

-   **Heating Status Check:** Skips prediction and learning when heating system is not in 'heat' or 'auto' mode
-   **Absolute Temperature Clamping:** Enforces safe minimum/maximum outlet temperatures (`CLAMP_MIN_ABS`, `CLAMP_MAX_ABS`)
-   **Confidence Monitoring:** Tracks model confidence and publishes it to Home Assistant for monitoring and automation

### External Heat Sources

-   **Fireplace Mode:** When fireplace is active, uses alternative temperature sensor (e.g., average of other rooms) to prevent incorrect learning
-   **PV Solar Warming:** Learns how solar power generation affects indoor temperature
-   **TV/Electronics Heat:** Can track heat contribution from electronics and appliances

### Live Performance Tracking 🆕

-   **Real-time Confidence Calculation:** Dynamic confidence based on rolling window of recent prediction accuracy (50-sample history)
-   **Adaptive Uncertainty Bounds:** Sigma automatically adjusts between 0.02°C and 0.5°C based on actual performance
-   **Live MAE/RMSE Updates:** Performance metrics recalculate every control cycle instead of being static from calibration
-   **Prediction Error Attribution:** Tracks actual vs predicted temperature changes for transparent performance monitoring
-   **Dynamic Performance Diagnostics:** ML state sensor includes real-time sigma and confidence that reflect current accuracy

### Monitoring & Diagnostics

-   **ML State Sensor:** Comprehensive sensor (`sensor.ml_heating_state`) with numeric state codes and detailed attributes for monitoring system health
-   **Performance Metrics:** Real-time MAE (Mean Absolute Error) and RMSE (Root Mean Square Error) tracking **🆕 Now updates every cycle**
-   **Shadow Mode Metrics:** When in shadow mode, tracks separate metrics for ML vs heat curve performance comparison
-   **Feature Importance Export:** Exports feature importance to InfluxDB for visualization and analysis
-   **Learning Metrics Export:** Exports 43 learning parameters (core, multi-lag, seasonal, sample counts) to InfluxDB every cycle, **including historical timestamps during calibration for detailed analysis**
-   **Learning Dashboard:** Interactive Jupyter notebook (`00_learning_dashboard.ipynb`) for at-a-glance learning status
-   **Detailed Logging:** Comprehensive logging of decisions, predictions, and learning progress

### Deployment

-   **Systemd Service:** Production-ready systemd service configuration for reliable background operation
-   **Automatic Restart:** Configured to restart on failure with 5-minute delay
-   **Jupyter Notebooks:** Six analysis notebooks included for learning dashboard, model diagnosis, performance monitoring, behavior analysis, validation, and multi-lag/seasonal analysis

## Contributing

Contributions are welcome! Areas for improvement:
- Additional external heat source integrations
- Alternative prediction algorithms
- Enhanced forecasting integration
- Grafana dashboard templates
- Documentation improvements

Please open an issue first to discuss major changes.

## License

See LICENSE file for details.

## Acknowledgments

This project builds on thermodynamic principles and machine learning techniques to create a practical, production-ready heating controller. Special thanks to the Home Assistant and InfluxDB communities for their excellent integration capabilities.

---

## Quick Reference

### File Structure
```
ml_heating/
├── src/
│   ├── main.py              # Main control loop
│   ├── physics_model.py     # RealisticPhysicsModel
│   ├── model_wrapper.py     # Optimization & pipeline
│   ├── physics_features.py  # Feature engineering
│   ├── physics_calibration.py # Historical training
│   ├── ha_client.py         # Home Assistant integration
│   ├── influx_service.py    # InfluxDB queries
│   ├── state_manager.py     # State persistence
│   └── config.py            # Configuration
├── notebooks/               # Analysis notebooks
│   ├── 00_learning_dashboard.ipynb
│   ├── 01_physics_model_diagnosis.ipynb
│   ├── 02_performance_monitoring.ipynb
│   ├── 03_behavior_analysis.ipynb
│   ├── 04_model_validation.ipynb
│   └── 05_multilag_seasonal_analysis.ipynb
├── tests/                   # Unit tests
├── .env                     # Your configuration (not in git)
├── .env_sample              # Configuration template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Active Mode** | ML controls heating directly |
| **Shadow Mode** | ML observes but doesn't control |
| **Online Learning** | Model learns after every cycle |
| **Physics Model** | Thermodynamics + ML parameters |
| **Prediction Pipeline** | 7-stage optimization & safety |
| **Monotonic Enforcement** | Ensures physical plausibility |
| **Smart Rounding** | Tests floor/ceiling, picks best |
| **Gradual Control** | Limits temperature change rate |
| **Grace Period** | Stabilization after blocking |
| **Blocking Events** | DHW, defrost, etc. pause control |
| **Feature Importance** | What factors matter most |
| **Shadow Metrics** | Compare ML vs heat curve |

### Typical Good Performance

| Metric | Good Value | Notes |
|--------|-----------|-------|
| Confidence | > 0.9 | Higher is better, **🆕 now updates every cycle** |
| MAE | < 0.2°C | Lower is better, **🆕 now live-tracked** |
| RMSE | < 0.3°C | Lower is better, **🆕 now live-tracked** |
| Real-time Sigma | < 0.1°C | **🆕 New metric: recent prediction accuracy** |
| State Code | 0 (OK) | Most of the time |

### 🎯 November 2025 Production Results
**Verified excellent performance with new live tracking system:**
- **Live MAE**: 0.141°C (excellent accuracy)
- **Live RMSE**: 0.180°C (outstanding precision)
- **Dynamic Confidence**: 0.990 (99% - exceptional)
- **Real-time Sigma**: ~0.02°C (near-perfect recent predictions)

### Emergency: Revert to Heat Curve

If something goes wrong:

```bash
# Stop ML heating
sudo systemctl stop ml_heating.service

# Re-enable your original heat curve automation in Home Assistant

# Or, quickly switch to shadow mode in .env:
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1
sudo systemctl restart ml_heating.service
```

The model continues learning in shadow mode, so you can analyze what went wrong and switch back when ready.

## How It Works

### Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Home Assistant │◄────►│  ML Heating      │◄────►│   InfluxDB      │
│                 │      │  Controller      │      │                 │
│ - Sensors       │      │                  │      │ - Historical    │
│ - Controls      │      │ - Physics Model  │      │   Data          │
│ - Metrics       │◄──── │ - Online Learning│      │ - Features      │
│ - 🆕 Live Perf  │      │ - 🆕 Live Track  │      └─────────────────┘
└─────────────────┘      │ - Optimization   │      
                         └──────────────────┘      
                                │ 🆕 Real-time Performance
                                ▼ Confidence & Accuracy Updates
                         ┌──────────────────┐      
                         │ Rolling Window   │      
                         │ Error Tracking   │      
                         │ (50 samples)     │      
                         └──────────────────┘      
```

### The Physics Model

The controller uses **RealisticPhysicsModel** - a physics-based machine learning approach that combines thermodynamic principles with data-driven learning:

**Core Physics Parameters (learned from data):**
- Base heating rate: How effectively the heat pump heats the house
- Target temperature influence: How target setting affects heating dynamics
- Outdoor temperature factor: Impact of outdoor conditions on heat loss

**External Heat Sources (automatically calibrated):**
- PV solar warming: Heat gain from solar power generation (with 4-lag time delays: 30, 60, 90, 120 min)
- Fireplace heating rate: Contribution from secondary heat sources (with 4-lag time delays: 0, 30, 60, 90 min)
- TV/electronics heat: Heat from appliances and devices (with 2-lag time delays: 0, 30 min)

**Enhanced Learning Features:**
- Multi-lag learning: Captures time-delayed thermal mass effects (activates at 200+ cycles)
- Seasonal adaptation: Cos/sin modulation for automatic seasonal variation (±30-50% ranges)
- Summer learning: Collects HVAC-off baseline data for clean seasonal learning (needs 100+ samples)
- Backward compatible: Old models auto-initialize new features on first use

**System States & Blocking:**
- DHW heating, defrosting, disinfection, DHW boost heater
- Automatically detected and handled with appropriate wait periods

**Forecast Integration:**
- 4-hour weather forecast: Anticipates temperature changes
- 4-hour PV forecast: Predicts solar heat gain
- Enables proactive rather than reactive control

### Model Calibration & Learning

**Initial Calibration (Recommended):**
```bash
python3 -m src.main --calibrate-physics
```
- Trains model on historical data from InfluxDB
- Duration: `TRAINING_LOOKBACK_HOURS` (default: 168 hours / 7 days)
- Learns your house's unique thermal characteristics
- Calibrates external heat source effects

**Continuous Online Learning:**
After calibration, the model learns from every heating cycle:
1. Sets outlet temperature based on current prediction
2. Waits one cycle (`CYCLE_INTERVAL_MINUTES`)
3. Measures actual indoor temperature change
4. Reads what temperature was actually applied (supports shadow mode)
5. Learns from the difference between prediction and reality
6. Updates model parameters to improve future predictions

This happens automatically in both **active mode** (ML controls heating) and **shadow mode** (heat curve controls, ML learns).

### How Learning Works: Feature Contribution Attribution

The physics model learns which features contribute how much through a **multi-level learning system**:

**Level 1: Core Physics Parameters (Every 50 Cycles)**
The model tracks prediction errors and adapts fundamental physics:

```python
error = actual_temperature_change - predicted_temperature_change

# Adapt base heating rate
if error is significant:
    base_heating_rate += error * learning_rate
    
# Adapt target temperature influence
if error is large:
    target_influence += error * learning_rate * 0.5
```

Example: If the model consistently under-predicts temperature rise, it increases `base_heating_rate`, making future predictions higher for the same outlet temperature.

**Level 2: External Heat Source Effects (Tracked Every Cycle)**
The model tracks when external heat sources are active and correlates them with temperature changes.

**Basic tracking** (before cycle 200):
```python
# When PV is active
if pv_power > 100W:
    track: (pv_power, actual_temperature_change)
    
# When fireplace is active  
if fireplace_on:
    track: (fireplace_status, actual_temperature_change)
    
# When TV is on
if tv_on:
    track: (tv_status, actual_temperature_change)
```

**Multi-lag tracking** (after cycle 200):
```python
# Maintain history buffers (ring buffers)
pv_history = [pv_30min_ago, pv_60min_ago, pv_90min_ago, pv_120min_ago]
fireplace_history = [fp_now, fp_30min_ago, fp_60min_ago, fp_90min_ago]
tv_history = [tv_now, tv_30min_ago]

# Track detailed effects with lags
if sufficient_data:
    track: (pv_history, fireplace_history, tv_history, actual_temp_change)
```

**Seasonal tracking** (during HVAC-off periods):
```python
# When heating is off (summer)
if not heating_active:
    month = current_month
    track: (month, external_sources, temp_change)  # Clean baseline without heating interference
```

**Level 3: Effect Correlation Learning (Every 50 Cycles)**
After collecting sufficient data, the model calculates correlations:

```python
# Learn PV contribution
if enough_pv_samples:
    average_pv_power = mean(all_tracked_pv_powers)
    average_temp_change = mean(all_tracked_temp_changes_with_pv)
    
    # Calculate warming per 100W
    pv_warming_coefficient = (average_temp_change / average_pv_power) * 100
    
    # Blend with current estimate (20% new, 80% old for stability)
    pv_coefficient = 0.8 * old_coefficient + 0.2 * learned_coefficient
```

**How Each Feature Contributes:**

1. **Outlet Temperature** (Primary Control Variable)
   - Contribution: `outlet_temp * base_heating_rate`
   - Learned via: Core parameter adaptation
   - Physics: Higher outlet = more heat transfer

2. **Target Temperature** (Setpoint Influence)
   - Contribution: `(target - indoor) * target_influence`
   - Learned via: Core parameter adaptation
   - Physics: Larger gap = more aggressive heating needed

3. **Outdoor Temperature** (Heat Loss Factor)
   - Contribution: `outdoor_penalty * outdoor_factor`
   - Built-in physics: Colder outside = more heat loss
   - Where: `outdoor_penalty = max(0, 10 - outdoor_temp) / 15`

4. **PV Solar Power** (External Heat Gain)
   - Contribution: `pv_now * pv_warming_coefficient * 0.01`
   - Learned via: Effect tracking and correlation
   - Updated: Every 50 cycles with sufficient data (20+ samples)

5. **Fireplace** (Secondary Heat Source)
   - Contribution: `fireplace_on * fireplace_heating_rate`
   - Learned via: Effect tracking during fireplace active periods
   - Updated: Every 50 cycles with sufficient data (10+ samples)

6. **TV/Electronics** (Minor Heat Source)
   - Contribution: `tv_on * tv_heat_contribution`
   - Learned via: Effect tracking when TV active
   - Updated: Every 50 cycles with sufficient data (10+ samples)

7. **Weather Forecast** (Anticipatory Adjustment)
   - Contribution: Calculated from 4-hour temperature forecast
   - Built-in logic: Reduce heating if warming expected
   - Time-decayed: Closer hours have more influence

8. **PV Forecast** (Solar Heat Anticipation)
   - Contribution: Calculated from 4-hour PV forecast
   - Built-in logic: Reduce heating if solar gain expected
   - Time-decayed: Closer hours have more influence

**Example Learning Session:**

```
Cycle 1-199: Collecting data (simple coefficients)
  PV effects tracked: [(500W, +0.15°C), (800W, +0.22°C), ...]
  Fireplace effects: [(1, +0.45°C), (1, +0.51°C), ...]
  Prediction errors: [-0.03°C, +0.05°C, +0.02°C, ...]

Cycle 200: Multi-lag activation
  History buffers filled, switching to multi-lag learning
  PV: 4 lag coefficients initialized
  Fireplace: 4 lag coefficients initialized
  TV: 2 lag coefficients initialized
  
Cycle 250: Learning update
  Core parameters:
    base_heating_rate: 0.002 → 0.00205 (increased due to under-prediction)
    target_influence: 0.01 → 0.0102
    
  External effects (simple):
    PV: 20+ samples collected
    avg_pv_power = 650W, avg_change = +0.18°C
    learned_coefficient = (0.18/650)*100 = 0.0277
    pv_warming_coefficient: 0.015 → 0.0178 (80% old + 20% new)
    
  Multi-lag effects:
    PV lags: [0.008, 0.012, 0.015, 0.010] → total: 0.045°C/100W
    Fireplace lags: [0.15, 0.12, 0.08, 0.05] → total: 0.40°C/hr
    TV lags: [0.015, 0.010] → total: 0.025°C
    
  HVAC-off samples: 45/100 (seasonal learning pending)
  
  Log: "Adapted: heating_rate=0.00205, target_influence=0.0102"
  Log: "Learned effects: PV=0.0178, fireplace=0.126, TV=0.005"
  Log: "Multi-lag active: PV total=0.045, FP total=0.40, TV total=0.025"

Cycle 500: Seasonal adaptation activation
  HVAC-off samples: 102/100 ✓
  Seasonal coefficients learned:
    PV: cos=0.15, sin=-0.20 (modulates 0.5x-1.3x through year)
    TV: cos=0.08, sin=-0.10 (modulates 0.7x-1.2x through year)
  
  Log: "Seasonal adaptation active"
```

**Key Learning Characteristics:**

- **Gradual Adaptation:** Uses weighted averaging (70-80% old, 20-30% new) to prevent sudden changes
- **Bounded Learning:** All parameters have min/max limits to prevent runaway values
- **Frequency Control:** Core parameters adapt every 50 cycles, external effects every 50 cycles
- **Data Requirements:** External effects need minimum samples (10-20) before learning
- **Error Threshold:** Core parameters only adapt if error > 0.03°C (significant)
- **Physics Preservation:** Learning adjusts coefficients but respects thermodynamic relationships

This multi-level approach ensures:
- **Stability:** No wild swings from single outlier measurements
- **Accuracy:** Continuous refinement based on real outcomes
- **Transparency:** Each parameter has clear physical meaning
- **Robustness:** Bounded values prevent model degradation

### The Prediction Pipeline

The final outlet temperature goes through a sophisticated multi-stage pipeline:

```
Raw Prediction → Monotonic Enforcement → Smoothing → Dynamic Boost → 
Smart Rounding → Gradual Control → Final Temperature
```

**Stage 1: Optimization Search**
- Searches range `CLAMP_MIN_ABS` to `CLAMP_MAX_ABS` in 0.5°C steps
- For each candidate, predicts resulting indoor temperature
- Finds temperature that gets closest to target
- If tie, chooses lowest temperature (energy efficiency priority)

**Stage 2: Monotonic Enforcement**
- Ensures physical plausibility: higher outlet → higher indoor
- Anchors to last actual outlet temperature for reliability
- Corrects any non-monotonic predictions from raw model

**Stage 3: Prediction Smoothing**
- Applies Exponential Moving Average (EMA)
- Formula: `new = α * raw + (1-α) * previous`
- Configured via `SMOOTHING_ALPHA` (default: 0.8)
- Prevents erratic jumps while staying responsive

**Stage 4: Dynamic Boost**
- Applies correction based on current temperature error
- If room too cold: boost up (capped at +5°C)
- If room too warm: reduce (capped at -5°C)
- Disabled when outdoor temp > 15°C (solar gain sufficient)

**Stage 5: Absolute Clamping**
- Enforces safety limits: `CLAMP_MIN_ABS` to `CLAMP_MAX_ABS`
- Default: 14°C to 65°C
- Prevents dangerous temperatures

**Stage 6: Smart Rounding**
- Heat pumps need integer temperatures
- Tests both floor and ceiling (e.g., 35°C and 36°C for 35.7°C)
- Predicts indoor outcome for each
- Chooses integer that results in temperature closest to target

**Stage 7: Gradual Temperature Control**
- Final safety: limits change per cycle to `MAX_TEMP_CHANGE_PER_CYCLE`
- Example: if max change = 2°C, outlet at 30°C, target 38°C → sets 32°C
- Baseline selection:
  - **Normal:** Uses last persisted target (`last_final_temp`)
  - **Soft-start (DHW-like):** Uses current measured outlet temp for gentle ramp
  - **Fallback:** Uses measured outlet if no history available
- Protects heat pump from large, inefficient jumps

### Active vs Shadow Mode

The system supports two operating modes:

**Active Mode:**
- ML directly controls heating by writing to `TARGET_OUTLET_TEMP_ENTITY_ID`
- Model learns from its own decisions
- MAE/RMSE track actual ML performance
- Configuration: Set both entity IDs to the same value

**Shadow Mode:**
- Heat curve controls heating (writes to one entity)
- ML calculates but doesn't apply (writes to different entity)
- Model learns from heat curve's decisions
- Separate shadow metrics compare ML vs heat curve:
  - `shadow_ml_mae/rmse`: What error ML would make
  - `shadow_hc_mae/rmse`: What error heat curve actually makes
- Use shadow mode to:
  - Safely test ML before going active
  - Compare performance quantitatively
  - Continue learning while heat curve controls
  - Decide when ML is ready to take over

**Switching Modes:**
Simply change entity IDs in `.env` and restart:
```bash
# Shadow mode (different entities)
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1

# Active mode (same entity)
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
```

Switch to active mode when: `shadow_ml_mae < shadow_hc_mae` AND `shadow_ml_rmse < shadow_hc_rmse`

### Performance Metrics

**Confidence:** 🆕 **Now Dynamic and Real-time**
- Normalized 0 to 1 scale (1.0 = perfect confidence)
- Formula: `confidence = 1.0 / (1.0 + sigma)`
- **Real-time Sigma Calculation:** Based on rolling 50-sample window of actual prediction errors
- **Adaptive Bounds:** Sigma automatically constrained between 0.02°C and 0.5°C
- **Live Updates:** Recalculated every control cycle based on recent accuracy
- Published to Home Assistant for monitoring

**MAE (Mean Absolute Error):** 🆕 **Now Live-Updated**
- Average absolute difference between predicted and actual temperature change
- In °C - lower is better
- **Real-time Tracking:** Updates every control cycle instead of being static from calibration
- Typical good performance: < 0.2°C

**RMSE (Root Mean Square Error):** 🆕 **Now Live-Updated**
- Similar to MAE but penalizes large errors more
- In °C - lower is better
- **Real-time Tracking:** Updates every control cycle alongside MAE
- Typical good performance: < 0.3°C

**Shadow Metrics (Shadow Mode Only):**
- `shadow_ml_mae/rmse`: ML's hypothetical performance
- `shadow_hc_mae/rmse`: Heat curve's actual performance
- Enables quantitative comparison

### ML State Sensor

The system publishes `sensor.ml_heating_state` with numeric state codes and rich diagnostic attributes:

**State Codes:**
| Code | Name | Meaning |
|------|------|---------|
| 0 | OK | Prediction completed successfully |
| 1 | LOW_CONFIDENCE | Model confidence below threshold - logged for monitoring |
| 2 | BLOCKED | Blocking activity active (DHW/defrost/etc) - cycle skipped |
| 3 | NETWORK_ERROR | Failed to communicate with Home Assistant |
| 4 | NO_DATA | Missing critical sensors or insufficient history |
| 5 | TRAINING | Running initial calibration |
| 6 | HEATING_OFF | Heating not in 'heat' or 'auto' mode - cycle skipped |
| 7 | MODEL_ERROR | Exception during prediction or learning |

**Key Attributes:**
- `state_description`: Human-readable status
- `confidence`, `sigma`: Model confidence metrics
- `mae`, `rmse`: Performance metrics
- `suggested_temp`, `final_temp`, `predicted_indoor`: Latest values
- `blocking_reasons`: List of active blocking events
- `missing_sensors`: List of unavailable critical sensors
- `last_prediction_time`, `last_updated`: Timestamps
- `last_error`: Error message for troubleshooting

**Automation Examples:**
```yaml
# Alert on network errors
- trigger:
    platform: state
    entity_id: sensor.ml_heating_state
    to: "3"
  action:
    service: notify.mobile_app
    data:
      message: "ML Heating: Network error detected"

# Track low confidence occurrences
- trigger:
    platform: state
    entity_id: sensor.ml_heating_state
    to: "1"
  action:
    service: counter.increment
    entity_id: counter.ml_low_confidence_count
```

## Analysis & Debugging

### Jupyter Notebooks

Six analysis notebooks are included in `notebooks/` with comprehensive interpretation guides:

**00_learning_dashboard.ipynb** ⭐ Start here!
- At-a-glance learning status and progress
- Core heating parameters visualization with interpretation
- Multi-lag coefficient breakdown (PV, fireplace, TV time delays)
- Seasonal modulation curves (monthly variation patterns)
- Learning milestones and activation status
- Includes detailed "How to interpret" sections for all charts

**01_physics_model_diagnosis.ipynb**
- Deep dive into physics model structure
- Test physics compliance (monotonicity)
- Verify model predictions
- Production readiness assessment

**02_performance_monitoring.ipynb**
- Track model performance over time
- Detect performance degradation
- Alert thresholds and trend analysis
- 30-day performance charts

**03_behavior_analysis.ipynb**
- Analyze heating patterns across seasons
- External heat source impact analysis
- Energy efficiency optimization insights
- Seasonal behavior comparisons

**04_model_validation.ipynb**
- Comprehensive validation checks
- Physics validation across multiple scenarios
- Range validation and bounds checking
- Production deployment decision criteria

**05_multilag_seasonal_analysis.ipynb**
- Detailed multi-lag coefficient analysis
- Time-delayed effect visualization (30-120min lags)
- Seasonal modulation pattern analysis
- Correlation analysis and statistical validation
- Summer learning effectiveness tracking

All notebooks now include **"How to interpret"** sections explaining:
- What each value means
- Typical/good ranges to expect  
- Real calculation examples
- When to take action
- Activation requirements

To use notebooks:
```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  EACH CYCLE (every CYCLE_INTERVAL_MINUTES)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CHECK BLOCKING EVENTS                                    │
│     - DHW heating, defrost, disinfection, boost heater      │
│     - If blocked: skip cycle, persist state                 │
│     - If just unblocked: enter grace period                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. CHECK HEATING STATUS                                     │
│     - Read climate entity state                             │
│     - If not 'heat' or 'auto': skip cycle (state=6)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ONLINE LEARNING (from previous cycle)                   │
│     - Read actual applied temp from previous cycle          │
│     - Measure actual indoor temperature change              │
│     - Compare to predicted change                           │
│     - Call model.learn_one() to update                      │
│     - If shadow mode: track ML vs heat curve errors         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. FETCH CURRENT DATA                                       │
│     Home Assistant: All current sensor values               │
│     InfluxDB: Historical data for feature engineering       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. BUILD FEATURES                                           │
│     - Current temps (indoor, outdoor, outlet)               │
│     - Historical temps (lags, deltas, gradients)            │
│     - Time features (hour, month, day-of-week)              │
│     - External sources (PV, fireplace, TV)                  │
│     - Forecasts (weather, PV next 4 hours)                  │
│     - System states (blocking, defrost)                     │
│     - Statistical aggregations (mean, std, trends)          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. PREDICTION PIPELINE                                      │
│     ┌─────────────────────────────────────────────────┐    │
│     │ a) Optimization search (0.5°C steps)            │    │
│     │ b) Monotonic enforcement                        │    │
│     │ c) Smoothing (EMA)                              │    │
│     │ d) Dynamic boost                                │    │
│     │ e) Absolute clamping                            │    │
│     │ f) Smart rounding                               │    │
│     │ g) Gradual control (max change limit)          │    │
│     └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. PUBLISH TO HOME ASSISTANT                                │
│     - Target outlet temperature                             │
│     - Predicted indoor temperature                          │
│     - ML state sensor (code + attributes)                   │
│     - Confidence, MAE, RMSE metrics                         │
│     - Feature importances (to InfluxDB)                     │
│     - Shadow metrics (if shadow mode)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  8. SAVE STATE & MODEL                                       │
│     - Persist model to MODEL_FILE                           │
│     - Save state to STATE_FILE                              │
│     - Store features for next cycle's learning              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  9. IDLE & BLOCKING POLL                                     │
│     - Wait remainder of CYCLE_INTERVAL_MINUTES              │
│     - Poll for blocking events every                        │
│       BLOCKING_POLL_INTERVAL_SECONDS                        │
│     - If blocking detected during idle: handle immediately  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                     (repeat cycle)
```

## Troubleshooting

### Common Issues

**State Code 3: Network Error**
- Check Home Assistant is running and accessible
- Verify `HASS_URL` and `HASS_TOKEN` in `.env`
- Check network connectivity
- Review logs: `journalctl -u ml_heating.service -n 50`

**State Code 4: No Data**
- Check all required entity IDs exist in Home Assistant
- Look at `missing_sensors` attribute on `sensor.ml_heating_state`
- Verify entity IDs match exactly (case-sensitive)
- Ensure InfluxDB has sufficient historical data

**State Code 6: Heating Off**
- Heating system not in 'heat' or 'auto' mode
- Check your climate entity
- Normal during summer or when heating manually disabled
- **Live tracking continues:** Confidence and performance metrics still update during observation periods

**State Code 7: Model Error**
- Check `last_error` attribute for details
- May indicate corrupted model file
- Try recalibrating: `--calibrate-physics`
- Check logs for Python exceptions
- **Check real-time metrics:** Confidence and sigma values may indicate model health
**State Code 7: Model Error**
- Check `last_error` attribute for details
- May indicate corrupted model file
- Try recalibrating: `--calibrate-physics`
- Check logs for Python exceptions

**Poor Performance (High MAE/RMSE)** 🆕 **Now with Live Tracking**
- **Monitor real-time confidence:** Watch for confidence drops indicating performance issues
- **Check dynamic sigma:** Values > 0.3°C suggest need for attention
- Recalibrate model with recent data
- Increase `CYCLE_INTERVAL_MINUTES` for better learning signal
- Check if blocking events are properly detected
- Verify all external heat sources configured correctly
- Consider if house characteristics changed (insulation, windows, etc.)
- **Review prediction error history:** Use live tracking to identify patterns

**Degrading Live Performance** 🆕 **New Troubleshooting**
- **Confidence dropping over time:** May indicate sensor drift or environmental changes
- **High real-time sigma (>0.2°C):** Recent predictions becoming less accurate
- **Live MAE/RMSE increasing:** Performance degrading from calibration baseline
- Solution: Recalibrate or investigate environmental/sensor changes

**Erratic Temperature Control**
- Increase `SMOOTHING_ALPHA` for more stability
- Increase `MAX_TEMP_CHANGE_PER_CYCLE` if too conservative
- Check for sensor noise or dropout
- Verify grace period settings

**Model Not Learning**
- Verify `ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID` is correct
- Check logs for "Online learning" messages
- Ensure sufficient time between cycles (30min recommended)
- Look for repeated blocking events preventing learning

### Debug Mode

Enable verbose logging:
```bash
# Temporarily for testing
python3 -m src.main --debug

# Or in service, edit /etc/systemd/system/ml_heating.service
# Add to [Service] section:
Environment="DEBUG=1"
```

Debug mode logs:
- Complete feature vectors
- All prediction candidates
- Monotonic enforcement details
- Smart rounding decisions
- Learning updates

### Getting Help

When reporting issues, include:
1. Relevant log excerpts (`journalctl -u ml_heating.service -n 200`)
2. Your `.env` configuration (redact tokens!)
3. State of `sensor.ml_heating_state` including attributes
4. Recent MAE/RMSE values
5. Whether active or shadow mode
6. Any recent changes to house or system

## Advanced Topics

### Feature Importance & Learning Metrics

The system exports both feature importance and learning parameters to InfluxDB for analysis and visualization.

**Configuration:**
```ini
# In .env
INFLUX_FEATURES_BUCKET=ml_heating_features
```

**InfluxDB Schema:**

*Feature Importance:*
- Bucket: `ml_heating_features`
- Measurement: `feature_importance`
- Fields: Each feature as float (e.g., `temp_forecast_3h`, `indoor_hist_mean`)
- Tags: `exported` (UNIX timestamp)

*Learning Parameters:*
- Bucket: `ml_heating_features`
- Measurement: `learning_parameters`
- Fields: 43 parameters including:
  - Core: `base_heating_rate`, `target_influence`, `outdoor_factor`
  - Simple coefficients: `pv_warming_coefficient`, `fireplace_heating_rate`, `tv_heat_contribution`
  - Multi-lag PV: `pv_lag_1` through `pv_lag_4` (4 coefficients)
  - Multi-lag Fireplace: `fireplace_immediate`, `fireplace_lag_1` through `fireplace_lag_3` (4 coefficients)
  - Multi-lag TV: `tv_immediate`, `tv_lag_1` (2 coefficients)
  - Seasonal: `pv_seasonal_cos`, `pv_seasonal_sin`, `tv_seasonal_cos`, `tv_seasonal_sin`
  - Totals: `pv_total_effect`, `fireplace_total_effect`, `tv_total_effect`
  - Sample counts: `training_count`, `pv_samples`, `fireplace_samples`, `tv_samples`, `hvac_off_samples`
- Tags: `exported` (UNIX timestamp)
- Frequency: Every prediction cycle (default: every 10 minutes)

**Example Flux Queries:**

List all measurements:
```flux
import "influxdata/influxdb/schema"
schema.measurements(bucket: "ml_heating_features")
```

Recent feature importances:
```flux
from(bucket: "ml_heating_features")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "feature_importance")
  |> keep(columns: ["_time", "_field", "_value", "exported"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 50)
```

Top 10 most important features:
```flux
from(bucket: "ml_heating_features")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "feature_importance")
  |> last()
  |> sort(columns: ["_value"], desc: true)
  |> limit(n: 10)
```

**Interpretation:**
- Higher values = more influential on predictions
- Monitor over time to see what matters in different conditions
- Useful for understanding model behavior and debugging

### Seasonal Recalibration

Consider recalibrating when seasons change significantly:

```bash
# Stop service
sudo systemctl stop ml_heating.service

# Backup current model
cp /opt/ml_heating/ml_model.pkl /opt/ml_heating/ml_model.pkl.backup

# Recalibrate
source /opt/ml_heating/.venv/bin/activate
python3 -m src.main --calibrate-physics

# Restart service
sudo systemctl start ml_heating.service
```

When to recalibrate:
- Major seasonal changes (winter ↔ summer)
- After house modifications (insulation, windows)
- After heat pump maintenance
- If performance degrades significantly
- When moving from shadow to active mode

### Custom Feature Engineering

The system uses 19 physics-based features. To understand what's available, see:
- `src/physics_features.py`: Feature construction
- `src/physics_model.py`: How features are used
- Notebooks: Analysis and visualization

Key feature categories:
- Core temperatures (indoor, outdoor, outlet)
- Historical data (lags, deltas, gradients)
- Time cyclical (hour, month, day of week)
- External sources (PV, fireplace, TV)
- Forecasts (weather, PV 4-hour ahead)
- System states (blocking events)
- Statistical (mean, std, trends, quartiles)

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Home Assistant with REST API access
- InfluxDB with Home Assistant data (for initial calibration)
- Systemd (for service deployment)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/helgeerbe/ml_heating.git
cd ml_heating

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the sample configuration and edit with your settings:

```bash
cp .env_sample .env
nano .env
```

#### Critical Configuration Settings

**Home Assistant Connection:**
- `HASS_URL`: Your Home Assistant URL (e.g., `http://homeassistant.local:8123`)
- `HASS_TOKEN`: Long-Lived Access Token from your HA profile

**InfluxDB Connection:**
- `INFLUX_URL`: Your InfluxDB URL with port
- `INFLUX_TOKEN`: API token with read access to HA bucket
- `INFLUX_ORG`: Your InfluxDB organization
- `INFLUX_BUCKET`: Bucket with Home Assistant data

**Model Files:**
- `MODEL_FILE`: Path for trained model (default: `/opt/ml_heating/ml_model.pkl`)
- `STATE_FILE`: Path for application state (default: `/opt/ml_heating/ml_state.pkl`)

**Core Entity IDs (MUST match your HA setup):**
- `TARGET_INDOOR_TEMP_ENTITY_ID`: Desired indoor temperature
- `INDOOR_TEMP_ENTITY_ID`: Actual indoor temperature sensor
- `ACTUAL_OUTLET_TEMP_ENTITY_ID`: Heat pump outlet temperature sensor
- `TARGET_OUTLET_TEMP_ENTITY_ID`: Entity ML writes its calculated temperature to
- `ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID`: Entity that was actually applied (for learning)
  - **Active mode:** Same as `TARGET_OUTLET_TEMP_ENTITY_ID`
  - **Shadow mode:** Different entity (e.g., from your heat curve automation)
- `HEATING_STATUS_ENTITY_ID`: Climate entity to check heating mode

**Blocking Event Entities:**
- `DHW_STATUS_ENTITY_ID`: DHW heating active sensor
- `DEFROST_STATUS_ENTITY_ID`: Defrost cycle active sensor
- `DISINFECTION_STATUS_ENTITY_ID`: DHW disinfection active sensor
- `DHW_BOOST_HEATER_STATUS_ENTITY_ID`: DHW boost heater active sensor

**External Heat Sources:**
- `FIREPLACE_STATUS_ENTITY_ID`: Fireplace active sensor
- `AVG_OTHER_ROOMS_TEMP_ENTITY_ID`: Alternative temperature sensor for fireplace mode
- `TV_STATUS_ENTITY_ID`: TV/electronics heat source indicator
- `PV1_POWER_ENTITY_ID`, `PV2_POWER_ENTITY_ID`, `PV3_POWER_ENTITY_ID`: Solar power sensors
- `PV_FORECAST_ENTITY_ID`: PV forecast sensor with `watts` attribute (15-min samples)

**Timing & Behavior:**
- `CYCLE_INTERVAL_MINUTES`: Time between predictions (default: 30 min)
  - Longer = clearer learning signal
  - Shorter = more responsive
- `MAX_TEMP_CHANGE_PER_CYCLE`: Max outlet temp change per cycle (default: 5°C)
  - Prevents abrupt jumps
  - Example: 5°C with 30min cycle = max 10°C/hour
- `GRACE_PERIOD_MAX_MINUTES`: Max wait after blocking ends (default: 30 min)
- `BLOCKING_POLL_INTERVAL_SECONDS`: How often to check for blocking during idle (default: 60s)

**Model Tuning:**
- `CONFIDENCE_THRESHOLD`: Minimum confidence (default: 0.3)
  - Formula: `threshold = 1.0 / (1.0 + max_tolerated_sigma_in_celsius)`
  - Examples: 1.0°C tolerance → 0.5, 0.5°C tolerance → 0.67
- `SMOOTHING_ALPHA`: EMA smoothing factor (default: 0.8)
  - Lower (0.1) = more smoothing, less responsive
  - Higher (0.9) = less smoothing, more responsive
- `CLAMP_MIN_ABS` / `CLAMP_MAX_ABS`: Absolute temperature limits (default: 14-65°C)

**Training:**
- `TRAINING_LOOKBACK_HOURS`: Historical data for calibration (default: 168 hours / 7 days)

**See `.env_sample` for complete configuration with detailed comments.**

### 3. Initial Calibration (Recommended)

Calibrate the physics model on your historical data:

```bash
source .venv/bin/activate
python3 -m src.main --calibrate-physics
```

This trains the model on `TRAINING_LOOKBACK_HOURS` (default: 168 hours) of historical data from InfluxDB, learning your house's unique thermal characteristics.

**Optional Validation:**
```bash
python3 -m src.main --validate-physics
```
Tests model predictions across temperature ranges without modifying anything.

### 4. Test Run

Start in foreground to verify configuration:

```bash
python3 -m src.main
```

Watch the logs to ensure:
- Connects to Home Assistant successfully
- Reads all required sensors
- Makes predictions
- No errors or warnings

Press Ctrl+C to stop.

### 5. Deploy as Systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/ml_heating.service
```

Add the following content (**update paths for your installation**):

```ini
[Unit]
Description=ML Heating Control Service
After=network.target home-assistant.service influxdb.service
Wants=home-assistant.service influxdb.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/opt/ml_heating
ExecStart=/opt/ml_heating/.venv/bin/python3 -m src.main
Restart=on-failure
RestartSec=5m
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml_heating.service
sudo systemctl start ml_heating.service
```

### 6. Monitor Operation

**Check service status:**
```bash
sudo systemctl status ml_heating.service
```

**View live logs:**
```bash
sudo journalctl -u ml_heating.service -f
```

**View recent logs:**
```bash
sudo journalctl -u ml_heating.service -n 100
```

**Check specific time range:**
```bash
sudo journalctl -u ml_heating.service --since "1 hour ago"
```

### 7. Home Assistant Monitoring

Add to your dashboard:
```yaml
type: entities
entities:
  - entity: sensor.ml_heating_state
    name: ML Heating Status
  - entity: sensor.ml_model_confidence
    name: Model Confidence
  - entity: sensor.ml_model_mae
    name: Mean Absolute Error
  - entity: sensor.ml_model_rmse
    name: Root Mean Square Error
  - entity: sensor.ml_vorlauftemperatur
    name: ML Target Outlet Temp
```

## Usage & Operation

### Command Line Options

**Calibration:**
```bash
python3 -m src.main --calibrate-physics
```
- Trains model on historical data from InfluxDB
- Duration: `TRAINING_LOOKBACK_HOURS` (default: 7 days)
- Then starts normal operation
- Run this when first setting up or after major house changes

**Validation:**
```bash
python3 -m src.main --validate-physics
```
- Tests model predictions across temperature ranges
- Exits without modifying anything
- Useful for verifying model behavior

**Debug Mode:**
```bash
python3 -m src.main --debug
```
- Enables verbose logging
- Includes detailed feature vectors and model decisions
- Warning: Can be very noisy

**Normal Operation:**
```bash
python3 -m src.main
```
- Standard operation mode
- Recommended for production use

### Operating Modes

**Starting in Shadow Mode (Recommended):**
1. Configure different entity IDs in `.env`:
   ```
   TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
   ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1
   ```
2. Keep your existing heat curve automation active
3. ML calculates but doesn't control heating
4. Monitor shadow metrics in logs
5. When `shadow_ml_mae < shadow_hc_mae` and confident, switch to active mode

**Switching to Active Mode:**
1. Update `.env` to use same entity:
   ```
   ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
   ```
2. Disable your heat curve automation
3. Restart ml_heating service:
   ```bash
   sudo systemctl restart ml_heating.service
   ```
4. ML now controls heating directly

**You can switch back and forth as needed** - the model continues learning in both modes.

### Typical Workflow

1. **Week 1: Shadow Mode**
   - Let ML observe and learn
   - Monitor shadow metrics
   - Compare ML vs heat curve performance

2. **Week 2: Active Mode (if metrics good)**
   - ML takes control
   - Monitor confidence and error metrics
   - Check comfort and energy consumption

3. **Ongoing:**
   - Monitor `sensor.ml_heating_state` for issues
   - Check logs occasionally
   - Recalibrate seasonally if needed

### Safety Considerations

- Always monitor initial operation closely
- Set up alerts for state code 3 (network error) and 7 (model error)
- Consider keeping your heat curve automation as a backup
- Monitor confidence levels - typical good operation shows 0.9+
- If performance degrades, recalibrate or switch back to shadow mode
