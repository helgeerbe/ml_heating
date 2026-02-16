# ML Heating Add-on Installation Guide

Complete step-by-step installation guide for the ML Heating Control Home Assistant add-on.

## Overview

This guide will help you install and configure the ML Heating Control add-on, which transforms your heat pump into an intelligent, self-learning control system that continuously adapts to your home's unique thermal characteristics.

## Prerequisites

### Required Components
- ✅ **Home Assistant OS** or **Home Assistant Supervised**
- ✅ **Heat pump** with controllable outlet temperature setpoint
- ✅ **Indoor temperature sensor** (reliable and representative)
- ✅ **Outdoor temperature sensor** (accurate external temperature)
- ✅ **Outlet temperature sensor** (heat pump water outlet temperature)

### Recommended Components
- ✅ **InfluxDB add-on** (for historical data and advanced analytics)
- ✅ **PV system** with power monitoring (optional but valuable)
- ✅ **Additional heat sources** sensors (fireplace, wood stove, etc.)

### Home Assistant Requirements
- **Version**: 2023.1 or newer
- **System**: OS or Supervised (required for add-ons)
- **Resources**: 2GB+ RAM, adequate storage for ML models

## Step-by-Step Installation

### Step 1: Add the Repository

1. **Open Home Assistant**
   - Navigate to **Settings** → **Add-ons** → **Add-on Store**

2. **Add Repository**
   - Click the **⋮** menu (three dots) in the top right
   - Select **Repositories**
   - Add this URL:
     ```
     https://github.com/helgeerbe/ml_heating
     ```
   - Click **Add**

3. **Refresh Store**
   - Close the repositories dialog
   - Refresh the add-on store page
   - You should see **"ML Heating Control"** in the list

### Step 2: Install the Add-on

1. **Find the Add-on**
   - Search for "ML Heating Control"
   - Click on the add-on card

2. **Install**
   - Click **Install** 
   - Wait for installation to complete (5-10 minutes)
   - Don't start yet - configuration needed first

### Step 3: Prepare Your Entities

Before configuration, ensure all required entities exist in Home Assistant:

#### Required Entities Checklist
- [ ] **Target indoor temperature**: `climate.your_thermostat` or `input_number.target_temp`
- [ ] **Indoor temperature sensor**: `sensor.living_room_temperature`
- [ ] **Outdoor temperature sensor**: `sensor.outdoor_temperature`
- [ ] **Heat pump outlet temperature**: `sensor.heat_pump_outlet_temp`
- [ ] **Heating control entity**: `climate.heating_system` or equivalent
- [ ] **DHW status sensor**: `binary_sensor.dhw_active` (if available)
- [ ] **Defrost status sensor**: `binary_sensor.defrost_active` (if available)

#### Optional Entities
- [ ] **PV power sensors**: `sensor.solar_power_1`, `sensor.solar_power_2`
- [ ] **Fireplace sensor**: `binary_sensor.fireplace_active`
- [ ] **PV forecast**: `sensor.solcast_pv_forecast_forecast_today` (if using Solcast)

**💡 Tip**: Use **Developer Tools** → **States** to verify all entity IDs exist and have current values.

### Step 4: Configure the Add-on

The add-on now includes **entity autocomplete** and **advanced feature configuration**. All core functionality is available directly in the add-on interface with intelligent entity selection.

1. **Open Add-on Configuration**
   - Go to the installed ML Heating Control add-on
   - Click the **Configuration** tab

2. **Entity Configuration with Autocomplete**
   - **Core Entities**: Use autocomplete dropdowns to select temperature sensors, climate entities, etc.
   - **Entity Validation**: Automatic validation ensures entities exist and have correct device classes
   - **Descriptions**: Each field includes detailed descriptions and requirements

3. **Advanced Features (New)**
   - **Multi-lag Learning**: Configure time-delayed learning for PV, fireplace, and TV effects
   - **Seasonal Adaptation**: Enable automatic seasonal learning with customizable parameters
   - **Summer Learning**: Activate HVAC-off period learning for better external source modeling
   - **System Behavior**: Fine-tune grace periods, polling intervals, and prediction horizons

4. **Save Configuration**
   - Click **Save**
   - Configuration validation provides clear feedback
   - All advanced features are now available without manual .env editing

### Step 5: Start the Add-on

1. **Initial Start**
   - Go to the **Info** tab
   - Click **Start**
   - Monitor the logs for any errors

2. **Check Logs**
   ```
   [INFO] ML Heating Add-on Configuration Adapter Starting...
   [INFO] Add-on configuration loaded successfully
   [INFO] Configuration validation passed
   [INFO] Created directory: /data/models
   [INFO] Created directory: /data/backups
   [INFO] Add-on environment initialized successfully
   [INFO] Starting ML Heating system...
   ```

3. **Enable Auto-start**
   - Toggle **"Start on boot"** to enabled
   - Toggle **"Watchdog"** to enabled

### Step 6: Access the Dashboard

1. **Sidebar Integration**
   - The dashboard automatically appears in your Home Assistant sidebar
   - Look for **"ML Heating Control"** panel

2. **Direct Access**
   - URL: `http://your-ha-ip:3001`
   - Should show the 4-page dashboard interface

3. **Verify Dashboard**
   - **Overview**: Current system status
   - **Control**: Start/stop controls
   - **Performance**: Live metrics
   - **Backup**: Model management

## Configuration Reference

This section details all available configuration parameters for the ML Heating Control add-on. These correspond to the settings found in your add-on's configuration tab.

#### Home Assistant Connection

*   **HASS_URL**: Base URL of your Home Assistant instance (e.g., `http://homeassistant.local:8123`).
*   **HASS_TOKEN**: A Long-Lived Access Token from your Home Assistant profile page. This is required for the add-on to communicate with Home Assistant.

#### InfluxDB Connection

*   **INFLUX_URL**: URL of your InfluxDB instance, including the protocol and port (e.g., `https://influxdb.example.com:8086`).
*   **INFLUX_TOKEN**: An InfluxDB API token with read access to the specified bucket where your Home Assistant data is stored.
*   **INFLUX_ORG**: The InfluxDB organization your Home Assistant data belongs to.
*   **INFLUX_BUCKET**: The InfluxDB bucket where your Home Assistant data is stored (e.g., `home_assistant/autogen`).

#### Model and State Storage

*   **UNIFIED_STATE_FILE**: Absolute path where the unified thermal state (including model parameters and learning history) is stored. Default: `/data/unified_thermal_state.json`.
*   **CALIBRATION_BASELINE_FILE**: Path to the file storing the calibrated physics baseline. Default: `/data/calibrated_baseline.json`.

#### Script Behavior and Learning Parameters

*   **HISTORY_STEPS**: Number of historical steps (or past data points) to use as features for the machine learning model.
*   **HISTORY_STEP_MINUTES**: The duration of each historical step in minutes. For example, if `HISTORY_STEPS=6` and `HISTORY_STEP_MINUTES=10`, the model will consider data from the last 60 minutes.
*   **PREDICTION_HORIZON_STEPS**: How many 5-minute steps into the future the model should predict. Example: `24` means `24 * 5min = 120` minutes ahead.
*   **TRAINING_LOOKBACK_HOURS**: The number of hours of historical data to use for the initial training or calibration of the model.
*   **CYCLE_INTERVAL_MINUTES**: The time in minutes between each full cycle of learning and prediction. Default: `10`.
*   **MAX_TEMP_CHANGE_PER_CYCLE**: The maximum allowable integer change (in degrees Celsius) for the heat pump's outlet temperature setpoint in a single cycle. This prevents abrupt temperature changes. Default: `2`.
*   **INFLUX_FEATURES_BUCKET**: InfluxDB bucket for exporting feature importances and learning parameters for analysis.

#### Home Assistant Entity IDs (Critical)

These parameters define which Home Assistant entities the add-on will read from and write to. **They must exactly match the entity IDs in your Home Assistant instance.**

*   **TARGET_INDOOR_TEMP_ENTITY_ID**: An `input_number` or `sensor` that holds the desired target indoor temperature. Example: `input_number.hp_auto_correct_target`.
*   **INDOOR_TEMP_ENTITY_ID**: The primary indoor temperature sensor that the system will try to maintain. Example: `sensor.your_indoor_temp_sensor`.
*   **ACTUAL_OUTLET_TEMP_ENTITY_ID**: The heat pump's actual outlet temperature sensor. Example: `sensor.your_hp_outlet_temp`.
*   **TARGET_OUTLET_TEMP_ENTITY_ID**: The entity this script will write its calculated target outlet temperature to. This should be an entity that controls your heat pump's setpoint. Example: `sensor.ml_vorlauftemperatur`.
*   **ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID**: The target outlet temperature that was *actually* set (either by the ML model or your existing heat curve). This is crucial for the model to learn correctly in both active and shadow modes. Example: `sensor.hp_target_temp_circuit1`.
*   **DHW_STATUS_ENTITY_ID**: A `binary_sensor` that is 'on' when Domestic Hot Water (DHW) is being heated.
*   **DEFROST_STATUS_ENTITY_ID**: A `binary_sensor` that is 'on' during a defrost cycle of the heat pump.
*   **DISINFECTION_STATUS_ENTITY_ID**: A `binary_sensor` that is 'on' during a DHW tank disinfection cycle.
*   **DHW_BOOST_HEATER_STATUS_ENTITY_ID**: A `binary_sensor` that is 'on' when the DHW boost heater is active.
*   **TV_STATUS_ENTITY_ID**: A `sensor` or `input_boolean` that indicates if a TV or other significant internal heat source is on.
*   **FIREPLACE_STATUS_ENTITY_ID**: A `binary_sensor` that is 'on' when the fireplace is active.
*   **AVG_OTHER_ROOMS_TEMP_ENTITY_ID**: An optional `sensor` that provides the average temperature of rooms not affected by a local heat source like a fireplace.
*   **OUTDOOR_TEMP_ENTITY_ID**: The outdoor temperature sensor, preferably compensated or located near the heat pump.
*   **HEATING_STATUS_ENTITY_ID**: The climate entity for your heating system, used to check if it's in 'heat' or 'auto' mode. Example: `climate.your_heating_entity`.
*   **OPENWEATHERMAP_TEMP_ENTITY_ID**: An external weather forecast temperature sensor (e.g., from OpenWeatherMap) for proactive adjustments.
*   **ML_HEATING_CONTROL_ENTITY_ID**: An `input_boolean` to enable/disable ML control. When 'on', ML actively controls heating (Active Mode). When 'off', it runs in Shadow Mode. Example: `input_boolean.ml_heating`.

#### PV (Solar) Power Entity IDs

These entities are used to measure current solar power generation and forecast.

*   **PV_POWER_ENTITY_ID**: A single aggregated sensor for total PV power generation. Example: `sensor.power_pv`.
*   **PV_FORECAST_ENTITY_ID**: The Home Assistant sensor that provides today's PV forecast with attributes (e.g., `watts` for 15-min samples) used to compute hourly forecast means. Example: `sensor.energy_production_today_4`.

#### Debugging and Monitoring

*   **DEBUG**: Set to `1` to enable detailed logging of feature vectors and model decisions. Set to `0` or remove for normal operation.
*   **CONFIDENCE_THRESHOLD**: The model uses a normalized confidence in the range (0..1], where 1.0 means perfect agreement between trees (σ = 0 °C). This threshold determines when the model's confidence is considered sufficient for optimal operation.
    *   To pick a threshold: decide the maximum tolerated σ (°C) and convert using the formula: `threshold = 1.0 / (1.0 + sigma_max)`.
    *   Example: tolerate σ_max = 1.0°C -> `CONFIDENCE_THRESHOLD = 0.5`. Default: `0.5`.
*   **BLOCKING_POLL_INTERVAL_SECONDS**: How often (in seconds) to poll blocking entities during the idle period. A value of `60` means checking the blocking state once per minute. Increase this value to reduce Home Assistant polling load, or lower it to react faster to defrost/DHW transitions. Default: `60`.
*   **GRACE_PERIOD_MAX_MINUTES**: Maximum minutes to wait during the grace period after blocking events end. Default: `30`.

#### Temperature Clamping

These settings define the absolute minimum and maximum allowed outlet temperatures proposed by the ML model. Any ML-proposed temperature will be clipped to this range.

*   **CLAMP_MIN_ABS**: Absolute minimum allowed outlet temperature (in °C). Default: `14.0`.
*   **CLAMP_MAX_ABS**: Absolute maximum allowed outlet temperature (in °C). Default: `65.0`.

#### Multi-Lag External Source Learning

These settings enable the model to learn time-delayed effects from external heat sources.

*   **ENABLE_MULTI_LAG_LEARNING**: Set to `true` to enable time-delayed learning for PV, fireplace, and TV. When enabled, the model learns how these heat sources affect indoor temperature with delays (e.g., PV warming peaks 60-90 minutes after production).
*   **PV_LAG_STEPS**: Number of 30-minute lag steps to track for PV. Example: `4` lags means tracking effects from 30, 60, 90, and 120 minutes ago.
*   **FIREPLACE_LAG_STEPS**: Number of 30-minute lag steps to track for a fireplace. Example: `4` lags means tracking immediate, 30, 60, and 90 minutes delays.
*   **TV_LAG_STEPS**: Number of 30-minute lag steps to track for TV. Example: `2` lags means tracking immediate and 30 minutes delays.

#### Seasonal Adaptation

These settings enable automatic seasonal learning to adapt to changing conditions.

*   **ENABLE_SEASONAL_ADAPTATION**: Set to `true` to enable automatic seasonal learning using cosine/sine modulation. This eliminates the need for manual recalibration between winter and summer by learning how external sources vary seasonally (e.g., windows open in summer reduce PV warming effect by ~50%).
*   **SEASONAL_LEARNING_RATE**: Learning rate for seasonal parameters. A lower value results in more stable but slower adaptation. Default: `0.01`.
*   **MIN_SEASONAL_SAMPLES**: Minimum number of samples required before seasonal learning starts. Default: `100`.

#### Summer Learning

*   **ENABLE_SUMMER_LEARNING**: Set to `true` to enable learning from periods when HVAC is off (typically during summer). This provides a cleaner signal for external source effects without heating interference, significantly improving PV and TV coefficients.

#### Hybrid Learning Strategy (Phase 2)

These settings control the intelligent learning phase classification system.

*   **HYBRID_LEARNING_ENABLED**: Set to `true` to enable the hybrid learning strategy.
*   **STABILITY_CLASSIFICATION_ENABLED**: Enable classification of periods into stable, transition, or chaotic.
*   **HIGH_CONFIDENCE_WEIGHT**: Learning weight for stable periods (high confidence). Default: `1.0`.
*   **LOW_CONFIDENCE_WEIGHT**: Learning weight for controlled transitions (low confidence). Default: `0.3`.
*   **LEARNING_PHASE_SKIP_WEIGHT**: Learning weight for chaotic periods (skip learning). Default: `0.0`.

#### Prediction Metrics Tracking

Settings for the comprehensive prediction accuracy tracking system.

*   **PREDICTION_METRICS_ENABLED**: Enable tracking of MAE and RMSE over time windows.
*   **METRICS_WINDOW_1H**: Number of samples for 1-hour window (default 12 for 5-min steps).
*   **METRICS_WINDOW_6H**: Number of samples for 6-hour window (default 72).
*   **METRICS_WINDOW_24H**: Number of samples for 24-hour window (default 288).
*   **PREDICTION_ACCURACY_THRESHOLD**: Temperature difference (°C) to consider a prediction "accurate". Default: `0.3`.

#### Trajectory Prediction Enhancement

Settings for the advanced trajectory prediction system.

*   **TRAJECTORY_PREDICTION_ENABLED**: Enable trajectory-based optimization.
*   **WEATHER_FORECAST_INTEGRATION**: Use weather forecasts in trajectory prediction.
*   **PV_FORECAST_INTEGRATION**: Use PV forecasts in trajectory prediction.
*   **OVERSHOOT_DETECTION_ENABLED**: Enable detection and prevention of temperature overshoot.
*   **TRAJECTORY_STEPS**: Number of hours to predict in trajectory optimization. Default: `4`.

#### Delta Temperature Forecast Calibration

*   **ENABLE_DELTA_FORECAST_CALIBRATION**: Enable local calibration of weather forecasts using measured temperature offset.
*   **DELTA_CALIBRATION_MAX_OFFSET**: Maximum allowed temperature offset (°C) to prevent unrealistic corrections. Default: `10.0`.

#### Thermal Equilibrium Model Parameters

These parameters control the physics-based calculations and can be tuned for different building characteristics.

*   **THERMAL_TIME_CONSTANT**: Building thermal response time in hours. Default: `4.0`.
*   **HEAT_LOSS_COEFFICIENT**: Heat loss rate per degree difference. Default: `0.3`.
*   **OUTLET_EFFECTIVENESS**: Heat pump outlet efficiency (0.5-1.0). Default: `0.5`.
*   **OUTDOOR_COUPLING**: Outdoor temperature influence factor. Default: `0.3`.
*   **THERMAL_BRIDGE_FACTOR**: Thermal bridging losses. Default: `0.1`.
*   **EQUILIBRIUM_RATIO**: Ratio defining the equilibrium point. Default: `0.17`.
*   **TOTAL_CONDUCTANCE**: Total thermal conductance of the building. Default: `0.24`.

#### External Heat Source Weights

Weights for external heat sources (only used if corresponding sensors are configured).

*   **PV_HEAT_WEIGHT**: Solar heating contribution per 100W of PV power. Default: `0.002`.
*   **FIREPLACE_HEAT_WEIGHT**: Fireplace heating contribution per unit time. Default: `5.0`.
*   **TV_HEAT_WEIGHT**: Electronics heating contribution. Default: `0.2`.

#### Adaptive Learning Parameters

Parameters controlling how the model learns from prediction errors.

*   **ADAPTIVE_LEARNING_RATE**: Base learning rate for parameter updates. Default: `0.01`.
*   **MIN_LEARNING_RATE**: Minimum allowed learning rate. Default: `0.001`.
*   **MAX_LEARNING_RATE**: Maximum allowed learning rate. Default: `0.01`.
*   **LEARNING_CONFIDENCE**: Initial confidence level for learning. Default: `3.0`.
*   **RECENT_ERRORS_WINDOW**: Number of recent errors to consider for adaptive learning. Default: `10`.

#### Historical Calibration System

Parameters for the physics-based historical parameter optimization.

*   **STABILITY_TEMP_CHANGE_THRESHOLD**: °C change threshold for stability filtering. Default: `0.1`.
*   **MIN_STABLE_PERIOD_MINUTES**: Minimum duration for stable periods. Default: `30`.
*   **OPTIMIZATION_METHOD**: Scipy optimization method. Default: `L-BFGS-B`.

#### Optional: Model Metrics Entity IDs

These entities are automatically created in Home Assistant for monitoring the model's performance. You can override their default names if needed.

*   **CONFIDENCE_ENTITY_ID**: Entity ID for the model's confidence sensor. Default: `sensor.ml_model_confidence`.
*   **MAE_ENTITY_ID**: Entity ID for the Mean Absolute Error (MAE) sensor. Default: `sensor.ml_model_mae`.
*   **RMSE_ENTITY_ID**: Entity ID for the Root Mean Square Error (RMSE) sensor. Default: `sensor.ml_model_rmse`.

## Initial Operation

### Shadow Mode (Recommended Start)

1. **Configure for Shadow Mode**
   - The add-on starts in shadow mode by default
   - It observes your current heating control but doesn't interfere
   - Perfect for safe initial learning

2. **Monitor Learning Progress**
   - Check the dashboard's **Performance** tab
   - Watch confidence levels increase over time
   - Review learning milestones

3. **Typical Learning Timeline**
   - **Week 1**: Basic learning, confidence building (0.3-0.7)
   - **Week 2-4**: Advanced features activate, confidence improves (0.7-0.9)
   - **Month 2+**: Mature operation with seasonal adaptation (0.9+)

### Switching to Active Mode

**When to Switch:**
- Confidence consistently > 0.9
- MAE (error) < 0.2°C
- System stable for 1-2 weeks

**How to Switch:**
1. **Dashboard Method**:
   - Go to **Control** tab
   - Toggle "Active Control Mode"
   - Confirm the switch

2. **Configuration Method**:
   - Add-on **Configuration** tab
   - Change mode setting
   - Restart add-on

## Monitoring & Verification

### Home Assistant Sensors

The add-on creates several sensors for monitoring:

```yaml
# Add to your dashboard
- type: entities
  title: ML Heating Status
  entities:
    - entity: sensor.ml_heating_state
      name: System Status
    - entity: sensor.ml_model_confidence
      name: Model Confidence
    - entity: sensor.ml_model_mae
      name: Prediction Error (MAE)
    - entity: sensor.ml_target_outlet_temp
      name: ML Target Temperature
```

### Performance Metrics

**Good Performance Indicators:**
- **Confidence**: > 0.9 (excellent), > 0.7 (good)
- **MAE**: < 0.2°C (excellent), < 0.3°C (good)
- **State**: "OK" most of the time
- **Temperature Stability**: Reduced variance vs. heat curve

### Dashboard Monitoring

**Overview Page:**
- System status and current operation
- Real-time confidence and performance
- Active learning milestones

**Performance Page:**
- Live MAE/RMSE tracking
- Prediction accuracy over time
- Comparison with baseline (shadow mode)

## Troubleshooting

### Common Installation Issues

**Add-on won't appear in store:**
- Verify repository URL is correct
- Check network connectivity
- Try refreshing the add-on store

**Installation fails:**
- Ensure sufficient disk space (2GB+)
- Check Home Assistant version (2023.1+)
- Review supervisor logs for errors

**Configuration errors:**
- Double-check all entity IDs exist
- Use Developer Tools → States to verify entities
- Ensure entity IDs are spelled exactly right (case-sensitive)

### Common Operation Issues

**High error rates (MAE > 0.2°C):**
- Verify sensors are stable and accurate
- Check for missing historical data in InfluxDB
- Ensure cycle interval allows proper measurement
- Review external heat sources configuration

**Low confidence (< 0.7):**
- Allow more learning time (2-4 weeks minimum)
- Verify heating system is responsive
- Check for sensor noise or dropouts
- Consider increasing cycle interval

**Dashboard not accessible:**
- Verify port 3001 is not blocked
- Check Home Assistant network settings
- Review add-on logs for startup errors

**System not learning:**
- Ensure heating system cycles properly
- Verify temperature changes are measurable
- Check for constant blocking conditions
- Review cycle timing and interval

## Advanced Topics

### Development API Usage

If you enabled the development API, you can access the system programmatically:

```python
# Example: Download live model for analysis
import requests

api_url = "http://your-ha-ip:3003"
api_key = "your-dev-api-key"

# Get system status
status = requests.get(f"{api_url}/status", 
                     headers={"X-API-Key": api_key}).json()

# Download model
model_data = requests.get(f"{api_url}/model/download",
                         headers={"X-API-Key": api_key}).content
```

### Model Backup and Migration

**Automatic Backups:**
- Models automatically backup before updates
- Stored in `/data/backups` with timestamps
- Configurable retention period

**Manual Backup:**
- Use the dashboard **Backup** tab
- Create named backups for experiments
- Export/import between systems

### Integration with Jupyter Notebooks

For advanced analysis, the original repository's Jupyter notebooks can connect to the add-on:

1. **Install notebook environment**:
   ```bash
   pip install jupyter pandas plotly requests
   ```

2. **Configure connection**:
   ```python
   # In your notebook
   from addon_client import AddonDevelopmentClient
   
   addon = AddonDevelopmentClient(
       base_url="http://homeassistant:3003",
       api_key="your-dev-api-key"
   )
   ```

3. **Download live data**:
   ```python
   # Get current model and data
   model = addon.download_live_model()
   state = addon.get_live_state()
   logs = addon.get_recent_logs(hours=24)
   ```

## Support and Resources

### Documentation
- **Main Project**: [GitHub Repository](https://github.com/helgeerbe/ml_heating)
- **Installation Guide**: This document
- **Configuration Reference**: `.env_sample` in repository
- **API Documentation**: `docs/development-api.md`

### Community
- **Issues**: [Report bugs](https://github.com/helgeerbe/ml_heating/issues)
- **Discussions**: [Community forum](https://github.com/helgeerbe/ml_heating/discussions)
- **Feature Requests**: [Enhancement proposals](https://github.com/helgeerbe/ml_heating/issues/new)

### Professional Support
For commercial installations or advanced customization needs, professional support is available through the project maintainers.

---

**🎯 Success Criteria**: You should see consistent confidence > 0.9, MAE < 0.2°C, and improved temperature stability within 2-4 weeks of operation.

**⚠️ Remember**: Always monitor initial operation closely and maintain backup heating controls until you're confident in the system's performance.
