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

### Step 4: Basic Configuration

1. **Open Add-on Configuration**
   - Go to the installed ML Heating Control add-on
   - Click the **Configuration** tab

2. **Configure Required Settings**
   ```yaml
   # Core temperature entities
   target_indoor_temp_entity: "climate.thermostat"
   indoor_temp_entity: "sensor.living_room_temperature"
   outdoor_temp_entity: "sensor.outdoor_temperature"
   heating_control_entity: "climate.heating_system"
   outlet_temp_entity: "sensor.heat_pump_outlet_temp"
   
   # Safety limits
   safety_max_temp: 25.0
   safety_min_temp: 18.0
   clamp_min_abs: 14.0
   clamp_max_abs: 65.0
   
   # Learning parameters
   learning_rate: 0.01
   cycle_interval_minutes: 30
   max_temp_change_per_cycle: 2.0
   ```

3. **Configure InfluxDB (Recommended)**
   ```yaml
   # InfluxDB connection
   influxdb_host: "a0d7b954-influxdb"  # Default HA InfluxDB add-on
   influxdb_port: 8086
   influxdb_database: "homeassistant"
   influxdb_username: ""  # Leave empty if using token
   influxdb_password: ""  # Leave empty if using token
   ```

4. **Save Configuration**
   - Click **Save**
   - Review any validation errors and fix them

### Step 5: Optional Features Configuration

#### PV Solar Integration
```yaml
# Solar power monitoring
pv_power_entity: "sensor.solar_power"
pv_forecast_entity: "sensor.solcast_pv_forecast_forecast_today"
```

#### External Heat Sources
```yaml
# Fireplace/wood stove
fireplace_status_entity: "binary_sensor.fireplace_active"

# TV/electronics heat
tv_power_entity: "sensor.tv_power"
```

#### Blocking Detection
```yaml
# DHW and maintenance cycles
dhw_status_entity: "binary_sensor.dhw_active"
defrost_status_entity: "binary_sensor.defrost_active"
disinfection_status_entity: "binary_sensor.disinfection_active"
dhw_boost_heater_entity: "binary_sensor.dhw_boost_active"
```

#### Development Access
```yaml
# Enable development API for Jupyter notebooks
enable_dev_api: true
dev_api_key: "your-secure-api-key-here"
```

### Step 6: Start the Add-on

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

### Step 7: Access the Dashboard

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

## Configuration Examples

### Minimal Configuration
```yaml
# Basic setup - only required entities
target_indoor_temp_entity: "climate.thermostat"
indoor_temp_entity: "sensor.living_room_temperature"
outdoor_temp_entity: "sensor.outdoor_temperature"
heating_control_entity: "climate.heating_system"
outlet_temp_entity: "sensor.heat_pump_outlet_temp"
```

### Advanced Configuration
```yaml
# Complete setup with all features
target_indoor_temp_entity: "climate.thermostat"
indoor_temp_entity: "sensor.living_room_temperature"
outdoor_temp_entity: "sensor.outdoor_temperature"
heating_control_entity: "climate.heating_system"
outlet_temp_entity: "sensor.heat_pump_outlet_temp"

# InfluxDB for historical data
influxdb_host: "a0d7b954-influxdb"
influxdb_database: "homeassistant"

# External heat sources
pv_power_entity: "sensor.solar_power"
fireplace_status_entity: "binary_sensor.fireplace_active"
tv_power_entity: "sensor.tv_power"

# Blocking detection
dhw_status_entity: "binary_sensor.dhw_active"
defrost_status_entity: "binary_sensor.defrost_active"

# Safety and performance
safety_max_temp: 24.0
safety_min_temp: 19.0
learning_rate: 0.015
cycle_interval_minutes: 30
smoothing_alpha: 0.8

# Development access
enable_dev_api: true
dev_api_key: "my-secure-api-key"
```

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

**High error rates (MAE > 0.5°C):**
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
