# Technology Context - ML Heating Control System

## Technology Stack

### Core Technologies

**Python 3.8+**
- **Primary Language**: Chosen for rich ML ecosystem and Home Assistant integration
- **Version**: 3.8+ required for type hints and asyncio improvements
- **Virtual Environment**: `.venv` for isolated dependency management

**River Framework**
- **Online Machine Learning**: Incremental learning algorithms
- **Streaming ML**: Designed for continuous data streams
- **Key Components**: 
  - `river.metrics.MAE/RMSE` for performance tracking
  - Online regression models for continuous adaptation
- **Why River**: Perfect fit for learning from every heating cycle without batch retraining

**Home Assistant REST API**
- **Integration Pattern**: RESTful API communication via HTTP/HTTPS
- **Authentication**: Long-lived access tokens
- **Bidirectional**: Read sensors + write calculated temperatures
- **Entity Management**: Create and update custom sensors for monitoring

**InfluxDB 2.x**
- **Time-Series Database**: Historical sensor data storage
- **Query Language**: Flux queries for feature engineering
- **Data Source**: Existing Home Assistant integration via InfluxDB connector
- **Usage Patterns**:
  - Historical data retrieval for model calibration
  - Feature importance export for analysis
  - Learning metrics tracking

### Development Infrastructure

**Jupyter Notebooks**
- **Analysis Platform**: 6 specialized notebooks for monitoring and debugging
- **Interactive Analysis**: Real-time model diagnostics and visualization
- **User Interface**: Primary tool for understanding system behavior
- **Notebooks**:
  - `00_learning_dashboard.ipynb` - At-a-glance status
  - `01_physics_model_diagnosis.ipynb` - Model validation
  - `02_performance_monitoring.ipynb` - Performance tracking
  - `03_behavior_analysis.ipynb` - Heating patterns
  - `04_model_validation.ipynb` - Physics compliance
  - `05_multilag_seasonal_analysis.ipynb` - Advanced features

**Systemd Service Management**
- **Production Deployment**: Linux systemd service for 24/7 operation
- **Auto-restart**: Configured with failure recovery and restart delays
- **Logging**: Journal integration for centralized log management
- **Dependencies**: Proper service ordering with Home Assistant and InfluxDB

**Environment Configuration**
- **dotenv**: `.env` file for configuration management
- **Secrets Management**: Tokens and sensitive data isolated from code
- **Configuration Validation**: Comprehensive parameter validation and defaults

## Key Dependencies

### Core Python Packages

```python
# requirements.txt key dependencies
numpy>=1.21.0           # Numerical computing for physics calculations
pandas>=1.3.0           # Data manipulation and feature engineering
requests>=2.26.0        # HTTP client for Home Assistant API
influxdb-client>=1.24.0 # InfluxDB 2.x integration
river>=0.15.0           # Online machine learning
python-dotenv>=0.19.0   # Environment configuration
matplotlib>=3.5.0       # Visualization in notebooks
jupyter>=1.0.0          # Notebook environment
```

### System Dependencies

**Operating System Requirements**:
- **Linux**: Primary target (systemd service support)
- **Python 3.8+**: Language runtime
- **systemd**: Service management (production deployment)
- **Network Access**: Connectivity to Home Assistant and InfluxDB

**Optional Dependencies**:
- **Git**: Version control and updates
- **Docker**: Alternative containerized deployment
- **Grafana**: Advanced visualization (future enhancement)

## Architecture Decisions

### Why Online Learning (River) vs Traditional ML

**Traditional Approach Issues**:
- **Batch Training**: Requires periodic model retraining
- **Data Management**: Complex pipeline for collecting training data
- **Concept Drift**: Poor adaptation to seasonal/house changes
- **Deployment Complexity**: Separate training and inference infrastructure

**River Online Learning Benefits**:
- **Continuous Learning**: Adapts every 30-minute cycle
- **No Retraining**: Model evolves automatically
- **Concept Drift Handling**: Built-in adaptation to changing conditions
- **Simple Deployment**: Single process handles learning and inference
- **Memory Efficiency**: Incremental updates, no large datasets

### Physics-Based Model Design

**Hybrid Approach**:
```python
class RealisticPhysicsModel:
    def predict_one(self, features):
        # Physics-based core (domain knowledge)
        base_heating = outlet_effect * self.base_heating_rate
        target_boost = temp_gap * self.target_influence
        weather_adjustment = base_heating * outdoor_penalty
        
        # Data-driven external sources (learned)
        pv_contribution = self._calculate_pv_lagged()
        fireplace_contribution = self._calculate_fireplace_lagged()
        
        return base_heating + target_boost + weather_adjustment + pv_contribution + fireplace_contribution
```

**Benefits**:
- **Faster Convergence**: Physics knowledge reduces required training data
- **Interpretability**: Clear understanding of model decisions
- **Bounds Checking**: Natural constraints prevent unrealistic predictions
- **Robustness**: Graceful degradation when external sources unavailable

### Multi-Lag Learning Architecture

**Ring Buffer Implementation**:
```python
# Efficient time-delay tracking
self.pv_history = []      # Last 5 cycles (0-120min)
self.fireplace_history = [] # Last 4 cycles (0-90min)  
self.tv_history = []      # Last 2 cycles (0-30min)

def _update_histories(self, pv_power, fireplace, tv):
    self.pv_history.append(pv_power)
    if len(self.pv_history) > 5:
        self.pv_history.pop(0)  # Ring buffer behavior
```

**Correlation-Based Learning**:
```python
def _learn_pv_lags(self):
    # Calculate correlations between power at different lags and temperature changes
    for lag_idx in range(1, 5):
        lag_powers = [effect['power_history'][-1-lag_idx] for effect in recent_effects]
        correlation = np.corrcoef(lag_powers, temperature_changes)[0, 1]
        # Distribute total effect by correlation strength
```

### Seasonal Adaptation via Trigonometric Functions

**Mathematical Foundation**:
```python
# Automatic seasonal variation modeling
month_radians = 2 * π * current_month / 12
seasonal_multiplier = 1.0 + (
    pv_seasonal_cos * cos(month_radians) + 
    pv_seasonal_sin * sin(month_radians)
)

# Applied to external sources
pv_effect = base_pv_effect * seasonal_multiplier
```

**Learning Approach**:
- **Summer Data Collection**: Clean signal when HVAC is off
- **Correlation Analysis**: Learn cos/sin coefficients from HVAC-off periods
- **Automatic Modulation**: ±30-50% seasonal variation without manual adjustment

### Safety and Robustness Patterns

**Multi-Layer Safety Architecture**:

**Layer 1: Configuration Bounds**
```python
CLAMP_MIN_ABS = 14.0°C   # Absolute minimum outlet temperature
CLAMP_MAX_ABS = 65.0°C   # Absolute maximum outlet temperature
```

**Layer 2: Rate Limiting**
```python
MAX_TEMP_CHANGE_PER_CYCLE = 2°C  # Gradual changes only
if abs(delta) > max_change:
    final_temp = baseline + sign(delta) * max_change
```

**Layer 3: Blocking Detection**
```python
# Pause during heat pump system operations
blocking_entities = [DHW_STATUS, DEFROST_STATUS, DISINFECTION_STATUS]
if any_blocking_active:
    skip_prediction_cycle()
    preserve_state_for_grace_period()
```

**Layer 4: Grace Period Restoration**
```python
# Intelligent recovery after blocking ends
if blocking_just_ended:
    if outlet_hotter_than_target:  # DHW scenario
        use_aggressive_cooldown_target()
    else:  # Defrost scenario  
        restore_exact_previous_target()
    wait_for_outlet_stabilization()
```

**Layer 5: Physics Validation**
```python
# Monotonic enforcement - higher outlet must predict higher indoor
def enforce_monotonic(candidates, predictions):
    # Correct any thermodynamically impossible predictions
```

## Development Patterns

### Configuration Management

**Environment-Based Configuration**:
```python
# config.py pattern
HASS_URL = os.getenv("HASS_URL", "https://home.example.com")
HASS_TOKEN = os.getenv("HASS_TOKEN", "").strip()
CYCLE_INTERVAL_MINUTES = int(os.getenv("CYCLE_INTERVAL_MINUTES", "30"))
```

**Entity ID Mapping**:
```python
# Flexible entity mapping for different HA setups
TARGET_INDOOR_TEMP_ENTITY_ID = os.getenv("TARGET_INDOOR_TEMP_ENTITY_ID", "input_number.target_temp")
INDOOR_TEMP_ENTITY_ID = os.getenv("INDOOR_TEMP_ENTITY_ID", "sensor.indoor_temp")
```

### Error Handling and Resilience

**Network Resilience**:
```python
def create_ha_client():
    try:
        return HAClient(config.HASS_URL, config.HASS_TOKEN)
    except Exception:
        logging.error("Failed to create HA client")
        publish_network_error_state()
        raise
```

**Graceful Degradation**:
```python
# Handle missing optional sensors
pv_power = ha_client.get_state(config.PV1_POWER_ENTITY_ID) or 0.0
fireplace_on = ha_client.get_state(config.FIREPLACE_STATUS_ENTITY_ID, is_binary=True) or False

# Continue operation with reduced functionality
if all_critical_sensors_available:
    perform_full_prediction()
else:
    log_missing_sensors()
    skip_cycle_safely()
```

### Monitoring and Observability

**Comprehensive State Reporting**:
```python
# ML State Sensor with rich diagnostics
attributes = {
    'confidence': round(confidence, 4),
    'mae': round(mae.get(), 4), 
    'rmse': round(rmse.get(), 4),
    'suggested_temp': round(suggested_temp, 2),
    'predicted_indoor': round(predicted_indoor, 2),
    'missing_sensors': missing_sensors,
    'blocking_reasons': blocking_reasons,
    'last_prediction_time': datetime.now().isoformat()
}
```

**Learning Metrics Export**:
```python
# Export to InfluxDB for analysis
learning_metrics = model.export_learning_metrics()
influx_service.write_feature_importances(
    learning_metrics, 
    bucket=config.INFLUX_FEATURES_BUCKET,
    measurement="learning_parameters"
)
```

## Deployment Architecture

### Production Service Configuration

**Systemd Service Unit**:
```ini
[Unit]
Description=ML Heating Control Service
After=network.target home-assistant.service influxdb.service

[Service]
Type=simple
User=ml_heating
WorkingDirectory=/opt/ml_heating
ExecStart=/opt/ml_heating/.venv/bin/python3 -m src.main
Restart=on-failure
RestartSec=5m

[Install]
WantedBy=multi-user.target
```

**File System Layout**:
```
/opt/ml_heating/
├── .venv/                 # Python virtual environment
├── src/                   # Source code modules
├── notebooks/            # Analysis notebooks  
├── memory-bank/          # Documentation and context
├── .env                  # Configuration (not in git)
├── requirements.txt      # Python dependencies
├── ml_model.pkl         # Trained model state
└── ml_state.pkl         # Application state
```

### Integration Patterns

**Home Assistant Integration**:
- **REST API**: Bidirectional communication via HTTP/HTTPS
- **Entity Creation**: Dynamic sensor creation for monitoring
- **Authentication**: Long-lived access tokens
- **Error Recovery**: Graceful handling of HA restarts/updates

**InfluxDB Integration**:
- **Historical Queries**: Feature engineering from time-series data
- **Metrics Export**: Learning parameters and feature importance
- **Batch Processing**: Efficient bulk data retrieval for calibration

**Logging and Monitoring**:
- **Systemd Journal**: Centralized logging via journalctl
- **Home Assistant Sensors**: Real-time status monitoring
- **Jupyter Notebooks**: Interactive analysis and debugging

This technology stack provides a robust, maintainable, and scalable foundation for sophisticated heating control with continuous learning and comprehensive safety mechanisms.
