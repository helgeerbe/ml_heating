# Home Assistant Sensor Attributes Guide

This guide documents all Home Assistant sensors created by the ML Heating system and explains the meaning of their attributes.

## Overview

The ML Heating system exports several sensors to Home Assistant to provide real-time monitoring and insights into the thermal model's behavior, learning progress, and performance metrics.

## Sensors and Attributes

### 1. `sensor.ml_heating_learning`

**Primary Function**: Tracks adaptive learning progress and current thermal parameters.

**State**: Learning confidence score (0.0 - 10.0)
- Higher values indicate more confident predictions
- Starts at 3.0 for new systems
- Increases as the model learns your house's thermal characteristics

#### Thermal Parameters (Core Physics Model)

| Attribute | Unit | Typical Range | Description |
|-----------|------|---------------|-------------|
| `thermal_time_constant` | hours | 2.0 - 8.0 | How quickly your house responds to temperature changes. Lower values = faster heating/cooling response |
| `total_conductance` | thermal units | 0.1 - 0.5 | Overall heat transfer efficiency of your heating system. Higher = more efficient heat transfer |
| `equilibrium_ratio` | ratio | 0.3 - 0.7 | Balance between direct heating and thermal mass effects. Critical for accurate predictions |
| `heat_loss_coefficient` | 1/hour | 0.01 - 0.5 | Rate of heat loss to environment. Higher values = more heat loss (less insulated house) |
| `outlet_effectiveness` | ratio | 0.5 - 1.0 | How effectively outlet temperature translates to indoor heating. 1.0 = perfect efficiency |

#### Learning Progress Indicators

| Attribute | Description |
|-----------|-------------|
| `cycle_count` | Number of heating cycles the system has learned from |
| `parameter_updates` | Total number of thermal parameter adjustments made |
| `model_health` | Overall system status: "healthy", "learning", "degraded", or "unknown" |
| `learning_progress` | Progress towards full learning (0.0 - 1.0, based on cycle_count/100) |
| `is_improving` | Boolean: whether predictions are getting more accurate over time |
| `improvement_percentage` | Rate of accuracy improvement as percentage |
| `total_predictions` | Total number of temperature predictions made |

### 2. `sensor.ml_model_mae`

**Primary Function**: Mean Absolute Error tracking across different time windows.

**State**: All-time Mean Absolute Error (°C)

#### Time-Windowed Accuracy

| Attribute | Unit | Description |
|-----------|------|-------------|
| `mae_1h` | °C | Average prediction error over last hour |
| `mae_6h` | °C | Average prediction error over last 6 hours |
| `mae_24h` | °C | Average prediction error over last 24 hours |
| `trend_direction` | text | "improving", "stable", or "degrading" based on recent performance |
| `prediction_count` | count | Total predictions used for MAE calculation |

**Interpretation**:
- **Good**: MAE < 0.5°C
- **Acceptable**: MAE 0.5-1.0°C  
- **Poor**: MAE > 1.0°C

### 3. `sensor.ml_model_rmse`

**Primary Function**: Root Mean Squared Error with error distribution analysis.

**State**: All-time Root Mean Squared Error (°C)

#### Error Distribution Analysis

| Attribute | Unit | Description |
|-----------|------|-------------|
| `recent_max_error` | °C | Largest single prediction error in recent period |
| `std_error` | °C | Standard deviation of prediction errors (consistency measure) |
| `mean_bias` | °C | Systematic over/under-prediction tendency |
| `prediction_count` | count | Total predictions used for RMSE calculation |

**Interpretation**:
- **RMSE vs MAE**: If RMSE >> MAE, indicates occasional large errors
- **std_error**: Lower values = more consistent predictions
- **mean_bias**: Values near 0 = unbiased; positive = over-predicting; negative = under-predicting

### 4. `sensor.ml_prediction_accuracy`

**Primary Function**: Control quality assessment over 24-hour window.

**State**: Percentage of "good control" predictions (within ±0.2°C tolerance)

#### Control Quality Breakdown

| Attribute | Unit | Description |
|-----------|------|-------------|
| `perfect_accuracy_pct` | % | Predictions within ±0.1°C (24h window) |
| `tolerable_accuracy_pct` | % | Predictions within ±0.2°C (24h window) |
| `poor_accuracy_pct` | % | Predictions with >0.5°C error (24h window) |
| `prediction_count_24h` | count | Total predictions in 24h window |
| `excellent_all_time_pct` | % | All-time percentage of excellent predictions |
| `good_all_time_pct` | % | All-time percentage of good predictions |

**Control Quality Thresholds**:
- **Excellent**: ±0.1°C (perfect for comfort)
- **Good**: ±0.2°C (acceptable for most users)
- **Tolerable**: ±0.5°C (noticeable but acceptable)
- **Poor**: >0.5°C (user likely to notice temperature swings)

### 5. `sensor.ml_heating_state`

**Primary Function**: Current operational status and mode information.

**State**: Numeric operational state code

**Common Attributes**: Varies by operational context, typically includes current temperatures, mode information, and system status.

### 6. `sensor.ml_feature_importance`

**Primary Function**: Shows which factors most influence heating decisions.

**State**: Number of features analyzed

#### Feature Analysis

| Attribute | Description |
|-----------|-------------|
| `top_features` | Dictionary of most important features and their influence percentages |

**Common Important Features**:
- Outdoor temperature (usually 20-40% influence)
- Time of day patterns
- Recent heating history
- Forecast temperatures
- Indoor/outdoor temperature difference

## Monitoring Best Practices

### Health Indicators to Watch

1. **Learning Confidence**: Should gradually increase from 3.0 to 6.0+ over first week
2. **MAE Trend**: Should show "improving" or "stable" after initial learning period
3. **Control Quality**: Aim for >70% good control (±0.2°C) after learning period
4. **Parameter Stability**: Thermal parameters should stabilize after initial calibration

### Warning Signs

- **Learning confidence decreasing**: May indicate changing house conditions or system issues
- **MAE trending "degrading"**: Possible sensor issues or environmental changes
- **High RMSE vs MAE ratio**: Indicates inconsistent predictions, possible data quality issues
- **Low outlet effectiveness (<0.5)**: May indicate heating system efficiency problems

### Troubleshooting

**High Prediction Errors**:
1. Check sensor accuracy (indoor, outdoor, outlet temperatures)
2. Verify heating system is functioning normally
3. Consider if house conditions have changed (windows open, different usage patterns)
4. Review if calibration is needed

**Poor Learning Progress**:
1. Ensure sufficient heating cycles (system needs variety of conditions to learn)
2. Check for consistent sensor readings
3. Verify no external heat sources interfering (fireplace, solar gain, etc.)

## Integration with Home Assistant

### Useful Automations

```yaml
# Example: Alert when prediction accuracy drops
- alias: "ML Heating: Poor Accuracy Alert"
  trigger:
    - platform: numeric_state
      entity_id: sensor.ml_prediction_accuracy
      below: 50
      for: "01:00:00"
  action:
    - service: notify.mobile_app
      data:
        message: "ML Heating prediction accuracy has dropped to {{ states('sensor.ml_prediction_accuracy') }}%"
```

### Dashboard Cards

**Thermal Parameters Card**:
```yaml
type: entities
title: Thermal Parameters
entities:
  - entity: sensor.ml_heating_learning
    name: Learning Confidence
  - type: attribute
    entity: sensor.ml_heating_learning
    attribute: thermal_time_constant
    name: Time Constant (h)
  - type: attribute
    entity: sensor.ml_heating_learning
    attribute: total_conductance
    name: Conductance
  - type: attribute
    entity: sensor.ml_heating_learning
    attribute: equilibrium_ratio
    name: Equilibrium Ratio
```

**Performance Overview**:
```yaml
type: entities
title: ML Performance
entities:
  - entity: sensor.ml_model_mae
    name: Prediction Error (MAE)
  - entity: sensor.ml_prediction_accuracy
    name: Control Quality
  - type: attribute
    entity: sensor.ml_model_mae
    attribute: trend_direction
    name: Accuracy Trend
```

## Version History

- **v1.0**: Initial sensor implementation with basic thermal parameters
- **v1.1**: Added JSON serialization fix for numpy boolean types
- **v1.2**: Enhanced sensor attributes with complete thermal parameter set (added total_conductance and equilibrium_ratio)

## Related Documentation

- [Shadow Mode User Guide](SHADOW_MODE_USER_GUIDE.md) - Understanding shadow mode operation
- [Thermal Model Deep Analysis](THERMAL_MODEL_DEEP_ANALYSIS.md) - Technical details of thermal physics
- [Installation Guide](INSTALLATION_GUIDE.md) - Setup and configuration
