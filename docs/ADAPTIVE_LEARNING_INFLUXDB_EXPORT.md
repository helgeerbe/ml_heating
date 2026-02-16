# Adaptive Learning InfluxDB Export

**Phase 2 Task 2.4: InfluxDB Export Integration**

This document describes the InfluxDB export functionality for adaptive learning metrics implemented as part of the Adaptive Learning Master Plan Phase 2.

## Overview

The adaptive learning system now automatically exports comprehensive metrics to InfluxDB for monitoring, visualization, and analysis. This enables real-time tracking of:

- Prediction accuracy and trends
- Thermal model learning progress
- Learning phase classification
- Trajectory prediction performance

## Architecture

### Components

1. **Metrics Schema (`adaptive_learning_metrics_schema.py`)**
   - Defines data validation rules for all measurement types
   - Ensures data quality and consistency
   - Provides schema summary and validation functions

2. **InfluxDB Service Extensions (`influx_service.py`)**
   - `write_prediction_metrics()` - Exports MAE/RMSE tracking data
   - `write_thermal_learning_metrics()` - Exports thermal parameter evolution
   - `write_learning_phase_metrics()` - Exports learning state classification
   - `write_trajectory_prediction_metrics()` - Exports trajectory accuracy data

3. **Model Wrapper Integration (`model_wrapper.py`)**
   - Automatic export every 5 learning cycles (~25 minutes)
   - Integrated with existing prediction feedback loop
   - Zero-impact on main control logic

## Measurement Schema

### ml_prediction_metrics

Tracks prediction accuracy across different time windows:

```influx
ml_prediction_metrics,source=ml_heating,version=2.0
  mae_1h=0.25,mae_6h=0.31,mae_24h=0.28,
  rmse_1h=0.33,rmse_6h=0.39,rmse_24h=0.35,
  accuracy_excellent_pct=78.5,accuracy_very_good_pct=85.2,
  mae_improvement_pct=12.5,is_improving=true,
  total_predictions=150,predictions_24h=288
```

### ml_thermal_parameters

Tracks learned thermal model parameters:

```influx
ml_thermal_parameters,source=ml_heating,parameter_type=current
  outlet_effectiveness=0.55,heat_loss_coefficient=0.045,
  thermal_time_constant=26.5,learning_confidence=3.8,
  current_learning_rate=0.01,parameter_updates_total=45,
  outlet_effectiveness_correction_pct=12.3,
  thermal_time_constant_stability=0.89,parameter_updates_24h=12
```

### ml_learning_phase

Tracks learning state and distribution:

```influx
ml_learning_phase,source=ml_heating,learning_phase=high_confidence
  current_learning_phase="high_confidence",stability_score=0.92,
  learning_weight_applied=1.0,stable_period_duration_min=45,
  high_confidence_updates_24h=78,low_confidence_updates_24h=12,
  learning_efficiency_pct=87.5,correction_stability=0.89
```

### ml_trajectory_prediction

Tracks trajectory prediction accuracy:

```influx
ml_trajectory_prediction,source=ml_heating,prediction_horizon=4h
  trajectory_mae_1h=0.25,trajectory_mae_2h=0.32,trajectory_mae_4h=0.48,
  overshoot_predicted=false,overshoot_prevented_24h=3,
  convergence_time_avg_min=42.5,convergence_accuracy_pct=89.2,
  forecast_integration_quality=0.87
```

## Integration

### Automatic Export

The system automatically exports metrics during normal operation:

```python
# In model_wrapper.py - EnhancedModelWrapper.learn_from_prediction_feedback()
if self.cycle_count % 5 == 0:  # Every 5 learning cycles
    self._export_metrics_to_influxdb()
```

### Manual Export

For testing or manual export:

```python
from src.model_wrapper import get_enhanced_model_wrapper

wrapper = get_enhanced_model_wrapper()
wrapper._export_metrics_to_influxdb()
```

## Monitoring and Visualization

### Grafana Dashboard Queries

**Prediction Accuracy Over Time:**
```flux
from(bucket: "home_assistant/autogen")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "ml_prediction_metrics")
  |> filter(fn: (r) => r["_field"] == "mae_1h" or r["_field"] == "mae_6h")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
```

**Thermal Parameters Evolution:**
```flux
from(bucket: "home_assistant/autogen")
  |> range(start: -7d)
  |> filter(fn: (r) => r["_measurement"] == "ml_thermal_parameters")
  |> filter(fn: (r) => r["_field"] == "outlet_effectiveness")
```

**Learning Phase Distribution:**
```flux
from(bucket: "home_assistant/autogen")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "ml_learning_phase")
  |> filter(fn: (r) => r["_field"] == "current_learning_phase")
```

### Key Performance Indicators (KPIs)

1. **Prediction Accuracy**: MAE < 0.2°C for excellent performance
2. **Learning Confidence**: > 4.0 for optimal thermal parameter trust
3. **Learning Efficiency**: > 85% for effective parameter updates
4. **Trajectory Accuracy**: < 0.2°C MAE for 4-hour predictions

## Testing

### Validation Test

Run the comprehensive test suite:

```bash
cd /opt/ml_heating
python test_influxdb_export.py
```

Expected output:
```
🎯 Overall: 6/6 tests passed (100.0%)
🎉 All tests passed! InfluxDB export functionality is working correctly.
```

### Schema Validation

Validate metrics data format:

```python
from src.adaptive_learning_metrics_schema import validate_metrics_data

# Test prediction metrics
is_valid = validate_metrics_data("ml_prediction_metrics", {
    "mae_1h": 0.25,
    "total_predictions": 150
})
```

## Performance Impact

- **Memory**: < 1MB additional memory usage
- **CPU**: < 0.1% additional CPU during export cycles  
- **Network**: ~2KB per export cycle (every 25 minutes)
- **Storage**: ~50MB per year in InfluxDB

## Error Handling

The export system is designed to be non-disruptive:

- Failed exports are logged but don't affect control logic
- Automatic retry logic for transient InfluxDB connection issues
- Graceful degradation when InfluxDB is unavailable
- Schema validation prevents corrupt data export

## Configuration

### InfluxDB Settings

Uses existing InfluxDB configuration from `config.py`:

```python
INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "your_token"
INFLUX_ORG = "your_org" 
INFLUX_BUCKET = "home_assistant/autogen"
```

### Export Frequency

Controlled in `model_wrapper.py`:

```python
# Export every N cycles (default: 5 = ~25 minutes)
EXPORT_FREQUENCY = 5
```

## Troubleshooting

### Common Issues

1. **InfluxDB Connection Failed**
   - Check `INFLUX_TOKEN` and `INFLUX_URL` in configuration
   - Verify InfluxDB service is running
   - Check network connectivity

2. **Schema Validation Errors**
   - Review data types in metrics (float vs int vs bool)
   - Check for required fields in measurement data
   - Validate timestamp format

3. **Export Frequency**
   - Adjust `EXPORT_FREQUENCY` in model_wrapper.py
   - Monitor CPU/network impact of frequent exports
   - Consider InfluxDB retention policies

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Verification

Check InfluxDB data:

```flux
from(bucket: "home_assistant/autogen")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] =~ /^ml_/)
  |> count()
```

## Future Enhancements

- **Alerting**: Automatic alerts for degraded learning performance
- **Anomaly Detection**: ML-based detection of unusual learning patterns
- **Batch Export**: Export multiple cycles in single batch for efficiency
- **Compression**: Compress historical data for long-term storage

## References

- [Adaptive Learning Master Plan](memory-bank/ADAPTIVE_LEARNING_MASTER_PLAN.md)
- [InfluxDB Line Protocol](https://docs.influxdata.com/influxdb/v2.0/reference/syntax/line-protocol/)
- [Grafana Dashboard Examples](docs/GRAFANA_DASHBOARD_EXAMPLES.md)
