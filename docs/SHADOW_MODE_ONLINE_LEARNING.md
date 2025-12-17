# Shadow Mode Online Learning - Technical Guide

## Overview

Shadow mode online learning is a revolutionary approach that solves the critical issue of ML heating learning bias. In traditional shadow mode, the system incorrectly learned from heat curve behavior, causing ML to predict progressively lower outlet temperatures. This implementation provides **pure physics learning** combined with comprehensive **ML vs heat curve benchmarking**.

## Core Problem Solved

**Issue**: Shadow mode was learning `ML_prediction → actual_indoor_temp`, but since the heat curve maintained correct indoor temperatures, ML learned to reduce outlet predictions over time.

**Solution**: Shadow mode now learns `heat_curve_outlet → actual_indoor_temp` (pure physics) while benchmarking ML efficiency against heat curve performance.

## Pure Physics Learning Algorithm

### Learning Context (Shadow Mode)

```python
# Shadow mode learns pure physics relationships
if config.SHADOW_MODE:
    prediction_context = {
        'outlet_temp': actual_applied_temp,    # What heat curve actually set
        'outdoor_temp': outdoor_temp,
        'pv_power': pv_power,
        'fireplace_on': fireplace_on,
        'tv_on': tv_on
        # NO target_indoor_temp - completely isolated from learning!
    }
    
    # Learn: outlet_temp → actual_indoor_temp (physics only)
    physics_prediction = thermal_model.predict_equilibrium_temperature(
        outlet_temp=prediction_context['outlet_temp'],  # Heat curve setting
        outdoor_temp=outdoor_temp,
        current_indoor=predicted_temp
        # Target temperature NOT involved in learning
    )
    
    # Update parameters based on physics error only
    physics_error = actual_temp - physics_prediction
    thermal_model.update_parameters_with_gradient(learning_features, physics_error)
```

### Parameter Learning Algorithm

The system uses **gradient-based parameter updates** with physics-only feedback:

1. **Outlet Effectiveness Learning**:
   ```python
   outlet_effectiveness += learning_rate * physics_error * outlet_gradient
   ```

2. **Heat Loss Coefficient Learning**:
   ```python
   heat_loss_coefficient += learning_rate * physics_error * heat_loss_gradient
   ```

3. **Thermal Time Constant Learning**:
   ```python
   thermal_time_constant += learning_rate * physics_error * time_constant_gradient
   ```

### Parameter Bounds Enforcement

All parameters maintain physically realistic bounds during learning:

```python
# Enforce realistic parameter ranges
outlet_effectiveness = np.clip(outlet_effectiveness, 0.1, 1.0)
heat_loss_coefficient = np.clip(heat_loss_coefficient, 10.0, 500.0)
thermal_time_constant = np.clip(thermal_time_constant, 900.0, 14400.0)
```

## ML vs Heat Curve Benchmarking

### Benchmarking Calculation

Shadow mode simultaneously calculates what ML would predict for current conditions:

```python
# Calculate ML's prediction for current target
ml_predicted_outlet = model_wrapper.calculate_optimal_outlet_temperature(
    target_indoor=target_indoor_temp,
    current_indoor=current_indoor_temp,
    outdoor_temp=outdoor_temp,
    pv_power=pv_power,
    fireplace_on=fireplace_on,
    tv_on=tv_on
)['optimal_outlet_temp']

# Compare against heat curve's actual setting
efficiency_advantage = heat_curve_actual - ml_predicted_outlet
# Positive = ML more efficient (predicts lower outlet temp)
```

### Efficiency Metrics

The system tracks comprehensive efficiency metrics:

- **Outlet Temperature Difference**: How much lower/higher ML predicts vs heat curve
- **Energy Savings Percentage**: Estimated energy reduction from ML efficiency
- **Target Achievement Accuracy**: How well each system maintains target temperature
- **Learning Progress**: Improvement in ML predictions over time

### InfluxDB Export

All benchmark data is exported to InfluxDB for analytics:

```python
def write_shadow_mode_benchmarks(self, benchmark_data):
    """Export shadow mode benchmarking data"""
    point = Point("ml_heating_shadow_benchmark") \
        .field("ml_outlet_prediction", benchmark_data['ml_outlet_prediction']) \
        .field("heat_curve_outlet_actual", benchmark_data['heat_curve_outlet_actual']) \
        .field("efficiency_advantage", benchmark_data['efficiency_advantage']) \
        .field("target_temp", benchmark_data['target_temp']) \
        .field("outdoor_temp", benchmark_data['outdoor_temp']) \
        .field("energy_savings_pct", benchmark_data['energy_savings_pct']) \
        .time(benchmark_data['timestamp'])
```

## Startup Workflows

### Fresh Start (No thermal_state.json)

1. **Load Default Parameters**: System starts with reasonable physics defaults
2. **Initialize Learning**: `learning_confidence = 3.0` (ready to learn)
3. **Begin Shadow Mode**: Start pure physics learning immediately
4. **Parameter Convergence**: Physics parameters adapt to real building characteristics
5. **Benchmarking**: Compare ML efficiency vs heat curve from day 1

### Existing Calibration (thermal_state.json exists)

1. **Load Calibrated Baseline**: Use existing physics parameters as starting point
2. **Continuous Refinement**: Shadow mode continues improving parameters
3. **Enhanced Benchmarking**: Compare improved ML model vs heat curve
4. **Seamless Transition**: No disruption to existing calibration

## Configuration

Shadow mode uses the existing `SHADOW_MODE` configuration flag:

```yaml
# config.yaml
SHADOW_MODE: true   # Enable pure physics learning + benchmarking
SHADOW_MODE: false  # Use learned parameters in active mode
```

**No additional configuration required!**

## Learning Confidence

Shadow mode builds confidence through successful physics learning cycles:

- **Initial Confidence**: 3.0 (ready to learn immediately)
- **Confidence Growth**: Increases with consistent parameter convergence
- **Transition Threshold**: Switch to active mode when confidence > 7.0
- **Learning Rate**: Adaptive based on prediction accuracy

## Parameter Convergence Indicators

Monitor these metrics to assess learning quality:

1. **Physics Error Reduction**: MAE/RMSE decreasing over time
2. **Parameter Stability**: Less parameter drift between cycles
3. **Prediction Consistency**: Similar predictions for similar conditions
4. **Efficiency Advantage**: ML showing energy savings potential

## Advanced Features

### Multi-Heat Source Support

Shadow mode learning works with complex heating systems:

- **Heat Pumps**: Learn outlet effectiveness curves
- **Fireplaces**: Track adaptive learning integration
- **Solar Systems**: Account for PV power contributions
- **Hybrid Systems**: Optimize multi-source coordination

### Weather Adaptation

Physics learning adapts to various weather conditions:

- **Temperature Ranges**: Learn across full outdoor temperature spectrum
- **Seasonal Variations**: Adapt parameters for heating season changes
- **Extreme Conditions**: Maintain performance in unusual weather

### Integration Points

Shadow mode integrates seamlessly with existing systems:

- **Home Assistant**: Real-time sensor data integration
- **InfluxDB**: Historical data storage and analytics
- **Streamlit Dashboard**: Live monitoring and visualization
- **Adaptive Learning**: Enhanced parameter optimization

## Troubleshooting

### Common Issues

1. **Slow Convergence**: Check sensor calibration and data quality
2. **Parameter Drift**: Verify heat curve stability and outdoor sensor accuracy
3. **Poor Benchmarking**: Ensure ML model has sufficient training data
4. **Learning Stagnation**: Consider increasing learning rate or resetting parameters

### Diagnostic Tools

- **Learning Progress Charts**: Monitor parameter convergence trends
- **Physics Error Metrics**: Track prediction accuracy improvement
- **Benchmark Comparisons**: Analyze ML vs heat curve performance
- **Parameter History**: Review learning trajectory and stability

## Performance Optimization

### Grace Period Optimization

Shadow mode automatically disables grace periods for faster learning:

```python
# Grace period bypass in shadow mode
if config.SHADOW_MODE:
    return False  # No grace period needed - ML only observing
```

**Benefits**:
- ✅ **Faster Learning**: More frequent observation cycles
- ✅ **Better Benchmarking**: More data points for ML vs heat curve comparison  
- ✅ **No Equipment Risk**: Since ML doesn't control heating equipment
- ✅ **Immediate Insights**: Real-time efficiency analysis without artificial delays

### Learning Rate Tuning

Adjust learning rates based on system characteristics:

```python
# Conservative learning for stable systems
learning_rate = 0.01

# Aggressive learning for rapid adaptation
learning_rate = 0.05
```

### Memory Management

Shadow mode maintains efficient memory usage:

- **Rolling History**: Keep last 1000 learning cycles
- **Parameter Snapshots**: Periodic parameter state backups
- **Gradient Caching**: Optimize repeated calculations

## Conclusion

Shadow mode online learning provides a robust, tested solution for ML heating calibration that:

- ✅ **Eliminates Learning Bias**: Pure physics learning without target temperature influence
- ✅ **Enables Continuous Improvement**: Ongoing parameter refinement in production
- ✅ **Provides Efficiency Insights**: Quantitative ML vs heat curve comparison
- ✅ **Supports All Configurations**: Works with fresh starts and existing calibrations
- ✅ **Maintains Production Stability**: Comprehensive testing and validation

The system is production-ready and provides the foundation for advanced ML heating optimization.
