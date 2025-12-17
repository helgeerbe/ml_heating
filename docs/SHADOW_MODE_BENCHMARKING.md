# Shadow Mode Benchmarking - Analysis Guide

## Overview

Shadow mode benchmarking provides real-time comparison between ML heating predictions and heat curve performance. This system enables data-driven decision making for when to switch from shadow mode to active ML control, while providing quantitative insights into energy efficiency potential.

## Benchmarking Methodology

### Core Comparison Logic

For every heating cycle in shadow mode, the system performs parallel analysis:

```python
# What heat curve actually did
heat_curve_outlet = actual_applied_temp  # From heating system

# What ML would have done for same conditions
ml_predicted_outlet = calculate_ml_benchmark_prediction(
    target_indoor=target_temp,
    current_indoor=current_indoor_temp,
    outdoor_temp=outdoor_temp,
    pv_power=pv_power,
    fireplace_on=fireplace_on,
    tv_on=tv_on
)

# Calculate efficiency advantage
efficiency_advantage = heat_curve_outlet - ml_predicted_outlet
# Positive = ML more efficient (lower outlet temp for same target)
```

### Benchmark Metrics

The system tracks comprehensive performance metrics:

#### 1. Outlet Temperature Efficiency
- **ML Prediction**: Outlet temp ML would use for target achievement
- **Heat Curve Actual**: Outlet temp heat curve actually used
- **Temperature Difference**: `heat_curve - ml_prediction`
- **Efficiency Score**: Lower outlet temp = more efficient operation

#### 2. Energy Savings Analysis
```python
# Estimate energy savings from temperature difference
energy_savings_pct = max(0, efficiency_advantage * 2.5)  # Rough conversion
cost_savings_per_hour = energy_savings_pct * 0.01 * baseline_consumption * energy_cost
```

#### 3. Target Achievement Accuracy
- **Heat Curve Accuracy**: How well heat curve maintains target temperature
- **ML Predicted Accuracy**: Estimated ML performance for target achievement
- **Stability Comparison**: Consistency of temperature control

#### 4. Learning Progress Indicators
- **ML Improvement Rate**: How ML efficiency changes over time
- **Convergence Quality**: Stability of ML predictions
- **Confidence Growth**: Learning system confidence progression

## Dashboard Visualizations

### 1. Outlet Temperature Comparison Chart
```python
# Real-time comparison of outlet temperatures
fig.add_trace(go.Scatter(
    x=timestamps, 
    y=ml_predictions,
    name='ML Prediction', 
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=timestamps, 
    y=heat_curve_actuals,
    name='Heat Curve Actual', 
    line=dict(color='red')
))
```

### 2. Efficiency Advantage Trend
Shows efficiency advantage over time with trend analysis:
- **Positive Values**: ML more efficient than heat curve
- **Negative Values**: Heat curve more efficient than ML
- **Trend Line**: Learning improvement direction

### 3. Energy Savings Distribution
Histogram showing distribution of potential energy savings:
- **Peak Efficiency**: Most common savings percentage
- **Outlier Analysis**: Extreme efficiency scenarios
- **Seasonal Variations**: Efficiency changes by weather

### 4. Smart Recommendations Panel
AI-powered recommendations based on benchmarking data:

```python
if avg_efficiency_advantage > 2.0:
    recommendation = "Switch to Active Mode - Significant efficiency potential"
elif abs(avg_efficiency_advantage) < 1.0:
    recommendation = "Continue Shadow Mode - Performance similar"
else:
    recommendation = "Review Configuration - Check heat curve settings"
```

## Interpretation Guide

### Efficiency Advantage Analysis

#### Scenario 1: ML Shows High Efficiency (advantage > 2°C)
```
ML Prediction: 35°C
Heat Curve:    38°C
Advantage:     +3°C (ML more efficient)

Interpretation:
✅ ML predicts significantly lower outlet temps
✅ Potential for substantial energy savings
✅ Ready for active mode transition
✅ Expected 7-8% energy reduction
```

#### Scenario 2: Similar Performance (advantage ±1°C)
```
ML Prediction: 36°C
Heat Curve:    37°C
Advantage:     +1°C (slight ML advantage)

Interpretation:
⚠️ Performance difference minimal
⚠️ Continue shadow mode learning
⚠️ Monitor trend development
⚠️ Consider heat curve optimization
```

#### Scenario 3: Heat Curve More Efficient (advantage < -1°C)
```
ML Prediction: 39°C
Heat Curve:    36°C
Advantage:     -3°C (heat curve more efficient)

Interpretation:
❌ ML predicts higher outlet temps
❌ Heat curve already optimized
❌ Check ML model training data
❌ Verify sensor calibration
```

### Target Achievement Accuracy

Monitor how well each system maintains target temperatures:

```python
# Calculate achievement accuracy
target_error_hc = abs(target_temp - actual_indoor_temp)  # Heat curve
target_error_ml = estimated_ml_accuracy  # Based on physics model

accuracy_comparison = {
    'heat_curve_mae': mean(target_errors_hc),
    'ml_estimated_mae': mean(target_errors_ml),
    'accuracy_advantage': 'ML' if ml_mae < hc_mae else 'Heat Curve'
}
```

## Performance Analysis Workflows

### Daily Review Process

1. **Check Efficiency Trends**: Review 24-hour efficiency advantage trend
2. **Analyze Energy Savings**: Calculate potential daily energy savings
3. **Monitor Learning Progress**: Verify ML predictions improving
4. **Review Target Accuracy**: Ensure temperature control quality

### Weekly Analysis

1. **Trend Analysis**: Calculate 7-day average efficiency advantage
2. **Weather Correlation**: Analyze performance across temperature ranges
3. **Seasonal Adjustments**: Account for changing weather patterns
4. **Mode Decision**: Evaluate readiness for active mode transition

### Monthly Optimization

1. **Statistical Analysis**: Comprehensive performance statistics
2. **Cost-Benefit Analysis**: Calculate actual vs potential energy costs
3. **System Optimization**: Recommend heat curve or ML adjustments
4. **Long-term Planning**: Seasonal optimization strategies

## InfluxDB Data Schema

### Benchmark Data Points

```python
# Primary benchmark metrics
measurement = "ml_heating_shadow_benchmark"
fields = {
    "ml_outlet_prediction": float,      # ML's predicted outlet temp
    "heat_curve_outlet_actual": float,  # Heat curve's actual setting
    "efficiency_advantage": float,       # Heat curve - ML prediction
    "energy_savings_pct": float,        # Estimated energy savings %
    "target_temp": float,               # Target indoor temperature
    "outdoor_temp": float,              # Outdoor temperature
    "indoor_temp_actual": float,        # Actual achieved indoor temp
    "ml_confidence": float,             # ML learning confidence
    "learning_cycles": int              # Number of learning cycles
}

# Statistical aggregations
measurement = "ml_heating_benchmark_stats"
fields = {
    "daily_avg_efficiency": float,      # Daily average efficiency
    "weekly_energy_savings": float,     # Weekly energy savings estimate
    "ml_accuracy_trend": float,         # ML accuracy improvement trend
    "optimal_transition_score": float   # Readiness for active mode
}
```

### Query Examples

```sql
-- Daily efficiency advantage trend
SELECT mean(efficiency_advantage) 
FROM ml_heating_shadow_benchmark 
WHERE time >= now() - 24h 
GROUP BY time(1h)

-- Weekly energy savings potential
SELECT sum(energy_savings_pct * estimated_consumption) as total_savings
FROM ml_heating_shadow_benchmark 
WHERE time >= now() - 7d

-- ML learning progress
SELECT mean(ml_confidence), count(learning_cycles)
FROM ml_heating_shadow_benchmark 
WHERE time >= now() - 30d 
GROUP BY time(1d)
```

## Alert System

### Efficiency Alerts

```python
# High efficiency potential detected
if efficiency_advantage > 3.0 and confidence > 0.8:
    send_alert("ML shows high efficiency potential - consider active mode")

# Performance degradation
if efficiency_advantage < -2.0:
    send_alert("Heat curve outperforming ML - check configuration")

# Learning stagnation
if confidence_growth_rate < 0.01:
    send_alert("ML learning stagnated - review parameters")
```

### Automation Triggers

- **Auto-transition to Active Mode**: When efficiency advantage > 2.5°C for 7+ days
- **Heat Curve Adjustment**: When ML consistently predicts very different outputs
- **Learning Rate Adjustment**: When convergence rate drops below threshold
- **Calibration Requests**: When both systems show poor target achievement

## Integration with Home Assistant

### Custom Sensors

```yaml
# Home Assistant sensor configuration
sensor:
  - platform: influxdb
    queries:
      - name: ML Efficiency Advantage
        query: 'SELECT last(efficiency_advantage) FROM ml_heating_shadow_benchmark'
        
      - name: Weekly Energy Savings
        query: 'SELECT sum(energy_savings_pct) FROM ml_heating_shadow_benchmark WHERE time >= now() - 7d'
        
      - name: ML Learning Confidence
        query: 'SELECT last(ml_confidence) FROM ml_heating_shadow_benchmark'
```

### Dashboard Cards

```yaml
# Lovelace dashboard integration
type: custom:apexcharts-card
title: ML vs Heat Curve Efficiency
series:
  - entity: sensor.ml_efficiency_advantage
    name: Efficiency Advantage
  - entity: sensor.weekly_energy_savings  
    name: Energy Savings %
```

## Advanced Analytics

### Statistical Analysis

```python
# Correlation analysis
outdoor_temp_correlation = calculate_correlation(outdoor_temps, efficiency_advantages)
seasonal_efficiency_pattern = group_by_season(efficiency_data)
learning_rate_analysis = calculate_learning_velocity(confidence_history)
```

### Predictive Modeling

- **Efficiency Forecasting**: Predict future efficiency based on weather forecasts
- **Optimal Transition Timing**: ML model to predict best active mode switch timing
- **Seasonal Optimization**: Adjust expectations based on seasonal patterns
- **Cost Optimization**: Predict energy cost savings with dynamic pricing

## Conclusion

Shadow mode benchmarking provides comprehensive, data-driven insights for:

- ✅ **Quantitative Performance Comparison**: ML vs heat curve efficiency metrics
- ✅ **Energy Savings Estimation**: Real-world cost-benefit analysis
- ✅ **Optimal Timing**: Data-driven active mode transition decisions
- ✅ **Continuous Optimization**: Ongoing system improvement insights
- ✅ **Transparency**: Clear visibility into ML system performance

The benchmarking system transforms shadow mode from a simple learning phase into a powerful optimization and analysis tool that provides actionable insights for energy efficiency improvement.
