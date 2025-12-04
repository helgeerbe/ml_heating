# Monitoring Notebooks

This directory contains real-time monitoring dashboards for Phase 2 adaptive learning features.

## Purpose

These notebooks provide monitoring and alerting capabilities for the adaptive learning system, allowing real-time visibility into system performance and health.

## Notebooks

### 01_hybrid_learning_monitor.ipynb
**Monitors: Hybrid Learning Strategy Performance**
- Real-time learning phase classification monitoring
- Phase transition frequency and duration analysis
- Weighted learning effectiveness tracking
- Stability detection accuracy assessment
- **Key Metrics**: Current phase, transition rate, learning improvement

### 02_prediction_accuracy_monitor.ipynb
**Monitors: Multi-Timeframe Prediction Accuracy**
- Rolling window MAE/RMSE tracking (1h, 6h, 24h)
- Accuracy trend analysis and degradation detection
- Performance classification breakdown
- Recent prediction performance monitoring
- **Key Metrics**: Multi-window accuracy, trend analysis, accuracy classification

### 03_trajectory_prediction_monitor.ipynb
**Monitors: Trajectory Prediction with Forecast Integration**
- Trajectory vs standard prediction performance comparison
- Weather and PV forecast integration effectiveness
- Overshoot detection and prevention monitoring
- Energy efficiency and comfort benefit tracking
- **Key Metrics**: Forecast usage, overshoot prevention, efficiency benefits

## Usage

### Real-Time Monitoring
1. Run notebooks periodically to check system health
2. Monitor for performance degradation alerts
3. Track improvement trends over time
4. Validate feature effectiveness

### Alert Thresholds
- **Accuracy Degradation**: >20% decrease in prediction accuracy
- **Phase Transitions**: Excessive phase switching (>5/hour)
- **Forecast Issues**: Data age >60 minutes or integration failure
- **Overshoot Prevention**: Rate below 70%

## Key Performance Indicators

### Hybrid Learning Strategy
- ✅ **Phase Classification Accuracy**: >90%
- ✅ **Stable Period Duration**: >15 minutes average
- ✅ **Learning Improvement**: >10% vs standard approach

### Prediction Accuracy
- ✅ **Good+ Accuracy**: >80% of predictions within 0.5°C
- ✅ **1H Window MAE**: <0.3°C target
- ✅ **Trend**: Stable or improving accuracy

### Trajectory Prediction
- ✅ **Prediction Improvement**: >20% vs standard
- ✅ **Forecast Usage**: >80% integration rate
- ✅ **Overshoot Prevention**: >80% prevention rate

## Prerequisites

- Active ml_heating system with Phase 2 features enabled
- Historical prediction data for trend analysis
- Access to forecast data for integration monitoring

## Integration

These monitoring notebooks integrate with:
- `prediction_metrics.py` for accuracy tracking
- `enhanced_trajectory.py` for trajectory monitoring  
- Phase 2 configuration parameters
- Real-time system state and metrics

## Notes

- Monitoring templates include placeholder data for testing
- Actual monitoring connects to live system metrics
- All notebooks handle missing data gracefully
- Alerts and thresholds are configurable
