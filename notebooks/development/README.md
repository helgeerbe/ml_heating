# Development Notebooks

This directory contains development notebooks for implementing Phase 2 adaptive learning features.

## Purpose

These notebooks provide development environments for implementing and testing new adaptive learning enhancements before integration into the main system.

## Notebooks

### 01_hybrid_learning_strategy_development.ipynb
**Phase 2 Enhancement: Intelligent Learning Phase Classification**
- Develops hybrid learning strategy with stability-based weighting
- Implements phase classification algorithms
- Tests weighted learning effectiveness
- **Configuration**: `HYBRID_LEARNING_ENABLED`, `STABILITY_CLASSIFICATION_ENABLED`

### 02_mae_rmse_tracking_development.ipynb  
**Phase 2 Enhancement: Multi-Timeframe Prediction Accuracy**
- Develops comprehensive MAE/RMSE tracking across multiple time windows
- Implements rolling window calculations (1h, 6h, 24h)
- Creates accuracy classification and trend analysis
- **Configuration**: `PREDICTION_METRICS_ENABLED`, `METRICS_WINDOW_*`

### 03_trajectory_prediction_development.ipynb
**Phase 2 Enhancement: Advanced Trajectory Prediction**
- Develops trajectory prediction with forecast integration
- Implements weather and PV forecast usage
- Creates overshoot detection algorithms
- **Configuration**: `TRAJECTORY_PREDICTION_ENABLED`, `*_FORECAST_INTEGRATION`

### 04_historical_calibration_development.ipynb
**Phase 0 Enhancement: Physics-Based Parameter Optimization**
- Develops historical calibration system
- Implements stability period detection
- Creates physics parameter optimization
- **Configuration**: `CALIBRATION_BASELINE_FILE`, `STABILITY_TEMP_CHANGE_THRESHOLD`

## Usage

1. **Development Workflow**: Use these notebooks to implement and test features
2. **Configuration**: All notebooks load Phase 2 configuration parameters
3. **Integration**: Test implementations before adding to main system
4. **Validation**: Validate features work correctly with real data

## Prerequisites

- Working ml_heating environment
- Access to InfluxDB for historical data
- Phase 2 configuration parameters enabled

## Notes

- All notebooks use corrected import functions and configuration access
- Development templates ready for actual implementation
- Integration testing included in each notebook
- Configuration issues have been resolved
