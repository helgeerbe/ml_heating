# Delta Forecast Calibration Guide

## Overview

The Delta Forecast Calibration feature improves heating prediction accuracy by locally calibrating weather forecasts based on the difference between current actual outdoor temperature and the weather service forecast. This addresses microclimate variations that global weather forecasts may miss.

## How It Works

### Basic Concept

1. **Calculate Delta**: Compare current actual outdoor temperature with the weather forecast for the current time
2. **Apply Offset**: Add this temperature difference to all future forecast values
3. **Safety Limits**: Constrain adjustments to prevent extreme corrections

### Example

```
Current actual outdoor temp: 8.5°C
Weather forecast for now: 10.0°C
Delta offset: 8.5 - 10.0 = -1.5°C

Original 4-hour forecasts: [11.0, 12.0, 13.0, 14.0]°C
Calibrated forecasts: [9.5, 10.5, 11.5, 12.5]°C
```

## Configuration

### Configuration Variables

Add these to your `config.yaml`:

```yaml
# Delta Forecast Calibration
ENABLE_DELTA_FORECAST_CALIBRATION: true  # Enable/disable the feature
DELTA_CALIBRATION_MAX_OFFSET: 10.0      # Maximum allowed offset in °C
```

### Default Values

- `ENABLE_DELTA_FORECAST_CALIBRATION`: `true` (enabled by default)
- `DELTA_CALIBRATION_MAX_OFFSET`: `10.0` (°C maximum offset)

## Benefits

### Improved Accuracy
- Corrects for local microclimate variations
- Accounts for site-specific weather patterns
- Better adapts to local topography and building effects

### Preserves Trends
- Maintains valuable forecast trend information
- Keeps relative temperature changes intact
- Preserves weather pattern timing

### Safety Features
- Configurable maximum offset limits
- Robust error handling for invalid data
- Automatic fallback to original forecasts

## Implementation Details

### Method: `get_calibrated_hourly_forecast()`

Located in `src/ha_client.py`, this method:

1. Fetches raw weather forecast data
2. Gets current outdoor temperature
3. Calculates the delta offset
4. Applies offset to all forecast values
5. Enforces safety limits
6. Returns calibrated forecast array

### Physics Integration

The `physics_features.py` module automatically uses calibrated forecasts when:
- Delta calibration is enabled in configuration
- The calibration method is available
- Falls back to original forecasts if disabled or unavailable

### Error Handling

Robust error handling includes:
- Invalid temperature data detection
- Extreme offset limiting (±10°C default)
- Automatic fallback to original forecasts
- Comprehensive logging for debugging

## Testing

### Comprehensive Test Suite

The feature includes 17 test cases covering:

- **Basic Functionality**: Positive/negative offsets, calibration math
- **Configuration**: Enabled/disabled states, offset limits
- **Edge Cases**: Zero offsets, extreme temperatures, fractional values
- **Error Handling**: Invalid data, missing sensors, extreme conditions
- **Integration**: Physics features integration, fallback behavior

Run tests with:
```bash
python -m pytest tests/test_delta_forecast_calibration.py -v
```

## Use Cases

### Ideal Scenarios

1. **Rural/Remote Locations**: Where weather stations are far from your location
2. **Microclimate Areas**: Valleys, hillsides, coastal areas with local effects
3. **Urban Heat Islands**: City locations with different temps than weather stations
4. **Seasonal Variations**: Times when local conditions consistently differ

### When to Disable

Consider disabling if:
- Your location matches weather station data very closely
- You prefer absolute forecast values
- Troubleshooting prediction accuracy issues

## Monitoring

### Log Output

When enabled, you'll see log entries like:
```
DEBUG: Delta calibration enabled: offset=-1.5°C, forecasts=[9.5, 10.5, 11.5, 12.5]
```

### Validation

Monitor heating prediction accuracy over time to verify improvement. The system should show:
- Better temperature prediction accuracy
- More responsive heating control
- Reduced over/under-heating cycles

## Troubleshooting

### Common Issues

1. **No Calibration Applied**
   - Check `ENABLE_DELTA_FORECAST_CALIBRATION` setting
   - Verify outdoor temperature sensor is working
   - Check weather forecast data availability

2. **Extreme Offsets**
   - Adjust `DELTA_CALIBRATION_MAX_OFFSET` value
   - Verify outdoor sensor accuracy
   - Check for sensor calibration issues

3. **Unexpected Results**
   - Enable debug logging to see calibration values
   - Compare original vs calibrated forecasts
   - Verify configuration settings

### Debug Steps

1. Check configuration:
   ```yaml
   ENABLE_DELTA_FORECAST_CALIBRATION: true
   DELTA_CALIBRATION_MAX_OFFSET: 10.0
   ```

2. Enable debug logging to see calibration details

3. Verify sensor data quality and accuracy

4. Run test suite to verify functionality:
   ```bash
   python -m pytest tests/test_delta_forecast_calibration.py
   ```

## Implementation History

- **Added**: December 2025
- **Purpose**: Improve local forecast accuracy for better heating predictions
- **Approach**: Delta-based calibration preserving forecast trends
- **Testing**: Comprehensive test suite with 17 test cases
- **Configuration**: Simple on/off toggle with safety limits

This feature represents a significant improvement in forecast accuracy while maintaining system reliability and providing easy configuration control.
