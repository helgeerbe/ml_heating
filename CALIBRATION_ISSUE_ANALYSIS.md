# Phase 5 Calibration Issue Analysis
*December 5, 2025*

## Problem Identified

You are correct to doubt the calibration result of `HEAT_LOSS_COEFFICIENT=0.250` for an old, poorly insulated house. The analysis reveals several issues with the calibration process.

## Root Cause Analysis

### 1. Constrained Parameter Bounds
Looking at the calibration data, the system has artificially constrained bounds:
```json
"parameter_bounds": {
  "heat_loss_coefficient": [0.002, 0.25]  // Maximum capped at 0.25!
}
```

**Problem**: The heat loss coefficient is artificially capped at 0.25, preventing the optimizer from finding realistic values for a poorly insulated building (which should be 3.0-8.0).

### 2. Insufficient Calibration Data
```json
"total_predictions": 3
"calibration_cycles": 3704
```

**Problem**: Only 3 actual predictions with real data, despite 3704 calibration cycles. This suggests the calibration was run on synthetic or very limited data.

### 3. Baseline Parameters Already Low
```json
"baseline_parameters": {
  "heat_loss_coefficient": 0.2044852555550354  // Already unrealistically low
}
```

**Problem**: The starting baseline was already too low (0.20), indicating previous calibrations may have been flawed.

## Expected Values for Old, Poorly Insulated House

Based on building physics, your house should have:

```bash
# Realistic parameters for old, poorly insulated house
HEAT_LOSS_COEFFICIENT=4.0-8.0     # High heat loss through poor insulation
THERMAL_TIME_CONSTANT=2.0-3.0     # Fast temperature changes (low thermal mass)
OUTLET_EFFECTIVENESS=0.15-0.3     # Your original guess was likely correct
```

## Recommended Fix Actions

### 1. Reset Parameter Bounds
Update the calibration bounds to realistic ranges:

```bash
# Remove artificial constraints
HEAT_LOSS_COEFFICIENT_MIN=1.0
HEAT_LOSS_COEFFICIENT_MAX=10.0
```

### 2. Manual Parameter Setting
For immediate use, set realistic parameters based on building type:

```bash
# For old, poorly insulated house
THERMAL_TIME_CONSTANT=2.5
HEAT_LOSS_COEFFICIENT=6.0
OUTLET_EFFECTIVENESS=0.17  # Your original estimate was good
PV_HEAT_WEIGHT=0.0005
FIREPLACE_HEAT_WEIGHT=3.0   # May need reduction
TV_HEAT_WEIGHT=0.14
```

### 3. Recalibrate with More Data
- Collect at least 48 hours of actual operational data
- Ensure diverse weather conditions
- Run calibration during periods with stable heating operation
- Remove artificial parameter constraints

## Physics Validation Check

A properly insulated house typically has:
- **Good insulation**: 1.0-2.0 heat loss coefficient  
- **Average insulation**: 2.0-4.0 heat loss coefficient
- **Poor insulation**: 4.0-8.0 heat loss coefficient

Your optimized value of 0.25 would indicate insulation better than a passive house, which contradicts your description of an old, poorly insulated building.

## Next Steps

1. **Immediate**: Use manual parameters based on building characteristics
2. **Short-term**: Remove calibration constraints and recalibrate with real data
3. **Long-term**: Validate parameters against energy bills and actual heating performance

The Phase 5 corrected physics is solid, but the calibration process needs better parameter bounds and more representative data.
