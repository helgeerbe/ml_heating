# Configuration Parameter Fixes - January 3, 2026

## Overview

This document describes the critical configuration parameter fixes implemented on January 3, 2026, to resolve "parameter out of bounds" warnings and ensure stable system operation.

## Issues Identified

### 1. Learning Rate Parameters Out of Bounds
**Problem**: Learning rate parameters exceeded safe operational bounds defined by thermal parameter validation.

**Specific Issues**:
- `MIN_LEARNING_RATE=0.05` - Was 50x higher than safe maximum of 0.001
- `MAX_LEARNING_RATE=0.1` - Exceeded validated bounds (should be ≤0.01)

**Impact**: Could cause unstable learning behavior and parameter oscillation.

### 2. Physics Parameters Outside Validation Ranges
**Problem**: Thermal physics parameters were below validated bounds.

**Specific Issues**:
- `OUTLET_EFFECTIVENESS=0.10` - Below validated range of 0.5-1.0

**Impact**: Physics model calculations would be outside tested parameter space.

### 3. System Behavior Parameters Too Extreme
**Problem**: Some system behavior parameters were set to extreme values.

**Specific Issues**:
- `MAX_TEMP_CHANGE_PER_CYCLE=20` - Dangerously high (could cause temperature spikes)
- `GRACE_PERIOD_MAX_MINUTES=10` - Too short for proper system transitions

**Impact**: System instability and poor transition handling.

## Solutions Implemented

### 1. Learning Rate Corrections
```bash
# Before (problematic)
MIN_LEARNING_RATE=0.05
MAX_LEARNING_RATE=0.1

# After (safe)
MIN_LEARNING_RATE=0.001
MAX_LEARNING_RATE=0.01
```

### 2. Physics Parameter Corrections
```bash
# Before (out of bounds)
OUTLET_EFFECTIVENESS=0.10

# After (within validated range)
OUTLET_EFFECTIVENESS=0.8
```

### 3. System Behavior Optimizations
```bash
# Before (extreme values)
MAX_TEMP_CHANGE_PER_CYCLE=20
GRACE_PERIOD_MAX_MINUTES=10

# After (balanced values)
MAX_TEMP_CHANGE_PER_CYCLE=10
GRACE_PERIOD_MAX_MINUTES=30
```

## Files Updated

### Production Configuration Files
1. **`.env`** - Production environment configuration
2. **`.env_sample`** - Template with safe example values and bound annotations
3. **`ml_heating/config.yaml`** - Stable Home Assistant addon configuration  
4. **`ml_heating_dev/config.yaml`** - Development Home Assistant addon configuration

### Configuration Consistency
All configuration files now contain:
- **Safe Parameter Values**: Within validated operational bounds
- **Bound Annotations**: Comments indicating safe parameter ranges
- **Consistent Values**: Same safe parameters across all deployment modes

## Validation Results

### System Log Analysis
After implementing fixes, system logs show:
- ✅ **No "parameter out of bounds" warnings**
- ✅ **Stable learning confidence** at 3.0 (healthy range)
- ✅ **Proper thermal parameter validation** (effectiveness=0.4051, time_constant=4.00)
- ✅ **Binary search convergence** in 7 iterations with ±0.030°C precision

### Shadow Mode Operation
System correctly operating in shadow mode:
- **Heat Curve Observation**: 56.0°C (actual heating control)
- **ML Prediction**: 52.2°C (4°C lower, more efficient)
- **Learning Pattern**: Correctly learning building physics from heat curve decisions

## Parameter Bounds Reference

### Learning Rate Bounds
```
MIN_LEARNING_RATE: 0.0001 - 0.01
MAX_LEARNING_RATE: 0.0001 - 0.01
ADAPTIVE_LEARNING_RATE: 0.01 - 0.2
```

### Physics Parameter Bounds  
```
OUTLET_EFFECTIVENESS: 0.5 - 1.0
THERMAL_TIME_CONSTANT: 4.0 - 96.0
HEAT_LOSS_COEFFICIENT: 0.005 - 0.25
```

### System Behavior Bounds
```
MAX_TEMP_CHANGE_PER_CYCLE: 0.5 - 10.0
GRACE_PERIOD_MAX_MINUTES: 10 - 120
CYCLE_INTERVAL_MINUTES: 5 - 60
```

## Impact and Benefits

### Immediate Benefits
- **Eliminated Configuration Warnings**: No more parameter validation errors
- **Stable Learning Behavior**: Learning rates within tested operational bounds
- **Proper Physics Operation**: Thermal parameters within validated ranges
- **Responsive Heating Control**: 10°C max change allows fast heating while preventing instability

### Long-term Benefits
- **Consistent Operation**: All deployment modes use same safe parameters
- **Future-Proof Configuration**: Safe examples prevent other users from hitting same issues
- **Maintained Performance**: System performance preserved while ensuring safety
- **Simplified Troubleshooting**: Consistent configuration reduces debugging complexity

## User Action Required

### For Existing Installations
If you experience "parameter out of bounds" warnings, update your `.env` file with the corrected values shown in this document.

### For New Installations
Use the updated `.env_sample` file as a template - it now contains safe parameter values with bound annotations.

## Technical Notes

### Validation Logic Location
Parameter bounds are enforced in:
- `src/thermal_parameters.py` - Thermal parameter validation
- `src/thermal_equilibrium_model.py` - Physics model bounds checking
- Configuration schema definitions in addon `config.yaml` files

### Testing Approach
All fixes were validated through:
- **System restart testing** - Verified no configuration warnings
- **Shadow mode validation** - Confirmed proper learning operation
- **Log analysis** - Verified stable system behavior

## Conclusion

The configuration parameter fixes resolve all identified parameter bound violations while maintaining system performance and responsiveness. The system now operates within validated parameter ranges with consistent configuration across all deployment modes.

**Status**: ✅ **COMPLETE** - All configuration issues resolved, system operational with no warnings.
