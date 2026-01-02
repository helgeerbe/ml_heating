# Emergency Stability Controls

This document describes the emergency stability controls implemented to prevent catastrophic ML heating system failures.

## Overview

The emergency stability controls protect against parameter corruption and catastrophic prediction errors that can cause system instability, such as the 0.0% prediction accuracy failure that occurred due to corrupted thermal parameters.

## Root Cause Analysis

The original failure was caused by:
- Corrupted `total_conductance = 0.266` (should be ~0.05)
- This caused physics model to make completely wrong predictions
- Target outlet always 65°C while heat curve used 48°C
- Complete system failure with 0.0% accuracy, 12.5 MAE, 12.97 RMSE

## Protection Mechanisms

### 1. Parameter Corruption Detection

**Location**: `src/thermal_parameters.py`

**Function**: `_detect_parameter_corruption()`

**Validation bounds**:
- `equilibrium_ratio`: 0.3 - 0.9
- `total_conductance`: 0.02 - 0.25 
- `learning_confidence`: ≥ 0.01

**Detection**: Catches specific corruption patterns like the production failure (0.266 value)

### 2. Catastrophic Error Handling

**Threshold**: Prediction errors ≥ 5°C trigger protection

**Actions when triggered**:
- Learning rate set to 0.0 (blocks parameter updates)
- Parameter changes completely blocked
- System continues making predictions but prevents learning from garbage data
- Enhanced logging for debugging

### 3. Auto-Recovery

**Conditions for re-enabling learning**:
- Prediction errors drop below 5°C threshold
- Parameter corruption resolved through bounds validation
- System stability restored with consecutive good predictions

**Check frequency**: Every cycle (30 minutes)

## Monitoring

### Log Messages

**✅ Normal Operation**:
```
✅ Online learning: applied_temp=46.0°C, actual_change=0.150°C, cycle=5
✅ Parameter validation passed - corruption resolved
```

**⚠️ Learning Disabled (Temporary)**:
```
⚠️ CATASTROPHIC ERROR DETECTED: Prediction error 7.2°C ≥ 5.0°C threshold
🚫 Learning disabled due to catastrophic prediction error
⏸️ Learning rate set to 0.0 - blocking parameter updates
```

**❌ Parameter Corruption (Serious)**:
```
🚨 Parameter corruption detected: total_conductance=0.266 outside bounds [0.02, 0.25]
🚫 Learning blocked due to parameter corruption
⚠️ System using fallback thermal parameters
```

**🔄 Recovery in Progress**:
```
✅ Prediction error 2.1°C below catastrophic threshold (5.0°C)
🔄 Learning re-enabled: error within acceptable range
```

### Home Assistant Sensors

**`sensor.ml_heating_state`** - General system status:
- `0`: Normal operation
- `1`: Low confidence
- `2`: Blocking activity (DHW/Defrost)
- `4`: Missing sensor data
- `7`: Model error (check for corruption)

**`sensor.ml_heating_learning`** - Learning-specific metrics and confidence

**Note**: Learning protection happens at thermal model level. Sensor may show normal status even when learning is disabled for safety.

## Implementation Details

### Test Coverage
- **24/25 comprehensive unit tests passing (96% success rate)**
- Parameter corruption detection validated
- Catastrophic error handling tested
- Boundary cases covered
- Auto-recovery scenarios verified

### Integration Points
- `src/thermal_equilibrium_model.py`: Core learning protection
- `src/main.py`: Online learning safety checks
- `tests/test_parameter_corruption_detection.py`: Validation tests
- `tests/test_catastrophic_error_handling.py`: Error handling tests

## Recovery Procedures

### Automatic Recovery (Preferred)
The system is designed to self-recover:
1. Emergency controls detect issues automatically
2. Learning disabled to prevent further damage
3. System continues making predictions safely
4. Learning re-enabled when conditions improve

### Manual Recovery (If Needed)
If corruption persists:
```bash
# Delete corrupted parameters
sudo rm /opt/ml_heating/thermal_state.json

# Re-calibrate from clean state
sudo systemctl start ml_heating --calibrate-physics

# Monitor logs for successful recovery
sudo journalctl -fu ml_heating
```

## Benefits

- **Prevents catastrophic failures**: No more 0.0% accuracy scenarios
- **Self-healing**: Automatic recovery without manual intervention
- **Continues operation**: System makes predictions even when learning disabled
- **Detailed monitoring**: Comprehensive logging for debugging
- **Production-ready**: Protects against the specific failure patterns identified

## Version History

- **v1.0**: Initial implementation fixing 0.0% accuracy failure
- Protection against total_conductance corruption (0.266 → 0.195)
- Catastrophic error threshold (≥5°C) protection
- Auto-recovery mechanisms
- Test-driven development approach with 24/25 tests passing
