# Shadow Mode Learning Fix

This document describes the architectural fix for shadow mode learning to ensure correct building physics learning.

## Problem Analysis

The shadow mode learning implementation had a fundamental architectural flaw that prevented it from learning building physics correctly.

### Incorrect Implementation (Before)

**What was happening:**
1. ML calculated optimal outlet temp (e.g., 45.9°C)
2. ML predicted indoor temp based on **its own calculation**
3. Heat curve actually applied different temp (e.g., 48°C)
4. Learning compared ML's prediction vs actual indoor temp
5. **Result**: ML was evaluating its own predictions, not learning from heat curve performance

**Problem**: This was self-evaluation, not learning building physics from heat curve decisions.

### Correct Implementation (After)

**What now happens:**
1. ML calculates optimal outlet temp (e.g., 45.9°C) - **for comparison only**
2. Heat curve applies its own temp (e.g., 48°C) - **actual applied setting**
3. ML observes heat curve's setting and predicts what indoor temp it will achieve
4. Learning compares thermal model prediction vs actual indoor temp
5. **Result**: ML learns real building physics from heat curve's control decisions

## Implementation Details

### Core Logic Change

**Location**: `src/main.py` - Online Learning section (around line 390)

**Key Detection**:
```python
# Check if we're in effective shadow mode for this learning cycle
# Look at what was ACTUALLY applied vs what ML calculated
was_shadow_mode_cycle = (actual_applied_temp != last_final_temp_stored)
```

**Shadow Mode Learning**:
```python
if was_shadow_mode_cycle:
    # SHADOW MODE LEARNING (CORRECTED):
    # Predict what indoor temp the heat curve's outlet setting will achieve
    predicted_indoor_temp = wrapper.thermal_model.predict_equilibrium_temperature(
        outlet_temp=actual_applied_temp,  # Heat curve's setting
        # ... other parameters
    )
    learning_mode = "shadow_mode_hc_observation"
```

**Active Mode Learning** (unchanged):
```python
else:
    # ACTIVE MODE LEARNING (UNCHANGED):
    # Predict what indoor temp ML's outlet setting will achieve
    predicted_indoor_temp = wrapper.thermal_model.predict_equilibrium_temperature(
        outlet_temp=actual_applied_temp,  # ML's setting (same as last_final_temp_stored)
        # ... other parameters
    )
    learning_mode = "active_mode_ml_feedback"
```

### Enhanced Context Tracking

The learning context now includes detailed mode information:

```python
enhanced_prediction_context = {
    'learning_mode': learning_mode,
    'was_shadow_mode_cycle': was_shadow_mode_cycle,
    'ml_calculated_temp': last_final_temp_stored,  # What ML wanted
    'hc_applied_temp': actual_applied_temp,        # What heat curve actually set
    # ... other context
}
```

## Learning Patterns

### Shadow Mode Learning Pattern
```
Heat Curve Decision → ML Prediction → Compare with Reality → Learn Building Physics
     (48°C)      →    (21.3°C)     →   (vs 20.5°C actual) →   (Update model)
```

### Active Mode Learning Pattern  
```
ML Decision → ML Prediction → Compare with Reality → Learn Prediction Accuracy
  (45°C)    →   (20.8°C)    →  (vs 20.5°C actual) →    (Update model)
```

## Validation

### Test Coverage

**Location**: `tests/test_shadow_mode_learning_fix.py`

**Key Tests**:
1. **Shadow mode learning uses heat curve prediction**
   - Validates ML predicts based on heat curve's 48°C setting
   - Confirms prediction is 21.3°C (not ML's 20.8°C calculation)
   - Verifies learning mode is "shadow_mode_hc_observation"

2. **Active mode learning uses ML prediction** 
   - Validates ML predicts based on its own 45°C setting
   - Confirms prediction is 20.8°C 
   - Verifies learning mode is "active_mode_ml_feedback"

3. **Shadow mode cycle detection**
   - Tests logic for distinguishing shadow vs active mode cycles

### Log Messages

**Shadow Mode Learning**:
```
🔍 SHADOW MODE LEARNING: Predicting indoor temp from heat curve's 48.0°C outlet setting
```

**Active Mode Learning**:
```
🎯 ACTIVE MODE LEARNING: Verifying ML prediction accuracy for 45.0°C outlet setting
```

## Benefits

### Correct Building Physics Learning
- **Before**: ML learned from its own (unused) predictions
- **After**: ML learns from heat curve's actual control decisions
- **Result**: Better understanding of how outlet temps affect indoor temps

### Meaningful Shadow Mode
- Shadow mode now serves its intended purpose: learning without controlling
- ML observes and learns from existing heat curve performance
- Builds knowledge for eventual transition to active mode

### Improved Accuracy Timeline
- Shadow mode learning now contributes to model accuracy
- Better preparation for active mode transition
- More realistic predictions based on actual system behavior

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Calculates │    │  Heat Curve      │    │  Thermal Model  │
│   Optimal Temp  │    │  Applies Temp    │    │  Predicts       │
│   (45.9°C)      │───▶│  (48.0°C)        │───▶│  Indoor Result  │
│   [Not Applied] │    │  [Actually Used] │    │  (21.3°C)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────┐              │
                       │  Actual Indoor  │◀─────────────┘
                       │  Temperature    │
                       │  (20.5°C)       │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Learn from     │
                       │  Prediction     │
                       │  Error: 0.8°C   │
                       └─────────────────┘
```

## Version History

- **v1.0**: Initial shadow mode implementation (flawed)
- **v2.0**: Architectural fix for correct building physics learning
  - Shadow mode learns from heat curve decisions
  - Active mode continues learning from ML decisions  
  - Enhanced context tracking and logging
  - Comprehensive test validation
