# Outlet Effectiveness Calibration Guide
*Phase 5 Corrected Physics Implementation*

## Current Configuration Analysis

Your `OUTLET_EFFECTIVENESS=0.15` is currently a guessed value that needs proper calibration with the corrected physics formula. Let's determine the actual effectiveness systematically.

## Corrected Physics Formula Impact

**Before (incorrect):** `heat_input = outlet_temp * effectiveness`
**After (correct):** `heat_input = max(0, outlet_temp - current_indoor) * effectiveness`

**Key Difference:** You now only get heating benefit when outlet temperature exceeds indoor temperature, which is physically correct.

## Systematic Outlet Effectiveness Calibration

Since your current value is a guess, let's determine the actual effectiveness through data analysis:

### Method 1: Historical Data Analysis (Recommended)
Look for steady-state periods in your data where:
- Indoor temperature was stable (±0.2°C for 2+ hours)
- Outlet temperature was constant
- No external heat sources active (fireplace, significant PV)
- Outdoor temperature relatively stable

**Formula:** `effectiveness = (indoor_final - outdoor) × heat_loss_coeff ÷ (outlet - indoor_initial)`

**Example Calculation:**
```
Stable period data:
- Outdoor: 5°C
- Indoor initial: 20°C
- Indoor final (equilibrium): 22°C  
- Outlet: 45°C
- Heat loss coefficient: 1.0

effectiveness = (22 - 5) × 1.0 ÷ (45 - 20) = 17 ÷ 25 = 0.68
```

### Method 2: Adaptive Learning Calibration
Start with a conservative range and let the system learn:

```bash
# Start with mid-range values for heat pump systems:
OUTLET_EFFECTIVENESS=0.4
HEAT_LOSS_COEFFICIENT=2.0
THERMAL_TIME_CONSTANT=5.0

# Enable aggressive adaptive learning:
ADAPTIVE_LEARNING_RATE=0.1
MIN_LEARNING_RATE=0.05
MAX_LEARNING_RATE=0.3
```

### Method 3: Bracket Testing
Test different effectiveness values systematically:

**Week 1:** `OUTLET_EFFECTIVENESS=0.2` (Conservative)
**Week 2:** `OUTLET_EFFECTIVENESS=0.4` (Moderate) 
**Week 3:** `OUTLET_EFFECTIVENESS=0.6` (Aggressive)

Monitor MAE/RMSE for each period and choose the best-performing value.

## Realistic Outlet Effectiveness Ranges

Based on heat pump system types:
```bash
# Air-to-water heat pump with radiators: 0.3 - 0.7
# Air-to-water with underfloor heating: 0.4 - 0.8
# Ground source with radiators: 0.4 - 0.8  
# Ground source with underfloor: 0.5 - 0.9

# Conservative starting point for unknown systems:
OUTLET_EFFECTIVENESS=0.4
```

## Recommended Starting Parameters

For calibration with unknown effectiveness, start with these conservative values:

### 1. Heat Loss Coefficient (Currently 0.2)
```bash
# Your current value is very low - suggests excellent insulation
# Recommended ranges based on building type:

# Excellent insulation (passive house): 0.8 - 1.2
# Good insulation (modern): 1.5 - 2.5  
# Average insulation: 3.0 - 4.0
# Poor insulation: 5.0 - 8.0

# For your system, try starting with:
HEAT_LOSS_COEFFICIENT=1.2
```

### 2. Thermal Time Constant (Currently 4.0)
```bash
# Your current value is reasonable
# Adjust based on building thermal mass:

# Heavy thermal mass (concrete/brick): 8-12 hours
# Medium thermal mass (wood frame): 4-6 hours  
# Light thermal mass: 2-3 hours

# Keep your current value or try:
THERMAL_TIME_CONSTANT=6.0
```

### 3. External Heat Source Weights
```bash
# With corrected physics, these may need adjustment:

# PV contribution (per watt)
PV_HEAT_WEIGHT=0.001  # Reduced from 0.002

# Fireplace contribution  
FIREPLACE_HEAT_WEIGHT=3.0  # Reduced from 5.0

# TV/Electronics contribution
TV_HEAT_WEIGHT=0.15  # Keep similar to current
```

## Physics Impact Example

**Scenario:** Outlet 45°C, Indoor 20°C, Effectiveness 0.15

**Old Formula:** `heat_input = 45 × 0.15 = 6.75 units`
**New Formula:** `heat_input = max(0, 45-20) × 0.15 = 25 × 0.15 = 3.75 units`

**Result:** The corrected formula gives ~44% less heat input, requiring recalibration.

## Recommended Calibration Steps

### Step 1: Update Configuration
```bash
# Edit your .env file:
HEAT_LOSS_COEFFICIENT=1.2
THERMAL_TIME_CONSTANT=6.0
PV_HEAT_WEIGHT=0.001
FIREPLACE_HEAT_WEIGHT=3.0
```

### Step 2: Enable Adaptive Learning
```bash
# Ensure these are enabled for auto-tuning:
ADAPTIVE_LEARNING_RATE=0.05
HYBRID_LEARNING_ENABLED=true
PREDICTION_METRICS_ENABLED=true
```

### Step 3: Monitor Performance
- Run system for 48-72 hours with new parameters
- Check MAE/RMSE metrics in dashboard
- Allow adaptive learning to fine-tune values
- Target prediction accuracy <0.2°C

### Step 4: Manual Fine-Tuning (if needed)
- If predictions consistently high: increase `HEAT_LOSS_COEFFICIENT`
- If predictions consistently low: decrease `HEAT_LOSS_COEFFICIENT` 
- If response too slow: decrease `THERMAL_TIME_CONSTANT`
- If response too fast: increase `THERMAL_TIME_CONSTANT`

## Expected Behavior Changes

With corrected physics:
1. **More Realistic Heat Calculations**: No heating when outlet ≤ indoor
2. **Better Temperature Differential Response**: System responds properly to actual heating potential
3. **Improved Equilibrium Predictions**: More accurate steady-state calculations
4. **Physically Correct Edge Cases**: Handles cold start and low outlet scenarios correctly

## Validation Tests

After implementing changes, verify:
- [ ] Predictions within ±0.2°C during stable periods
- [ ] No heating predicted when outlet < indoor + 5°C
- [ ] Reasonable equilibrium temperatures (indoor + 1-5°C)
- [ ] MAE/RMSE showing improvement over 24-48 hours

## Physical Constraints Validation

Your system should respect these bounds:
- `outlet_temp >= indoor_temp + 5°C` (minimum useful heating)
- `heat_input >= 0` (no negative heating from outlets)
- `equilibrium_temp >= outdoor_temp` (basic physics)
- `heat_loss_coefficient > 0` (buildings always lose heat)

The corrected physics ensures all these constraints are naturally satisfied.
