# ML Heating: Thermal Model Implementation Guide

This document describes how the physics-based thermal model works in ml_heating. It's written for users with a technical background who want to understand how the system predicts and controls heating.

---

## Table of Contents

1. [Goal](#goal)
2. [Core Concept: What is Thermal Equilibrium?](#core-concept-what-is-thermal-equilibrium)
3. [The Physics Model](#the-physics-model)
4. [How Prediction Works](#how-prediction-works)
5. [Parameters Reference](#parameters-reference)
6. [Finding Good Starting Parameters (Calibration)](#finding-good-starting-parameters-calibration)
7. [Troubleshooting](#troubleshooting)
8. [How the Model Learns and Improves](#how-the-model-learns-and-improves)

---

## Goal

The ml_heating system replaces static heating curves with an adaptive, physics-based control system. 

### What It Solves

Traditional heating curves are simple lookup tables:
- "If it's 0°C outside → set outlet to 45°C"
- "If it's 10°C outside → set outlet to 35°C"

**Traditional Heating Curve (Static):**

```
Outlet                                    Fixed lookup table:
Temp                                      Outdoor → Outlet
  │                                       
55°C├────●                                  -10°C → 55°C
    │     ╲                                  0°C → 45°C
45°C├──────●───                             10°C → 35°C
    │        ╲                              20°C → 25°C
35°C├─────────●───
    │           ╲
25°C├────────────●───
    │
    └──┬──┬──┬──┬──┬──→ Outdoor Temp
     -10  0  10  20 °C

    ❌ Same curve for ALL houses
    ❌ Ignores solar, fireplace, etc.
    ❌ Needs manual seasonal adjustment
```

**ml_heating Approach (Adaptive Physics):**

```
                    ┌─────────────────────────────────────────┐
                    │           YOUR HOUSE MODEL              │
                    │                                         │
 Outdoor ──────────►│  ┌─────────────────────────────────┐   │
 Temp               │  │  Heat Balance Equation:         │   │
                    │  │                                  │   │
 Target  ──────────►│  │  T_eq = (eff×T_out + loss×T_od  │   │──► Outlet
 Indoor             │  │          + Q_external)           │   │    Temp
                    │  │         ───────────────          │   │
 PV Power ─────────►│  │           eff + loss             │   │
                    │  │                                  │   │
 Fireplace ────────►│  │  Parameters learned from YOUR   │   │
                    │  │  house's actual behavior        │   │
                    │  └─────────────────────────────────┘   │
                    │                                         │
                    │  ✓ Adapts to YOUR insulation           │
                    │  ✓ Accounts for solar/fireplace        │
                    │  ✓ Learns continuously                 │
                    └─────────────────────────────────────────┘
```

**Comparison: Same Day, Different Results**

```
Scenario: 5°C outside, sunny day (3kW solar), target 21°C

┌─────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL HEATING CURVE                                            │
│                                                                      │
│ Time:    6AM    8AM    10AM   12PM   2PM    4PM    6PM              │
│          │      │      │      │      │      │      │                │
│ Indoor   │──────┼──────┼──────┼──────┼──────┼──────┼────────        │
│ Temp   21├      └──────┴──────┴──────┴──────┘                       │
│        23├                  ████████████  ← OVERSHOOT!              │
│        22├               ███            ███                         │
│        21├─────────────██──────────────────██────── Target          │
│        20├           ██                       ██                    │
│        19├         ██                                               │
│          │       ██                                                 │
│        18├─────██                                                   │
│          │                                                          │
│ Problem: Ignores 3kW solar gain → house overheats to 23°C!         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ML_HEATING (Physics-Based)                                          │
│                                                                      │
│ Time:    6AM    8AM    10AM   12PM   2PM    4PM    6PM              │
│          │      │      │      │      │      │      │                │
│ Indoor   │──────┼──────┼──────┼──────┼──────┼──────┼────────        │
│ Temp   21├──────┴──────┴──────┴──────┴──────┴──────┴─────── Target  │
│        22├                                                          │
│        21├───────────────────────────────────────────────           │
│        20├         ██████████████████████████████████               │
│        19├       ██                                                 │
│        18├─────██                                                   │
│          │                                                          │
│ Solution: Detects solar, reduces outlet temp, maintains 21°C       │
│                                                                     │
│ Outlet: 42°C→40°C→35°C→30°C→32°C→38°C (adapts to solar)           │
└─────────────────────────────────────────────────────────────────────┘
```

**Problems with this approach:**
- No adaptation to your specific house (insulation, size, window area)
- No awareness of solar gains, fireplace, or other heat sources
- Requires manual seasonal recalibration
- Results in temperature overshoot or undershoot

### What ml_heating Does Instead

1. **Models your house physics** - Learns how your house gains and loses heat
2. **Predicts equilibrium** - Calculates what temperature your house will stabilize at
3. **Works backwards** - Finds the outlet temperature needed to reach your target
4. **Learns continuously** - Improves predictions based on actual measurements

### Success Criteria

| Metric | Good | Acceptable |
|--------|------|------------|
| Mean Absolute Error (MAE) | < 0.2°C | < 0.3°C |
| Root Mean Square Error (RMSE) | < 0.3°C | < 0.4°C |
| Learning Confidence | > 0.9 | > 0.7 |

---

## Core Concept: What is Thermal Equilibrium?

**Equilibrium** is the temperature your house will eventually stabilize at if you keep the heating running at a constant outlet temperature for a long time.

### Bathtub Analogy

Think of filling a bathtub with hot water while the drain is slightly open:

```
┌─────────────────────────────────────┐
│  HOT WATER IN                       │
│      ↓↓↓                            │
│  ┌─────────────────────────────┐    │
│  │ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ │    │  Water Level = Indoor Temp
│  │ ~ ~ ~ ~ WATER LEVEL ~ ~ ~ ~ │    │
│  │ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ │    │
│  └──────────────────────────┬──┘    │
│                             │       │
│                          DRAIN OUT  │
│                             ↓       │
└─────────────────────────────────────┘
```

- **Water flowing in** = Heat from the heat pump (outlet temperature)
- **Water flowing out** = Heat escaping through walls, windows, roof
- **Water level** = Indoor temperature

The water level **stabilizes** when inflow equals outflow. That stable level is the **equilibrium**.

### In Your House

The same principle applies:

1. **Heat Input**: Your heat pump adds heat to the house via radiators/floor heating
2. **Heat Loss**: Heat escapes through walls, windows, roof, ventilation
3. **Equilibrium**: The temperature where heat input = heat loss

If you set your heat pump to 45°C outlet and leave it running:
- Initially, heat input > heat loss → temperature rises
- As temperature rises, heat loss increases (bigger difference to outside)
- Eventually, heat input = heat loss → temperature stabilizes

**That stable temperature is the equilibrium temperature.**

### Why This Matters

The system uses equilibrium to work **backwards**:

> "I want 21°C indoors. It's 5°C outside. What outlet temperature do I need?"

The model calculates that you need approximately 42°C outlet to reach equilibrium at 21°C.

---

## The Physics Model

### The Heat Balance Equation

At equilibrium, heat input equals heat loss:

```
Heat from outlet + External heat = Heat loss to outdoor
```

Mathematically:

```
eff × (T_outlet - T_eq) + Q_external = loss × (T_eq - T_outdoor)
```

Solving for equilibrium temperature **T_eq**:

```
        eff × T_outlet + loss × T_outdoor + Q_external
T_eq = ─────────────────────────────────────────────────
                        eff + loss
```

Where:
- **T_eq** = Equilibrium temperature (what we're calculating)
- **T_outlet** = Heat pump outlet temperature (°C)
- **T_outdoor** = Outdoor temperature (°C)
- **eff** = Outlet effectiveness (how efficiently heat transfers to the room)
- **loss** = Heat loss coefficient (how quickly heat escapes)
- **Q_external** = External heat gains (solar, fireplace, electronics)

### Numerical Example 1: Basic Heating

**Given:**
- Outlet temperature: 45°C
- Outdoor temperature: 0°C
- Outlet effectiveness: 0.10
- Heat loss coefficient: 0.10
- No external heat sources (Q_external = 0)

**Calculation:**
```
T_eq = (0.10 × 45 + 0.10 × 0 + 0) / (0.10 + 0.10)
     = (4.5 + 0 + 0) / 0.20
     = 4.5 / 0.20
     = 22.5°C
```

**Result:** With 45°C outlet on a 0°C day, your house stabilizes at 22.5°C.

### Numerical Example 2: With Solar Gain

**Given:** Same as above, plus 2000W of solar PV generating heat
- PV heat weight: 0.002 °C/W

**External heat calculation:**
```
Q_external = 2000W × 0.002 °C/W = 4.0°C contribution
```

**Equilibrium calculation:**
```
T_eq = (0.10 × 45 + 0.10 × 0 + 4.0) / (0.10 + 0.10)
     = (4.5 + 0 + 4.0) / 0.20
     = 8.5 / 0.20
     = 42.5°C  ← Too hot!
```

**What happens:** The system detects this and **reduces outlet temperature** to compensate for solar gains.

### Numerical Example 3: Finding Required Outlet Temperature

**Given:**
- Target indoor: 21°C
- Outdoor: -5°C (cold day)
- No external heat

**Question:** What outlet temperature is needed?

**Rearranging the equation:**
```
T_outlet = (T_eq × (eff + loss) - loss × T_outdoor) / eff
         = (21 × 0.20 - 0.10 × (-5)) / 0.10
         = (4.2 + 0.5) / 0.10
         = 47°C
```

**Result:** Need 47°C outlet to maintain 21°C on a -5°C day.

### Physical Background

This model is based on **Newton's Law of Cooling** and **first-order thermal systems**.

**Newton's Law of Cooling:**
> The rate of heat loss is proportional to the temperature difference between the object and its surroundings.

```
dQ/dt = k × (T_inside - T_outside)
```

Your house behaves like a first-order thermal system with a **time constant** (τ). This determines how quickly the temperature responds to changes:

- **Small τ (2-4h):** House responds quickly (thin walls, lots of windows)
- **Large τ (8-12h):** House responds slowly (thick walls, good insulation, thermal mass)

For more details, see:
- [Newton's Law of Cooling (Wikipedia)](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling)
- [Thermal Time Constant (Engineering Toolbox)](https://www.engineeringtoolbox.com/thermal-time-constant-d_1482.html)

---

## How Prediction Works

### The Prediction Pipeline

When the system runs (every 10 minutes by default), it executes this sequence:

```
┌──────────────────┐
│ 1. Read Sensors  │ ← Indoor, outdoor, outlet temps
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 2. Check Target  │ ← What temperature do you want?
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 3. Binary Search │ ← Find outlet temp that gives equilibrium = target
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 4. Apply Limits  │ ← Safety bounds (min 25°C, max 65°C)
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 5. Update HA     │ ← Send to Home Assistant
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 6. Learn         │ ← Compare prediction to actual, adjust parameters
└──────────────────┘
```

### Binary Search Algorithm

The system doesn't solve the equation algebraically for outlet temperature. Instead, it uses **binary search** - a practical approach that always works:

```
Goal: Find outlet temperature that produces T_eq = 21°C

Step 1: Try outlet = 45°C → predicts T_eq = 23°C (too high)
Step 2: Try outlet = 35°C → predicts T_eq = 18°C (too low)
Step 3: Try outlet = 40°C → predicts T_eq = 20.5°C (close)
Step 4: Try outlet = 41°C → predicts T_eq = 21.0°C (found!)
```

This converges in about 10-15 iterations (< 1ms).

### Trajectory Prediction

While equilibrium tells us **where** the temperature will end up, trajectory prediction tells us **how it gets there** - the path over time.

#### Why Trajectory Matters

```
Scenario: It's 7 AM, house is at 18°C, target is 21°C

Option A: Set outlet to 50°C (aggressive)
- Reaches 21°C in 2 hours ✓
- But continues to 23°C (overshoot!) ✗
- Then slowly drops back to 21°C

Option B: Set outlet to 42°C (calculated)
- Reaches 21°C in 3 hours
- Stabilizes at 21°C (no overshoot) ✓
```

Trajectory prediction helps the system choose **Option B** - getting to target without overshooting.

#### The Physics: Exponential Approach

Temperature doesn't change linearly - it follows an **exponential curve**:

```
T(t) = T_current + (T_eq - T_current) × (1 - e^(-t/τ))
```

Where:
- **T(t)** = Temperature at time t
- **T_current** = Starting temperature
- **T_eq** = Equilibrium temperature (where it would eventually stabilize)
- **τ** = Thermal time constant (hours)
- **e** = Mathematical constant (≈2.718)

**Key insight:** Temperature changes quickly at first, then slows down as it approaches equilibrium.

#### Numerical Example: Morning Warm-Up

**Given:**
- Current indoor: 18°C (cold morning)
- Outlet temperature: 45°C
- Outdoor temperature: 2°C
- Equilibrium (calculated): 22°C
- Thermal time constant: 4 hours

**Trajectory calculation:**

| Time | Formula | Predicted Temp |
|------|---------|----------------|
| 0h | 18 + (22-18) × (1 - e^0) = 18 + 0 | **18.0°C** (start) |
| 1h | 18 + 4 × (1 - e^(-1/4)) = 18 + 4 × 0.22 | **18.9°C** |
| 2h | 18 + 4 × (1 - e^(-2/4)) = 18 + 4 × 0.39 | **19.6°C** |
| 3h | 18 + 4 × (1 - e^(-3/4)) = 18 + 4 × 0.53 | **20.1°C** |
| 4h | 18 + 4 × (1 - e^(-4/4)) = 18 + 4 × 0.63 | **20.5°C** |
| 8h | 18 + 4 × (1 - e^(-8/4)) = 18 + 4 × 0.86 | **21.5°C** |
| ∞ | 18 + 4 × (1 - 0) = 18 + 4 | **22.0°C** (equilibrium) |

**Visual representation:**

```
Temperature
    │
22°C├─────────────────────────────────────────●●●●● ← Equilibrium
    │                               ●●●●●●●●●
21°C├─ ─ ─ ─ ─ ─ ─ ─ ─ ─●●●●●●●●●●●  ← Target (21°C)
    │              ●●●●●
20°C├─ ─ ─ ─ ─●●●●●
    │       ●●●
19°C├─ ─ ●●●
    │   ●●
18°C├──●  ← Start
    │
    └──┬──┬──┬──┬──┬──┬──┬──┬───
       1  2  3  4  5  6  7  8  Hours
```

Notice how the curve is steep at first (fast warming) and flattens out (slow approach to equilibrium).

#### Using Trajectory Prediction in Code

```python
predict_thermal_trajectory(
    current_indoor=19.0,
    target_indoor=21.0,
    outlet_temp=45.0,
    outdoor_temp=5.0,
    time_horizon_hours=4
)
```

Returns:
```
{
    'trajectory': [19.5, 20.0, 20.4, 20.7],  # Temperature each hour
    'reaches_target_at': 3,                   # Reaches target in ~3 hours
    'overshoot_predicted': False,             # Won't overshoot target
    'equilibrium_temp': 22.5                  # Would stabilize at 22.5°C
}
```

#### Overshoot Detection

The system uses trajectory to prevent overshoot:

```
Situation: Target = 21°C, Equilibrium = 24°C

Trajectory: 18°C → 19.5°C → 20.8°C → 21.9°C → 22.8°C → 23.5°C → 24°C
                                     ↑
                              Crosses target here
                              
Problem: Will overshoot to 24°C before stabilizing!

Solution: Reduce outlet temperature so equilibrium = 21°C
```

#### Weather Forecast Integration

Trajectory prediction becomes more accurate with weather forecasts:

```python
predict_thermal_trajectory(
    current_indoor=20.0,
    target_indoor=21.0,
    outlet_temp=40.0,
    outdoor_temp=5.0,
    time_horizon_hours=4,
    weather_forecasts=[5, 7, 10, 12],  # Warming up during day
    pv_forecasts=[0, 500, 2000, 3000]  # Solar increasing
)
```

The model adjusts each hour's prediction based on:
- **Future outdoor temperature** (warmer outside = less heating needed)
- **Future PV production** (solar gains will help)

This allows the system to **anticipate** conditions and reduce outlet temperature proactively.

#### The 63% Rule

A useful rule of thumb: After **one time constant (τ)**, temperature reaches **63%** of the way to equilibrium.

| Time | % of change completed |
|------|----------------------|
| 1τ | 63% |
| 2τ | 86% |
| 3τ | 95% |
| 4τ | 98% |
| 5τ | 99% (essentially at equilibrium) |

**Example:** If τ = 4 hours and you need to go from 18°C to 22°C (4°C change):
- After 4 hours: 63% × 4°C = 2.5°C gained → at 20.5°C
- After 8 hours: 86% × 4°C = 3.4°C gained → at 21.4°C
- After 12 hours: 95% × 4°C = 3.8°C gained → at 21.8°C

### Gentle Trajectory Correction

When the predicted trajectory deviates from the target temperature path, the system applies **gentle additive corrections** to bring the system back on track.

#### Why Gentle Corrections Matter

Naive multiplicative correction approaches can cause outlet temperature spikes:

```
❌ AGGRESSIVE MULTIPLICATIVE APPROACH:
correction_factor = 1.0 + (temp_error * 12.0)  # 12x multiplier per degree
corrected_outlet = outlet_temp * correction_factor

Example problem:
- Outlet temp: 30°C
- Temperature error: 0.5°C (trajectory drifting)
- Correction factor: 1.0 + (0.5 * 12.0) = 7.0
- Result: 30°C * 7.0 = 210°C! (WAY too high)
```

This approach causes:
- **Outlet temperature spikes** (hitting maximum 65°C limits)
- **System stress** from dramatic temperature changes
- **Overshooting** target temperatures during recovery

#### The Gentle Additive Solution

The ml_heating system uses **gentle additive corrections** inspired by proven heat curve automation logic:

```
✅ GENTLE ADDITIVE APPROACH:
if temp_error <= 0.5°C:
    correction_amount = temp_error * 5.0   # +5°C per degree
elif temp_error <= 1.0°C:
    correction_amount = temp_error * 8.0   # +8°C per degree
else:
    correction_amount = temp_error * 12.0  # +12°C per degree

corrected_outlet = outlet_temp + correction_amount  # Additive adjustment
```

#### Correction Examples

**Scenario 1: Small trajectory error (0.5°C)**
```
Original outlet: 35°C
Temperature error: 0.5°C (house cooling too fast)
Correction: 0.5°C × 5.0 = +2.5°C
Final outlet: 35°C + 2.5°C = 37.5°C (reasonable adjustment)
```

**Scenario 2: Medium trajectory error (0.8°C)**
```
Original outlet: 40°C
Temperature error: 0.8°C (significant drift)
Correction: 0.8°C × 8.0 = +6.4°C
Final outlet: 40°C + 6.4°C = 46.4°C (moderate adjustment)
```

**Scenario 3: Large trajectory error (1.5°C)**
```
Original outlet: 32°C
Temperature error: 1.5°C (major deviation, e.g., window opened)
Correction: 1.5°C × 12.0 = +18°C
Final outlet: 32°C + 18°C = 50°C (strong but bounded correction)
```

#### Heat Curve Alignment

The correction factors (5°C, 8°C, 12°C per degree) are based on successful heat curve automation patterns already proven in real deployments. This ensures trajectory corrections feel natural and don't conflict with existing heating control experience.

#### Open Window Adaptation

The gentle correction system excels at handling sudden thermal disturbances:

```
Timeline: Open Window Scenario

T+0min:   Window opens → rapid heat loss detected
T+5min:   Trajectory prediction shows temperature will drop 2°C below target
          Gentle correction: +12°C outlet adjustment (reasonable)
T+10min:  System applies increased heating, begins temperature recovery
T+15min:  Adaptive learning adjusts heat loss coefficient upward
T+45min:  Window closes → heat loss returns to normal
T+50min:  System detects reduced heat loss, begins readjustment
T+60min:  Parameters return to pre-window values, temperature stable

Key benefits:
✓ No outlet temperature spikes (stayed under 55°C)
✓ Appropriate response magnitude (12°C increase vs 7x multiplication)
✓ Automatic parameter adaptation
✓ Smooth restabilization when disturbance ends
```

#### Implementation Details

**Trajectory Verification Phase**: The system tracks whether the predicted thermal trajectory is on course to reach the target temperature. If deviations exceed the gentle correction thresholds, additive adjustments are applied.

**Forecast Integration**: During binary search phases, the system stores current forecast data (`_current_features`) to ensure trajectory verification has access to real PV and temperature forecast information, not static assumptions.

**Fallback Safety**: If forecast data is unavailable during trajectory verification, the system gracefully falls back to equilibrium-only control without trajectory corrections, ensuring robust operation.

This gentle approach maintains effective thermal control while preventing the outlet temperature spikes and system stress that characterized earlier aggressive correction methods.

---

## Parameters Reference

### Core Thermal Parameters

| Parameter | Symbol | Unit | Description | Default | Typical Range |
|-----------|--------|------|-------------|---------|---------------|
| `thermal_time_constant` | τ | hours | How quickly the house responds to temperature changes | 4.0 | 2 - 12 |
| `heat_loss_coefficient` | loss | °C/unit | Rate of heat loss per degree temperature difference | 0.10 | 0.05 - 0.5 |
| `outlet_effectiveness` | eff | dimensionless | How efficiently outlet heat transfers to room | 0.10 | 0.05 - 0.3 |

### External Heat Source Weights

| Parameter | Unit | Description | Default | Typical Range |
|-----------|------|-------------|---------|---------------|
| `pv_heat_weight` | °C/W | Temperature rise per watt of solar | 0.002 | 0.001 - 0.01 |
| `fireplace_heat_weight` | °C | Temperature contribution when fireplace is on | 5.0 | 2 - 10 |
| `tv_heat_weight` | °C | Temperature contribution from electronics | 0.2 | 0.1 - 2 |

### Learning Parameters

| Parameter | Unit | Description | Default |
|-----------|------|-------------|---------|
| `adaptive_learning_rate` | - | How fast parameters adjust | 0.05 |
| `learning_confidence` | - | Current confidence in predictions | 3.0 (initial) |
| `recent_errors_window` | samples | Number of predictions for error analysis | 10 |

### Understanding the Parameters

#### Thermal Time Constant (τ)

**What it means:** How many hours it takes for your house to respond to a temperature change.

| House Type | Typical τ | Characteristics |
|------------|-----------|-----------------|
| Modern lightweight | 2-4h | Thin walls, large windows, quick response |
| Standard construction | 4-6h | Brick/concrete, moderate insulation |
| Heavy thermal mass | 8-12h | Stone walls, underfloor heating, slow response |

**How it affects behavior:**
- Low τ → Temperature changes quickly, more responsive to outlet adjustments
- High τ → Temperature changes slowly, needs earlier anticipation

#### Heat Loss Coefficient

**What it means:** How quickly heat escapes from your house.

| Insulation Quality | Typical Value | Meaning |
|-------------------|---------------|---------|
| Poor (old house) | 0.3 - 0.5 | Loses heat quickly |
| Average | 0.1 - 0.2 | Normal heat loss |
| Excellent (passive house) | 0.05 - 0.1 | Retains heat well |

#### Outlet Effectiveness

**What it means:** How efficiently the heat pump's outlet temperature translates to room heating.

| Heating System | Typical Value | Why |
|----------------|---------------|-----|
| Underfloor heating | 0.05 - 0.1 | Large area, low temperature |
| Large radiators | 0.1 - 0.15 | Good heat distribution |
| Small radiators | 0.15 - 0.25 | Less efficient distribution |

---

## Finding Good Starting Parameters (Calibration)

### Automatic Calibration

The system includes an automatic calibration tool that analyzes your historical data:

```bash
# Run from ml_heating directory
python -c "from src.physics_calibration import run_phase0_calibration; run_phase0_calibration()"
```

This process:

1. **Fetches 4 weeks of historical data** from InfluxDB
2. **Filters for stable periods** (when temperature wasn't changing much)
3. **Excludes blocking states** (DHW heating, defrost, etc.)
4. **Optimizes parameters** using scipy's L-BFGS-B algorithm
5. **Validates results** on held-out data
6. **Saves calibrated baseline** for the model to use

### What "Stable Periods" Means

The calibration only uses data where the system was in equilibrium:

| Included | Excluded |
|----------|----------|
| Temperature stable (< 0.1°C change over 30 min) | DHW heating active |
| Outlet temperature stable | Defrost cycle running |
| Normal operation | Fireplace state changed |
| Clear sensor data | Missing data points |

This ensures the calibration learns from valid equilibrium data, not transient states.

### Manual Parameter Estimation

If you want to estimate parameters manually before calibration:

#### 1. Estimate Thermal Time Constant

Wait for your house to be at equilibrium (stable for 2+ hours), then:
- Change the outlet temperature by 5°C
- Measure how long it takes to reach 63% of the new equilibrium
- That time is approximately τ

**Example:**
- House at 21°C (stable)
- Increase outlet by 5°C
- New equilibrium would be ~22°C
- After 4 hours, temperature reaches 21.6°C (63% of the way)
- τ ≈ 4 hours

#### 2. Estimate Heat Loss Coefficient

On a cold day with no heating:
- Note the indoor temperature drop rate
- Heat loss ≈ temperature drop × building mass factor

**Rough guide:**
- 0.5°C drop per hour with heating off → high heat loss (0.3+)
- 0.2°C drop per hour → moderate heat loss (0.15)
- 0.1°C drop per hour → low heat loss (0.08)

#### 3. Start Conservative

When in doubt, start with these conservative values:

```yaml
thermal_time_constant: 4.0   # Middle of range
heat_loss_coefficient: 0.10  # Average insulation
outlet_effectiveness: 0.10   # Standard heating system
```

The adaptive learning will adjust these over time.

### Environment Variables for Parameters

All parameters can be overridden via environment variables:

```bash
# Core thermal parameters
export THERMAL_TIME_CONSTANT=6.0
export HEAT_LOSS_COEFFICIENT=0.08
export OUTLET_EFFECTIVENESS=0.12

# External heat sources
export PV_HEAT_WEIGHT=0.003
export FIREPLACE_HEAT_WEIGHT=4.0
export TV_HEAT_WEIGHT=0.3

# Learning parameters
export ADAPTIVE_LEARNING_RATE=0.05
export LEARNING_CONFIDENCE=3.0
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Temperature Oscillates Around Target

**Symptoms:**
- Indoor temperature swings ±1-2°C around target
- Outlet temperature changes frequently

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Outlet effectiveness too high | Reduce `OUTLET_EFFECTIVENESS` by 20-30% |
| Thermal time constant too low | Increase `THERMAL_TIME_CONSTANT` |
| Learning rate too aggressive | Reduce `ADAPTIVE_LEARNING_RATE` |

**Recommended fix:**
```bash
export OUTLET_EFFECTIVENESS=0.08  # Reduce from default 0.10
export MAX_TEMP_CHANGE_PER_CYCLE=1  # Limit to 1°C change per cycle
```

#### Issue: House Never Reaches Target Temperature

**Symptoms:**
- Indoor temp stays 2-3°C below target
- Outlet temperature at maximum (65°C)

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Heat loss coefficient too low | Increase `HEAT_LOSS_COEFFICIENT` |
| Outlet effectiveness too low | Increase `OUTLET_EFFECTIVENESS` |
| Actual heat loss underestimated | Run calibration with more data |

**Quick test:**
```bash
# Temporarily increase effectiveness
export OUTLET_EFFECTIVENESS=0.15
```

#### Issue: Overshoot After Heating Period

**Symptoms:**
- Temperature exceeds target by 1-2°C
- Then slowly decreases

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Thermal time constant too high | Reduce `THERMAL_TIME_CONSTANT` |
| Solar gains not accounted for | Ensure PV sensor is configured |
| Trajectory prediction disabled | Enable `TRAJECTORY_PREDICTION_ENABLED=true` |

#### Issue: Erratic Behavior After DHW/Defrost

**Symptoms:**
- Wild temperature predictions after water heating
- Takes 30+ minutes to stabilize

**Cause:** The system needs time to recover after blocking events.

**Solution:** Increase grace period:
```bash
export GRACE_PERIOD_MAX_MINUTES=45  # Default is 30
```

#### Issue: Parameters Drift to Extreme Values

**Symptoms:**
- `heat_loss_coefficient` becomes very small (< 0.02) or large (> 0.5)
- `outlet_effectiveness` hits bounds

**Cause:** Adaptive learning overcorrecting based on unusual data.

**Solutions:**

1. Reset to calibrated baseline:
```bash
# Delete learned adjustments, keep calibrated values
rm /data/thermal_state.json
# Restart service to reload calibrated baseline
systemctl restart ml_heating
```

2. Reduce learning aggressiveness:
```bash
export MAX_LEARNING_RATE=0.1  # Default is 0.2
export LEARNING_CONFIDENCE=2.0  # Start lower
```

### Diagnostic Commands

#### Check Current Parameters

```bash
# View thermal state file
cat /data/thermal_state.json | python -m json.tool
```

#### View Recent Predictions

```bash
# Check logs for prediction accuracy
journalctl -u ml_heating --since "1 hour ago" | grep "prediction"
```

#### Validate Model Behavior

```python
# Quick physics check
from src.thermal_equilibrium_model import ThermalEquilibriumModel

model = ThermalEquilibriumModel()
print(f"Thermal time constant: {model.thermal_time_constant}h")
print(f"Heat loss coefficient: {model.heat_loss_coefficient}")
print(f"Outlet effectiveness: {model.outlet_effectiveness}")

# Test prediction
eq_temp = model.predict_equilibrium_temperature(
    outlet_temp=45, outdoor_temp=5, current_indoor=21
)
print(f"45°C outlet, 5°C outdoor → equilibrium: {eq_temp:.1f}°C")
```

### When to Re-Calibrate

Run calibration again if:

- You've made significant changes to your house (insulation, windows)
- Heating system was modified (new radiators, underfloor heating added)
- Predictions are consistently off by > 1°C
- Seasonal transition (winter → summer usage changes)

```bash
# Run full calibration
python -c "from src.physics_calibration import run_phase0_calibration; run_phase0_calibration()"
```

### Getting Help

If issues persist:

1. Check the logs: `journalctl -u ml_heating -f`
2. Enable debug mode: `export DEBUG=1`
3. Review the calibration file: `cat /opt/ml_heating/calibrated_baseline.json`
4. Check sensor data in InfluxDB for gaps or anomalies

---

## How the Model Learns and Improves

This section explains the adaptive learning mechanism - **when** the model updates its parameters and **how** it knows which parameter to adjust.

### When Does Learning Happen?

The model learns **every 10 minutes** (configurable via `CYCLE_INTERVAL_MINUTES`). Here's the learning cycle:

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. PREDICT: Calculate expected indoor temperature                │
│    → "Based on current outlet temp, I predict 21.5°C in 30 min"  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. WAIT: Let the heating system run for one cycle (10 min)       │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. MEASURE: Read actual indoor temperature                       │
│    → "Actual temperature is 21.8°C"                              │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. CALCULATE ERROR: Compare prediction to reality                │
│    → Error = 21.8 - 21.5 = +0.3°C (underpredicted)              │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. ADJUST PARAMETERS: Update model based on error direction      │
│    → "I need to account for more heat - maybe PV contribution?"  │
└──────────────────────────────────────────────────────────────────┘
```

### How Does It Know Which Parameter to Change?

The model uses **gradient descent** - a mathematical technique that finds which parameter adjustment would most reduce the prediction error.

#### The Gradient Concept (Simplified)

For each parameter, the model asks: *"If I increase this parameter slightly, does my prediction error go up or down?"*

```
Example: Learning the PV heat weight

Current state:
- PV power: 3000W
- pv_heat_weight: 0.002 °C/W (so PV contributes 6°C)
- Prediction error: +0.5°C (actual was hotter than predicted)

Question: "Did I underestimate the PV effect?"

Test by simulation:
1. Try pv_heat_weight = 0.0025 → predicts 7.5°C from PV → prediction closer to actual
2. Try pv_heat_weight = 0.0015 → predicts 4.5°C from PV → prediction further from actual

Result: Gradient is negative (increasing pv_heat_weight reduces error)
Action: Increase pv_heat_weight slightly
```

#### The Finite Difference Method

The model calculates gradients using **finite differences** - a practical way to measure sensitivity:

```
                    predict(param + ε) - predict(param - ε)
Gradient = Error × ────────────────────────────────────────
                                    2 × ε
```

Where ε (epsilon) is a small step size for testing.

**Numerical Example: PV Weight Gradient**

Given:
- Current pv_heat_weight = 0.002
- ε = 0.0001 (small test step)
- Current prediction error = +0.3°C

Calculation:
```
1. Predict with pv_heat_weight = 0.0021 → predicts 21.7°C
2. Predict with pv_heat_weight = 0.0019 → predicts 21.3°C
3. Finite difference = (21.7 - 21.3) / (2 × 0.0001) = 2000
4. Gradient = 0.3 × 2000 = 600 (positive → need to increase PV weight)
```

### Which Parameters Get Updated?

The model only updates parameters when there's **sufficient evidence**:

| Condition | Learning Behavior |
|-----------|-------------------|
| PV was producing power during prediction | PV weight may be adjusted |
| Fireplace was on during prediction | Fireplace weight may be adjusted |
| TV was on during prediction | TV weight may be adjusted |
| Normal heating only | Core thermal parameters may be adjusted |

**Important:** The model only learns about heat sources **that were active** during the prediction period.

#### Example: Why PV Weight Only Changes When PV is Active

```
Scenario 1: Nighttime (no PV)
- PV power: 0W
- Any pv_heat_weight × 0W = 0°C contribution
- Changing pv_heat_weight has NO EFFECT on prediction
- Gradient = 0 → pv_heat_weight stays unchanged

Scenario 2: Sunny day (PV active)
- PV power: 4000W  
- pv_heat_weight × 4000W = significant contribution
- Changing pv_heat_weight CHANGES the prediction
- Gradient ≠ 0 → pv_heat_weight gets updated
```

### Learning Rate and Confidence

The model doesn't make large jumps - it adjusts gradually:

```
new_parameter = old_parameter - (learning_rate × gradient)
```

**Learning rate** determines how big each adjustment is:

| Situation | Learning Rate | Behavior |
|-----------|---------------|----------|
| Large errors (>2°C) | Higher (×3) | Learn faster to fix big problems |
| Medium errors (0.5-2°C) | Normal | Standard learning |
| Small errors (<0.5°C) | Lower | Fine-tuning only |
| Parameters very stable | Reduced | Avoid over-correction |

**Confidence** tracks how well the model is performing:

- Starts at 3.0 (initial confidence)
- Increases when predictions improve
- Decreases when predictions worsen
- Affects how aggressively parameters are updated

### Practical Example: Learning Fireplace Effect

**Day 1: First fireplace use**
```
Before fireplace: Indoor = 20°C, predicted equilibrium = 21°C
Fireplace turned on
Actual result: Indoor reaches 23°C (2°C higher than predicted)

Model thinks: "Error = +2°C, fireplace was on"
Gradient calculation shows: fireplace_heat_weight should increase
Update: fireplace_heat_weight: 5.0 → 5.3°C
```

**Day 2: Second fireplace use**
```
Model now predicts higher with fireplace
Predicted with fireplace: 21.5°C
Actual result: 22°C (0.5°C error)

Smaller adjustment: fireplace_heat_weight: 5.3 → 5.4°C
```

**Day 5: Model has learned**
```
Predicted with fireplace: 22°C
Actual result: 22.1°C (0.1°C error - excellent!)

Minimal adjustment needed - model has converged
```

### Calibration vs. Online Learning

The system uses **two types of learning**:

| Type | When | What it does |
|------|------|--------------|
| **Calibration** | Once (at setup) | Analyzes 4 weeks of data to find optimal starting parameters |
| **Online Learning** | Every 10 minutes | Fine-tunes parameters based on recent predictions |

**Calibration** gives you a good starting point. **Online learning** keeps the model accurate as conditions change (seasons, house modifications, etc.).

---

## Home Assistant Sensors (Refactored Schema)

The ml_heating system exports metrics to Home Assistant through a clean, non-redundant sensor architecture. Each sensor has a distinct purpose with no overlapping attributes.

**Key Design Principles:**
- ✅ **Zero Redundancy**: Each attribute appears in exactly one sensor
- ✅ **Clear Separation**: Each sensor has a distinct monitoring purpose  
- ✅ **Enhanced Insights**: Time-windowed analysis and error distribution
- ✅ **User-Friendly**: Meaningful thresholds and interpretable values

### Overview of Sensors

| Sensor | Purpose | State Value | Unit | Monitoring Focus |
|--------|---------|-------------|------|-----------------|
| `sensor.ml_heating_state` | Operational status | Status code (0-7) | state | Real-time prediction info |
| `sensor.ml_heating_learning` | Learning confidence | Confidence score (0-5) | confidence | Adaptive learning progress |
| `sensor.ml_model_mae` | Prediction accuracy | Mean Absolute Error | °C | Time-windowed accuracy |
| `sensor.ml_model_rmse` | Error distribution | Root Mean Square Error | °C | Error analysis & bias |
| `sensor.ml_prediction_accuracy` | Control quality | Good control % | % | 24h heating performance |

---

### sensor.ml_heating_state

**Purpose**: Current operational status and real-time prediction information.

**State**: Numeric status code (0-7)

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | OK | Prediction completed successfully |
| 1 | Low Confidence | Model confidence below threshold |
| 2 | Blocked | DHW/Defrost/Disinfection active |
| 3 | Network Error | Cannot reach Home Assistant API |
| 4 | Missing Sensors | Critical sensor data unavailable |
| 5 | (reserved) | - |
| 6 | Heating Off | Climate entity not in heat/auto mode |
| 7 | Model Error | Exception during prediction |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `state_description` | string | Human-readable status message |
| `suggested_temp` | float | Model's calculated outlet temperature (°C) |
| `final_temp` | float | Applied temperature after clamping (°C) |
| `predicted_indoor` | float | Predicted indoor temperature (°C) |
| `temperature_error` | float | Current deviation from target (°C) |
| `last_prediction_time` | timestamp | When the prediction was made |
| `blocking_reasons` | list | Active blocking entities (if blocked) |
| `heating_state` | string | Current climate entity state |

---

### sensor.ml_heating_learning

**Purpose**: Learning status, model confidence, and learned thermal parameters.

**State**: Learning confidence score (0.0 - 5.0)

**Confidence Interpretation**:

| Score | Quality | Meaning |
|-------|---------|---------|
| ≥ 4.0 | Excellent | Fully trusted predictions, model has converged |
| 3.0 - 4.0 | Good | Reliable predictions, still fine-tuning |
| 2.0 - 3.0 | Fair | Use with caution, model still learning |
| < 2.0 | Poor | Consider recalibration or check data quality |

**Attributes**:

| Attribute | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `thermal_time_constant` | float | Learned τ parameter (hours) | 2-12 hours |
| `heat_loss_coefficient` | float | Learned heat loss rate | 0.05-0.5 |
| `outlet_effectiveness` | float | Learned heating efficiency | 0.05-0.3 |
| `cycle_count` | int | Total learning cycles completed | - |
| `parameter_updates` | int | Times parameters were adjusted | - |
| `model_health` | string | Health assessment based on learning confidence | excellent/good/fair/poor |
| `learning_progress` | float | Progress toward maturity (0-1) | ≥0.8 = mature |
| `is_improving` | bool | Whether accuracy trend is positive | - |
| `improvement_percentage` | float | MAE improvement trend (%) | >0 = improving |
| `total_predictions` | int | Total predictions tracked | - |

**Model Health Calculation**:

The `model_health` attribute is derived from the `learning_confidence` score using these thresholds:

| Learning Confidence | Model Health | Meaning |
|-------------------|-------------|---------|
| ≥ 4.0 | excellent | Model has fully converged, predictions highly reliable |
| 3.0 - 4.0 | good | Model is stable, predictions are reliable |
| 2.0 - 3.0 | fair | Model still learning, use predictions with caution |
| < 2.0 | poor | Model needs more data or recalibration |

**Parameters Considered**:
- **Primary**: `learning_confidence` score (0.0 - 5.0)
- **Contributing factors to confidence**:
  - Prediction accuracy over recent cycles
  - Parameter stability (how much thermal parameters are changing)
  - Error trend (improving vs. degrading)
  - Number of learning cycles completed

**Learning Progress Interpretation**:
- **0.0 - 0.3**: Initial learning phase, parameters may shift significantly
- **0.3 - 0.6**: Active learning, parameters stabilizing  
- **0.6 - 0.8**: Near-mature, fine-tuning adjustments
- **0.8 - 1.0**: Mature model, minimal parameter changes needed

---

### sensor.ml_model_mae

**Purpose**: Primary accuracy metric - Mean Absolute Error in temperature predictions.

**State**: All-time MAE (°C)

**MAE Interpretation**:

| MAE | Quality | Action |
|-----|---------|--------|
| < 0.1°C | Excellent | No action needed |
| 0.1 - 0.2°C | Good | Normal operation |
| 0.2 - 0.3°C | Acceptable | Monitor for trends |
| ≥ 0.3°C | Poor | Investigate cause, consider recalibration |

**Attributes**:

| Attribute | Type | Description | Good Threshold |
|-----------|------|-------------|----------------|
| `mae_1h` | float | MAE over last hour | < 0.2°C |
| `mae_6h` | float | MAE over last 6 hours | < 0.2°C |
| `mae_24h` | float | MAE over last 24 hours | < 0.2°C |
| `trend_direction` | string | Error trend | "improving"/"stable"/"degrading" |
| `prediction_count` | int | Predictions in calculation | - |
| `last_updated` | timestamp | When metrics were updated | - |

---

### sensor.ml_model_rmse

**Purpose**: Error distribution metric - Root Mean Square Error penalizes large errors more than MAE.

**State**: All-time RMSE (°C)

**RMSE Interpretation**:

| RMSE | Quality | Meaning |
|------|---------|---------|
| < 0.15°C | Excellent | Consistently accurate predictions |
| 0.15 - 0.25°C | Good | Acceptable error distribution |
| 0.25 - 0.4°C | Fair | Some larger errors occurring |
| ≥ 0.4°C | Poor | Unpredictable errors, investigate |

**Attributes**:

| Attribute | Type | Description | Good Threshold |
|-----------|------|-------------|----------------|
| `recent_max_error` | float | Max error in last 10 predictions | < 0.5°C |
| `std_error` | float | Standard deviation of errors | < 0.2°C |
| `mean_bias` | float | Average signed error (systematic bias) | Near 0 |
| `prediction_count` | int | Predictions in calculation | - |
| `last_updated` | timestamp | When metrics were updated | - |

---

### sensor.ml_prediction_accuracy

**Purpose**: Easy-to-understand accuracy percentages based on 24-hour window.

**State**: Good control percentage (%) - predictions within ±0.2°C

**Accuracy Interpretation**:

| Good Control % | Quality | Meaning |
|----------------|---------|---------|
| ≥ 90% | Excellent | Outstanding temperature control |
| 80 - 90% | Good | Reliable control, occasional deviations |
| 70 - 80% | Acceptable | Room for improvement |
| < 70% | Poor | Needs investigation |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `perfect_accuracy_pct` | float | % predictions with exactly 0.0°C error |
| `tolerable_accuracy_pct` | float | % predictions within 0.0-0.2°C error |
| `poor_accuracy_pct` | float | % predictions with >0.2°C error |
| `prediction_count_24h` | int | Predictions in 24-hour window |
| `excellent_all_time_pct` | float | % within ±0.1°C (all-time) |
| `good_all_time_pct` | float | % within ±0.5°C (all-time) |
| `last_updated` | timestamp | When metrics were updated |

---

### Monitoring Dashboard Example

Create a Lovelace dashboard card to monitor ML Heating:

```yaml
type: entities
title: ML Heating Status
entities:
  - entity: sensor.ml_heating_state
    name: Status
  - entity: sensor.ml_heating_learning
    name: Learning Confidence
  - entity: sensor.ml_model_mae
    name: Prediction Error (MAE)
  - entity: sensor.ml_prediction_accuracy
    name: Control Quality
  - type: attribute
    entity: sensor.ml_heating_learning
    attribute: model_health
    name: Model Health
  - type: attribute
    entity: sensor.ml_heating_learning
    attribute: cycle_count
    name: Learning Cycles
```

### Alerting Thresholds

Recommended alert thresholds for monitoring:

| Metric | Warning | Critical |
|--------|---------|----------|
| `ml_heating_state` | = 1 (Low Confidence) | ≥ 3 (Network/Sensor/Model errors) |
| `ml_heating_learning` | < 2.5 | < 2.0 |
| `ml_model_mae` | > 0.25°C | > 0.35°C |
| `ml_model_rmse` | > 0.3°C | > 0.45°C |
| `ml_prediction_accuracy` | < 75% | < 65% |

**Note**: State 2 (Blocked) is normal operation during DHW/Defrost and should not trigger alerts.

---

## Summary

The ml_heating system uses a physics-based approach instead of black-box machine learning:

1. **Heat balance equation** predicts equilibrium temperature
2. **Binary search** finds the outlet temperature needed for your target
3. **Adaptive learning** continuously improves parameter estimates
4. **Calibration** optimizes parameters using your historical data

The key parameters are:
- **Thermal time constant (τ)** - How fast your house responds
- **Heat loss coefficient** - How quickly heat escapes
- **Outlet effectiveness** - How well the heating system transfers heat

Start with calibration, let the system learn, and adjust parameters if you see oscillation, undershoot, or overshoot.

Monitor system health using the Home Assistant sensors described above, and set up alerts for the recommended thresholds to catch issues early.
