# Adaptive Learning Master Plan - ML Heating System

**Created:** December 4, 2025  
**Status:** 🔄 IN PROGRESS  
**Priority:** CRITICAL - Core Project Objective  
**Goal:** "Rock solid indoor temperature" through intelligent adaptive learning

---

## Executive Summary

This document outlines the comprehensive implementation plan to transform the ML Heating System from a fixed-parameter physics model into a fully adaptive, intelligent heating controller that learns your specific house's thermal characteristics while maintaining physical constraints.

### Core Philosophy
> "After calibration, have a production-ready system trained to your specific house."

The system achieves **rock solid temperature control** through:
- ✅ **House-Specific Historical Calibration**: 648 hours of your data optimizes ALL parameters
- ✅ **Two-Stage Learning**: Historical optimization + ongoing adaptive corrections  
- ✅ **Hybrid Learning Strategy**: Learn during stable periods, skip chaotic transitions
- ✅ **Multi-Parameter Optimization**: Thermal params + heat source weights (PV, fireplace, TV)
- ✅ **Bounded Correction Learning**: Safe ±20% corrections around proven baselines
- ✅ **Production-Ready from Day 1**: No "learning period" - starts with your house's parameters
- ✅ **Intelligent Learning Timing**: High weight during equilibrium, low weight during transitions, skip during chaos

---

## Current State Analysis (December 4, 2025)

### 🔴 Critical Issues Identified

| Issue | Location | Impact |
|-------|----------|--------|
| Adaptive learning DISABLED | `thermal_equilibrium_model.py:43` | No learning happening |
| Trajectory methods EMPTY | `thermal_equilibrium_model.py:430-442` | No prediction capability |
| Enhanced trajectory uses wrong model | `enhanced_trajectory.py:13` | Code unusable |
| No MAE/RMSE tracking | Removed during migration | No accuracy monitoring |

### ✅ Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| Binary search outlet optimization | ✅ Working | In model_wrapper.py |
| Physics equilibrium calculation | ✅ Working | predict_equilibrium_temperature() |
| State persistence | ✅ Working | save_state/load_state |
| HA integration | ✅ Working | Sensors being updated |
| Multi-heat source physics | ✅ Implemented | multi_heat_source_physics.py |
| Fireplace adaptive learning | ✅ Implemented | adaptive_fireplace_learning.py |

---

## Implementation Phases

### Phase 0: House-Specific Calibration 🏠 NEW FOUNDATION  
**Timeline:** Week -1 (Pre-Production)  
**Goal:** Create production-ready system trained to your specific house

#### Task 0.1: Comprehensive Historical Parameter Optimization
- [ ] Analyze 648 hours of historical data with stability filtering
- [ ] Optimize ALL thermal parameters simultaneously (outlet_effectiveness, heat_loss_coefficient, thermal_time_constant)
- [ ] Optimize ALL heat source weights (PV, fireplace, TV/electronics)
- [ ] Filter for equilibrium periods only (temp change < 0.1°C per 30min)
- [ ] Use multi-parameter optimization with physical bounds
- [ ] **Unit Tests Required:**
  - [ ] Test historical data stability filtering logic
  - [ ] Test multi-parameter optimization with mock data
  - [ ] Test parameter bounds enforcement during optimization
  - [ ] Test optimization objective function calculation
  - [ ] Test calibration with different data quality scenarios
  - [ ] Test edge cases (insufficient stable periods, extreme values)
  - [ ] Achieve >85% test coverage for historical calibration module

**Implementation Details:**
```python
def comprehensive_thermal_calibration(historical_data_648h):
    """
    Calibrate ALL parameters using stable periods from historical data.
    
    Parameters optimized:
    - outlet_effectiveness: 0.3-0.8 (vs manual 0.550)
    - heat_loss_coefficient: 0.02-0.15 (vs manual 0.05) 
    - thermal_time_constant: 12-48h (vs manual 24h)
    - pv_heat_weight: 0.0005-0.005 (vs manual 0.0015)
    - fireplace_heat_weight: 1.0-6.0 (vs manual 2.5)
    - tv_heat_weight: 0.1-1.5 (vs manual 0.5)
    """
    
    # Step 1: Filter for learning-appropriate periods
    stable_periods = filter_stable_periods(
        historical_data_648h,
        temp_change_threshold=0.1,  # °C per 30min
        min_duration=30,            # 30+ minute stable periods
        no_external_disturbances=True
    )
    
    # Step 2: Multi-parameter optimization  
    def optimization_objective(params):
        model = create_model_with_params(params)
        total_error = 0
        
        for period in stable_periods:
            predicted = model.predict_equilibrium_temperature(
                outlet_temp=period['outlet_temp'],
                outdoor_temp=period['outdoor_temp'],
                pv_power=period['pv_power'],
                fireplace_on=period['fireplace_on'],
                tv_on=period['tv_on']
            )
            error = abs(predicted - period['actual_indoor'])
            total_error += error
            
        return total_error / len(stable_periods)  # MAE
    
    # Step 3: Bounded optimization
    optimal_params = scipy.optimize.minimize(
        optimization_objective,
        initial_guess=current_config_values,
        bounds=parameter_bounds,
        method='L-BFGS-B'
    )
    
    return optimal_params

# Expected improvement: 30-40% reduction in MAE vs manual parameters
```

#### Task 0.2: Validation and Baseline Creation
- [ ] Validate optimized parameters on held-out historical data
- [ ] Create calibrated baseline configuration file
- [ ] Generate calibration quality report
- [ ] Export enhanced HA metrics showing calibration results
- [ ] **Unit Tests Required:**
  - [ ] Test parameter validation on held-out data
  - [ ] Test baseline configuration file creation and loading
  - [ ] Test calibration quality metrics calculation
  - [ ] Test edge cases (validation failure scenarios)
  - [ ] Test HA metrics export for calibration results
  - [ ] Integration test with historical calibration workflow
  - [ ] Achieve >85% test coverage for validation module

**Calibrated Baseline Results:**
```python
# Before calibration (manual estimates)
manual_parameters = {
    'outlet_effectiveness': 0.550,
    'pv_heat_weight': 0.0015,
    'fireplace_heat_weight': 2.5,
    'historical_mae': 0.45  # Estimated
}

# After calibration (house-specific optimization)
calibrated_parameters = {
    'outlet_effectiveness': 0.485,      # 12% more heating needed
    'pv_heat_weight': 0.0022,           # 47% more effective than expected
    'fireplace_heat_weight': 3.8,       # 52% more effective than expected  
    'tv_heat_weight': 0.3,              # 40% less heat than expected
    'thermal_time_constant': 28.5,      # 19% more thermal mass
    'historical_mae': 0.28              # 38% improvement!
}
```

### Phase 1: Foundation & Hybrid Learning 🎯 PRODUCTION START
**Timeline:** Week 1  
**Goal:** Deploy with house-specific parameters + enable intelligent adaptive learning

#### Task 1.1: MAE/RMSE Tracking System
- [ ] Create `PredictionMetrics` class with rolling windows
- [ ] Track MAE over 1h, 6h, 24h windows
- [ ] Track RMSE over same windows
- [ ] Track prediction accuracy (% within ±0.3°C)
- [ ] Persist metrics in state file
- [ ] **Unit Tests Required:**
  - [ ] Test rolling window management (add/trim entries)
  - [ ] Test MAE calculation accuracy with known data
  - [ ] Test RMSE calculation accuracy with known data
  - [ ] Test edge cases (empty windows, single sample)
  - [ ] Test state persistence and loading
  - [ ] Test prediction accuracy percentage calculation
  - [ ] Achieve >90% test coverage for PredictionMetrics class

**Implementation Details:**
```python
class PredictionMetrics:
    """Rolling window prediction accuracy tracking."""
    
    def __init__(self):
        self.predictions_1h = []   # Last 12 samples (5min each)
        self.predictions_6h = []   # Last 72 samples
        self.predictions_24h = []  # Last 288 samples
    
    def record_prediction(self, predicted: float, actual: float, timestamp: datetime):
        error = actual - predicted
        self.predictions_1h.append({'error': error, 'ts': timestamp})
        # Trim old entries...
    
    @property
    def mae_1h(self) -> float:
        errors = [abs(p['error']) for p in self.predictions_1h]
        return np.mean(errors) if errors else 0.0
    
    @property
    def rmse_1h(self) -> float:
        errors = [p['error']**2 for p in self.predictions_1h]
        return np.sqrt(np.mean(errors)) if errors else 0.0
```

#### Task 1.2: Enhanced HA Metrics Export
- [ ] Add MAE/RMSE to sensor.ml_heating_state
- [ ] Add current learned parameters display
- [ ] Add adaptive learning status indicator
- [ ] Add trajectory prediction accuracy
- [ ] Create sensor.ml_heating_learning for detailed metrics
- [ ] **Unit Tests Required:**
  - [ ] Test HA sensor attribute formatting
  - [ ] Test metric value conversion and validation
  - [ ] Test sensor update integration
  - [ ] Test error handling for missing metrics
  - [ ] Test metric aggregation logic
  - [ ] Integration test with model_wrapper sensor updates
  - [ ] Achieve >85% test coverage for HA metrics module

**HA Sensor Attributes:**
```yaml
sensor.ml_heating_state:
  # Existing
  confidence: 3.45
  suggested_temp: 45.0
  final_temp: 45.0
  
  # NEW: Prediction Accuracy
  mae_1h: 0.23
  mae_24h: 0.31
  rmse_1h: 0.28
  prediction_accuracy_pct: 87.5
  
  # NEW: Learned Parameters
  outlet_effectiveness: 0.55
  heat_loss_coefficient: 0.25
  thermal_time_constant: 24.0
  
  # NEW: Learning Status
  adaptive_learning_active: true
  learning_updates_24h: 45
  parameter_drift_rate: 0.02

sensor.ml_heating_learning:
  # Detailed learning metrics
  pv_effectiveness_learned: 0.0015
  fireplace_effectiveness_learned: 2.5
  trajectory_mae_1h: 0.45
  trajectory_mae_4h: 0.82
  convergence_time_avg: 2.3
  overshoot_count_24h: 2
  undershoot_count_24h: 1
```

#### Task 1.3: Implement Trajectory Prediction
- [ ] Implement `predict_thermal_trajectory()` in ThermalEquilibriumModel
- [ ] Include weather forecast integration
- [ ] Include PV forecast integration
- [ ] Add trajectory accuracy tracking
- [ ] Add overshoot detection from trajectory
- [ ] **Unit Tests Required:**
  - [ ] Test trajectory calculation with known scenarios
  - [ ] Test thermal dynamics (exponential approach to equilibrium)
  - [ ] Test weather forecast integration logic
  - [ ] Test PV forecast integration logic
  - [ ] Test overshoot detection accuracy
  - [ ] Test edge cases (no forecasts, extreme values)
  - [ ] Test trajectory accuracy metrics calculation
  - [ ] Achieve >90% test coverage for trajectory prediction

**Implementation Details:**
```python
def predict_thermal_trajectory(
    self,
    current_indoor: float,
    target_indoor: float, 
    outlet_temp: float,
    outdoor_temp: float,
    weather_forecasts: List[float] = None,  # [1h, 2h, 3h, 4h]
    pv_forecasts: List[float] = None,       # [1h, 2h, 3h, 4h]
    time_horizon_hours: int = 4,
    **external_sources
) -> Dict:
    """
    Predict temperature trajectory over time horizon.
    
    Returns:
        {
            'trajectory': [21.1, 21.3, 21.5, 21.6],  # Predicted temps
            'times': [1, 2, 3, 4],                    # Hours ahead
            'reaches_target_at': 2.3,                 # Hours to target
            'overshoot_predicted': False,
            'max_predicted': 21.6,
            'equilibrium_temp': 21.8
        }
    """
    trajectory = []
    current_temp = current_indoor
    
    for hour in range(time_horizon_hours):
        # Use forecasts if available, else current values
        future_outdoor = weather_forecasts[hour] if weather_forecasts else outdoor_temp
        future_pv = pv_forecasts[hour] if pv_forecasts else external_sources.get('pv_power', 0)
        
        # Calculate equilibrium at this future point
        equilibrium = self.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=future_outdoor,
            pv_power=future_pv,
            fireplace_on=external_sources.get('fireplace_on', 0),
            tv_on=external_sources.get('tv_on', 0)
        )
        
        # Apply thermal dynamics (exponential approach to equilibrium)
        time_constant_hours = self.thermal_time_constant
        approach_factor = 1 - np.exp(-1.0 / time_constant_hours)
        delta = (equilibrium - current_temp) * approach_factor
        
        current_temp = current_temp + delta
        trajectory.append(current_temp)
    
    # Analyze trajectory
    reaches_target_at = None
    for i, temp in enumerate(trajectory):
        if abs(temp - target_indoor) < 0.3:
            reaches_target_at = i + 1
            break
    
    return {
        'trajectory': trajectory,
        'times': list(range(1, time_horizon_hours + 1)),
        'reaches_target_at': reaches_target_at,
        'overshoot_predicted': max(trajectory) > target_indoor + 0.5,
        'max_predicted': max(trajectory),
        'equilibrium_temp': trajectory[-1] if trajectory else current_indoor
    }
```

#### Task 1.4: Hybrid Adaptive Learning with Weighted Periods
- [ ] Implement correction-based learning around calibrated baselines
- [ ] Add hybrid learning strategy with weighted periods
- [ ] Implement stability period classification
- [ ] Add bounded parameter correction enforcement
- [ ] Track learning phase and weights in HA metrics
- [ ] **Unit Tests Required:**
  - [ ] Test learning phase classification (high_confidence/low_confidence/skip)
  - [ ] Test stability metrics calculation (temp change rate, std dev)
  - [ ] Test external disturbance detection
  - [ ] Test weighted gradient calculation
  - [ ] Test correction bounds enforcement (±20%)
  - [ ] Test baseline loading and application
  - [ ] Test learning event recording for HA metrics
  - [ ] Test predictable transition detection
  - [ ] Achieve >95% test coverage for hybrid learning class

**Implementation Details - Hybrid Learning Strategy:**
```python
class HybridConstrainedAdaptiveLearning:
    """
    Two-stage learning: Calibrated baselines + weighted online corrections.
    
    Key insights:
    1. Start with house-specific calibrated baselines (Phase 0)
    2. Learn corrections around proven baselines (not absolute values)
    3. Use hybrid timing: High weight during stable periods, low weight during transitions
    4. Skip learning during chaotic periods to avoid "transient trap"
    """
    
    def __init__(self):
        # Stage 1: Load calibrated baselines (from Phase 0 historical analysis)
        calibration = load_calibrated_baseline()
        self.base_outlet_effectiveness = calibration.get('outlet_effectiveness', 0.485)  # House-specific
        self.base_heat_loss_coefficient = calibration.get('heat_loss_coefficient', 0.045)
        self.base_thermal_time_constant = calibration.get('thermal_time_constant', 28.5)
        
        # Stage 2: Online correction learning (bounded around calibrated baselines)
        self.outlet_effectiveness_correction = 0.0    # ∈ [-0.2, +0.2] around calibrated baseline
        self.heat_loss_correction = 0.0               # ∈ [-0.2, +0.2] around calibrated baseline  
        self.time_constant_correction = 0.0           # ∈ [-0.2, +0.2] around calibrated baseline
        
        # Hybrid learning weights
        self.learning_weights = {
            'high_confidence': 1.0,     # Stable periods: temp change < 0.1°C per 30min
            'low_confidence': 0.3,      # Controlled transitions with momentum compensation
            'skip': 0.0                 # Chaotic periods: rapid changes, disturbances
        }
        
        # Stability tracking for learning phase classification
        self.stability_window = []
        self.min_stability_samples = 6  # 30 minutes of stable data
    
    @property
    def outlet_effectiveness(self) -> float:
        """Current effectiveness = calibrated_baseline × (1 + online_correction)"""
        return self.base_outlet_effectiveness * (1 + self.outlet_effectiveness_correction)
    
    def classify_learning_phase(self, context: Dict, history: List[float]) -> Tuple[str, float]:
        """
        Classify current period for appropriate learning strategy.
        
        Returns: (phase_name, learning_weight)
        """
        if len(history) < self.min_stability_samples:
            return 'skip', 0.0
        
        # Calculate stability metrics
        temp_change_rate = abs(history[-1] - history[-6]) / 0.5  # Change per 30min
        temp_std = np.std(history[-6:])  # Recent stability
        
        # Check for external disturbances  
        external_disturbances = (
            context.get('fireplace_just_changed', False) or
            context.get('target_temp_just_changed', False) or
            context.get('dhw_active', False)
        )
        
        time_since_change = context.get('time_since_last_change_min', 0)
        
        # Primary Learning: Near-Equilibrium Periods (High Weight)
        if (temp_change_rate < 0.1 and 
            temp_std < 0.15 and
            time_since_change > 30 and 
            not external_disturbances):
            return 'high_confidence', self.learning_weights['high_confidence']
        
        # Secondary Learning: Controlled Transition Periods (Lower Weight)
        elif (temp_change_rate < 0.3 and
              not external_disturbances and
              self._in_predictable_transition(context)):
            return 'low_confidence', self.learning_weights['low_confidence']
        
        # No Learning: Chaotic Periods (Zero Weight)
        else:
            return 'skip', self.learning_weights['skip']
    
    def _in_predictable_transition(self, context: Dict) -> bool:
        """Check if we're in a predictable transition that can be learned from."""
        # Gradual target changes (not rapid)
        gradual_target_change = (
            context.get('target_change_rate', 0) < 0.5 and  # <0.5°C change
            context.get('time_since_target_change', 60) > 15  # >15min ago
        )
        
        # Steady outdoor conditions (no weather fronts)
        steady_outdoor = (
            abs(context.get('outdoor_temp_change_1h', 0)) < 2.0  # <2°C change per hour
        )
        
        return gradual_target_change and steady_outdoor
    
    def update_from_feedback(self, predicted: float, actual: float, context: Dict):
        """
        Update corrections using hybrid learning strategy with weighted periods.
        
        Only learns corrections around calibrated baseline during appropriate periods.
        """
        # Classify current learning phase
        learning_phase, learning_weight = self.classify_learning_phase(context, self.stability_window)
        
        if learning_phase == 'skip':
            return  # No learning during chaotic periods
        
        error = actual - predicted
        
        # Calculate weighted correction gradient
        base_learning_rate = 0.005  # Conservative learning rate
        gradient = error * base_learning_rate * learning_weight
        
        # Update outlet effectiveness correction (most important parameter)
        if abs(error) > 0.1:  # Only learn from significant errors
            self.outlet_effectiveness_correction = np.clip(
                self.outlet_effectiveness_correction + gradient,
                -0.2, 0.2  # ±20% around calibrated baseline
            )
        
        # Track learning for HA metrics
        self._record_learning_event(learning_phase, learning_weight, error, gradient)
    
    def _record_learning_event(self, phase: str, weight: float, error: float, gradient: float):
        """Record learning event for HA metrics tracking."""
        self.last_learning_event = {
            'phase': phase,
            'weight': weight,
            'error': error,
            'gradient': gradient,
            'timestamp': time.time(),
            'outlet_effectiveness_current': self.outlet_effectiveness
        }
```

**Enhanced HA Metrics for Hybrid Learning:**
```yaml
sensor.ml_heating_learning:
  # Hybrid learning status
  current_learning_phase: "high_confidence"        # high_confidence/low_confidence/skip
  stability_score: 0.92                            # 0.0-1.0 stability rating
  learning_weight_applied: 1.0                     # Weight used in last update
  stable_period_duration_min: 45                   # Minutes in current stable period
  
  # Learning distribution (24h)
  learning_updates_24h:
    high_confidence: 78                            # Updates during stable periods
    low_confidence: 12                             # Updates during transitions
    skipped: 23                                    # Skipped due to chaos
  
  # Calibrated baselines vs current values
  outlet_effectiveness_baseline: 0.485             # From historical calibration
  outlet_effectiveness_current: 0.512             # Current with online corrections
  outlet_effectiveness_correction_pct: 5.6        # % correction learned online
  
  # Learning quality metrics
  learning_efficiency: 87.5                       # % accuracy improvement from learning
  correction_stability: 0.89                      # How stable the learned corrections
  false_learning_prevention: 94.2                 # % chaotic periods correctly skipped
```

### Phase 2: Configuration & Implementation 🔧
**Timeline:** Week 2  
**Goal:** Clean up configs and implement missing parameters

#### Task 2.1: Configuration Parameter Cleanup
- [ ] Remove retired parameters from all config files
- [ ] Add new adaptive learning parameters to .env_sample
- [ ] Add new parameters to src/config.py
- [ ] Update addon config.yaml files
- [ ] **Retired Parameters to Remove:**
  - [ ] CHARGING_MODE_THRESHOLD (deprecated 3-phase control)
  - [ ] MAINTENANCE_MODE_THRESHOLD (deprecated 3-phase control)
  - [ ] Old experimental thermal parameters

#### Task 2.2: New Adaptive Learning Configuration
- [ ] Add hybrid learning strategy parameters
- [ ] Add MAE/RMSE tracking configuration
- [ ] Add trajectory prediction settings
- [ ] Add calibration baseline parameters
- [ ] **New Parameters to Add:**
  ```env
  # Hybrid Learning Strategy
  HYBRID_LEARNING_ENABLED=true
  STABILITY_CLASSIFICATION_ENABLED=true
  HIGH_CONFIDENCE_WEIGHT=1.0
  LOW_CONFIDENCE_WEIGHT=0.3
  LEARNING_PHASE_SKIP_WEIGHT=0.0
  
  # MAE/RMSE Tracking
  PREDICTION_METRICS_ENABLED=true
  METRICS_WINDOW_1H=12
  METRICS_WINDOW_6H=72
  METRICS_WINDOW_24H=288
  PREDICTION_ACCURACY_THRESHOLD=0.3
  
  # Trajectory Prediction
  TRAJECTORY_PREDICTION_ENABLED=true
  WEATHER_FORECAST_INTEGRATION=true
  PV_FORECAST_INTEGRATION=true
  OVERSHOOT_DETECTION_ENABLED=true
  
  # Historical Calibration
  CALIBRATION_BASELINE_FILE=/data/calibrated_baseline.json
  STABILITY_TEMP_CHANGE_THRESHOLD=0.1
  MIN_STABLE_PERIOD_MINUTES=30
  OPTIMIZATION_METHOD=L-BFGS-B
  ```

#### Task 2.3: Complete Notebook Reorganization & Implementation
- [ ] **Phase A: Archive Migration - Fresh Start Approach**
  - [ ] Move ALL current notebooks to `notebooks/archive/legacy-notebooks/`
  - [ ] Preserve existing archive subdirectories (adaptive-learning-development, benchmarking-iterations, etc.)
  - [ ] Create clean new directory structure: monitoring/, analysis/, development/
  
- [ ] **Phase B: New Organized Structure Implementation**
  ```
  notebooks/
  ├── archive/
  │   ├── legacy-notebooks/          # ALL current numbered notebooks (00-23)
  │   ├── adaptive-learning-development/  # Existing
  │   ├── benchmarking-iterations/   # Existing  
  │   ├── debug-notebooks/          # Existing
  │   └── validation-experiments/   # Existing
  │
  ├── monitoring/                   # NEW: Real-time system monitoring
  ├── analysis/                    # NEW: Deep-dive investigations  
  └── development/                 # NEW: Feature development & testing
  ```

- [ ] **Phase C: Create Working Notebooks with Proper JSON Structure**
  
  **📊 monitoring/ - Real-time System Monitoring (4 notebooks)**
  - [ ] `adaptive_learning_status.ipynb` - Current learning phase, parameter drift, quick health check
  - [ ] `prediction_accuracy_monitor.ipynb` - Live MAE/RMSE tracking, prediction vs actual plots
  - [ ] `system_behavior_overview.ipynb` - Temperature control performance, trajectory accuracy
  - [ ] `learning_phase_dashboard.ipynb` - Learning event timeline, stability periods classification
  
  **🔍 analysis/ - Deep-dive Investigations (4 notebooks)**  
  - [ ] `thermal_equilibrium_deep_dive.ipynb` - Physics parameter evolution, calibration quality
  - [ ] `trajectory_prediction_analysis.ipynb` - Trajectory accuracy by horizon, forecast impact
  - [ ] `learning_efficiency_investigation.ipynb` - Learning phase accuracy, false learning prevention
  - [ ] `multi_source_heat_analysis.ipynb` - PV/Fireplace/TV effectiveness, seasonal patterns
  
  **⚙️ development/ - Feature Development & Testing (3 notebooks)**
  - [ ] `historical_calibration_workbench.ipynb` - Parameter optimization testing, baseline creation
  - [ ] `hybrid_learning_development.ipynb` - Learning strategy testing, correction bounds validation
  - [ ] `influxdb_export_testing.ipynb` - Data export schema validation, performance analysis

- [ ] **Phase D: Support Infrastructure Updates**
  - [ ] Update `notebook_imports.py` for new structure and fix all import issues
  - [ ] Create `notebook_utilities.py` - Common functions for all notebooks
  - [ ] Create `monitoring_config.py` - Configuration for monitoring notebooks
  - [ ] Test each notebook independently to ensure proper JSON format and working imports

- [ ] **Quality Assurance Requirements:**
  - [ ] All notebooks must have correct Jupyter JSON structure (no malformed cells)
  - [ ] All notebooks must import successfully using updated notebook_imports.py
  - [ ] All notebooks must execute without errors in both addon and standalone environments
  - [ ] All notebooks must follow consistent formatting and documentation standards

#### Task 2.4: InfluxDB Export Schema Implementation
- [ ] Define complete behavioral tracking dataset
- [ ] Add export functions to model classes
- [ ] Test data export functionality
- [ ] **Core Schema Tables:**
  ```python
  ml_learning_metrics = {
      "measurement": "ml_learning",
      "fields": {
          "current_learning_phase": "string",
          "stability_score": "float", 
          "learning_weight_applied": "float",
          "stable_period_duration_min": "int",
          "mae_1h": "float",
          "mae_24h": "float", 
          "rmse_1h": "float",
          "prediction_accuracy_pct": "float"
      }
  }
  ```

### Phase 3: Optimization & Polish
**Timeline:** Week 3  
**Goal:** Fine-tuning and advanced features

#### Task 3.1: Seasonal Adaptation
- [ ] Track parameter variations by season
- [ ] Implement seasonal correction factors
- [ ] Add summer/winter mode detection

#### Task 3.2: Advanced Monitoring Dashboard
- [ ] Create comprehensive Grafana dashboard
- [ ] Add prediction vs actual charts
- [ ] Add learning progress visualization
- [ ] Add trajectory prediction charts

#### Task 3.3: Documentation & Testing
- [ ] Update all memory bank documentation
- [ ] Create comprehensive test suite
- [ ] Performance benchmarking
- [ ] User documentation

---

## Success Criteria

### Phase 1 Success Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| MAE (24h average) | < 0.3°C | From metrics tracking |
| Prediction accuracy | > 85% within ±0.3°C | From metrics |
| HA metrics visible | 100% | All new metrics in HA |
| Adaptive learning | Active | Learning updates occurring |

### Phase 2 Success Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Trajectory accuracy (1h) | < 0.4°C MAE | From trajectory tracking |
| Trajectory accuracy (4h) | < 0.8°C MAE | From trajectory tracking |
| Overshoot rate | < 5% | From overshoot tracking |
| Convergence time | Optimal | Compare to heat curve |

### Phase 3 Success Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Temperature stability | ±0.2°C at target | Long-term monitoring |
| Energy efficiency | > Heat curve | Comparison analysis |
| User satisfaction | Rock solid | Manual validation |

---

## Technical Architecture

### File Structure
```
src/
├── thermal_equilibrium_model.py  # Core physics + adaptive learning
├── model_wrapper.py              # HA integration + binary search
├── prediction_metrics.py         # NEW: MAE/RMSE tracking
├── constrained_learning.py       # NEW: Bounded learning system
├── trajectory_predictor.py       # NEW: Trajectory prediction
├── multi_heat_source_physics.py  # Existing: Multi-source physics
├── adaptive_fireplace_learning.py # Existing: Fireplace learning
├── forecast_analytics.py         # Existing: Forecast utilities
└── main.py                       # Control loop orchestration
```

### Data Flow
```
Sensors → Features → ThermalEquilibriumModel → Trajectory Prediction
                            ↓                          ↓
                    Binary Search              Overshoot Check
                            ↓                          ↓
                    Optimal Outlet ←────── Adjustment if needed
                            ↓
                    HA Set State
                            ↓
                    Feedback Loop → Constrained Learning
                            ↓
                    Metrics Update → HA Sensors
```

---

## Progress Tracking

### Phase 1 Progress
- [ ] Task 1.1: MAE/RMSE Tracking System
  - [ ] PredictionMetrics class created
  - [ ] Rolling windows implemented
  - [ ] State persistence added
  - [ ] Tests passing
  
- [ ] Task 1.2: Enhanced HA Metrics Export
  - [ ] MAE/RMSE in ml_heating_state
  - [ ] Learned parameters displayed
  - [ ] Learning status indicator
  - [ ] sensor.ml_heating_learning created
  
- [ ] Task 1.3: Trajectory Prediction
  - [ ] predict_thermal_trajectory() implemented
  - [ ] Weather forecast integration
  - [ ] PV forecast integration  
  - [ ] Trajectory accuracy tracking
  
- [ ] Task 1.4: Constrained Adaptive Learning
  - [ ] Correction-based learning implemented
  - [ ] Parameter bounds enforced
  - [ ] Stability filtering added
  - [ ] Tests passing

### Phase 2 Progress
- [ ] Task 2.1: Weather Forecast Integration
- [ ] Task 2.2: PV Forecast Integration
- [ ] Task 2.3: Multi-Heat Source Learning
- [ ] Task 2.4: Overshoot Prevention

### Phase 3 Progress
- [ ] Task 3.1: Seasonal Adaptation
- [ ] Task 3.2: Advanced Dashboard
- [ ] Task 3.3: Documentation & Testing

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Learning drifts to unrealistic values | Bounded corrections (±30% max) |
| Learning from transient data | Stability period filtering |
| Forecast inaccuracy | Forecast confidence weighting |
| System instability | Safe fallback to fixed parameters |

---

## References

- [THERMALEQUILIBRIUM_MODEL_MIGRATION_PLAN.md](./THERMALEQUILIBRIUM_MODEL_MIGRATION_PLAN.md) - Previous migration
- [TODO_THERMAL_EQUILIBRIUM_ENHANCEMENT.md](./TODO_THERMAL_EQUILIBRIUM_ENHANCEMENT.md) - Enhancement todos
- [MULTI_HEAT_SOURCE_OPTIMIZATION_STRATEGY.md](./MULTI_HEAT_SOURCE_OPTIMIZATION_STRATEGY.md) - Multi-source strategy
- [WEEK4_ADVANCED_CONTROL_LOGIC_PLAN.md](./WEEK4_ADVANCED_CONTROL_LOGIC_PLAN.md) - Control logic details

---

**Last Updated:** December 4, 2025  
**Next Review:** Weekly progress check
