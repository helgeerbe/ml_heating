# Validation Scripts

This directory contains validation and testing scripts for the ML Heating system. These scripts are used for development, debugging, and validation purposes rather than production unit tests.

## Directory Structure

```
validation/
├── README.md                          # This file
├── validate_equilibrium_scenarios.py  # Phase 4: Equilibrium equation validation
├── validate_trajectory_prediction.py  # Phase 4: Trajectory prediction validation
├── demo_multi_heat_source.py         # Multi-heat source demonstration
├── debug_physics_prediction.py       # Debug physics prediction components
├── debug_production_model.py         # Debug production model behavior
└── analyze_log_discrepancy.py        # Analyze log discrepancies
```

## Script Categories

### 🔬 Phase 4 Physics Validation (NEW)
- **`validate_equilibrium_scenarios.py`** - Validates corrected equilibrium equation physics
- **`validate_trajectory_prediction.py`** - Tests trajectory prediction with thermal dynamics

### 🧪 Scenario Testing
- **`demo_multi_heat_source.py`** - Multi-heat source demonstration and testing

### 🔍 Debug & Analysis
- **`debug_physics_prediction.py`** - Traces physics prediction components step-by-step
- **`debug_production_model.py`** - Debugs production model behavior with real parameters
- **`analyze_log_discrepancy.py`** - Analyzes discrepancies between expected and logged predictions

## Usage

### Running Individual Scripts

```bash
# Test user scenario reproduction  
cd /opt/ml_heating
python validation/test_user_scenario.py

# Comprehensive model validation
python validation/test_model_validation.py

# Debug physics predictions
python validation/debug_physics_prediction.py
```

### Running from Validation Directory

```bash
cd /opt/ml_heating/validation

# Make sure to run from parent directory or adjust paths
python ../validation/test_user_scenario.py
```

## Key Validation Scripts

### 🏠 User Scenario Test
**File:** `test_user_scenario.py`

Reproduces the exact user problem scenario:
- Current: 20.4°C, Target: 21.0°C (0.6°C gap)
- Old selection: 14°C (WRONG)
- New selection: 65°C (CORRECT)

**Success Criteria:**
- Stage 1 detection (0.6°C > 0.2°C threshold)
- Maximum heating selection (≥60°C)
- CHARGING mode activation
- Significant improvement over old logic

### 📈 Model Validation
**File:** `test_model_validation.py`

Comprehensive validation using real InfluxDB data:
- Loads historical data with enhanced filtering
- Splits into 2/3 training, 1/3 testing
- Trains new model and evaluates on unseen data
- Compares with production model performance

**Validation Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Prediction accuracy distribution
- Physics behavior compliance

### 🔬 Phase 4 Equilibrium Validation (NEW)
**File:** `validate_equilibrium_scenarios.py`

Validates the corrected thermal equilibrium equation after Phase 1-3 physics fixes:
- Tests equilibrium calculation without thermal_time_constant interference
- Validates PV, fireplace, and TV heat source contributions
- Verifies energy conservation principles
- Tests multiple heat sources working additively

**Test Scenarios:**
1. No external heat sources (baseline)
2. PV solar contribution (1kW test case)
3. Fireplace active contribution
4. Multiple heat sources combined
5. Energy conservation verification

**Success Criteria:**
- Equilibrium temperatures within physical bounds (outdoor < indoor < outlet)
- External heat sources increase temperature additively
- Energy balance equation satisfied
- Physics checks pass for all scenarios

### 🔮 Phase 4 Trajectory Validation (NEW)
**File:** `validate_trajectory_prediction.py`

Tests thermal trajectory prediction functionality after physics corrections:
- Validates heating scenarios with thermal dynamics
- Tests PV solar contribution to trajectory
- Verifies cooling scenario behavior
- Tests weather forecast integration

**Test Scenarios:**
1. Heating scenario (18°C → 22°C)
2. Trajectory with PV contribution (500W)
3. Cooling scenario (25°C → 22°C)
4. Weather forecast integration

**Success Criteria:**
- Temperature increases in heating scenarios
- PV power increases final temperatures
- Weather forecasts affect trajectory calculations
- Thermal time constant influences trajectory shape

### 🔍 Physics Debug
**File:** `debug_physics_prediction.py`

Step-by-step tracing of physics predictions:
- Basic heating/cooling calculations
- External heat source contributions
- Forecast adjustments
- Multi-lag effects
- Physics bounds enforcement

## Unit Tests vs Validation Scripts

### Unit Tests (`../tests/`)
- **Purpose:** Automated testing of individual components
- **Scope:** Specific functions, classes, modules
- **Runtime:** Fast execution for CI/CD
- **Examples:** `test_heat_balance_controller.py`, `test_state_manager.py`

### Validation Scripts (`validation/`)
- **Purpose:** End-to-end validation, debugging, analysis
- **Scope:** Complete workflows, real data, scenarios
- **Runtime:** Longer execution, detailed analysis
- **Examples:** Model validation, user scenario reproduction

## Integration with Development Workflow

### Before Deployment
1. Run battery charger logic tests
2. Validate model performance with real data
3. Test physics constraint compliance
4. Debug any discrepancies

### After Issues
1. Reproduce problem with scenario tests
2. Debug predictions with analysis scripts
3. Validate fixes with comprehensive tests
4. Document results and learnings

### Model Updates
1. Run model validation with train/test split
2. Compare new vs production model
3. Test physics behavior compliance
4. Validate Heat Balance Controller logic

## Environment Requirements

- Python 3.8+
- Required packages: pandas, numpy, logging, pickle
- Access to InfluxDB (for model validation)
- ML Heating source code in `../src/`

## Contributing

When adding new validation scripts:

1. **Naming:** Use descriptive names starting with test_, debug_, or analyze_
2. **Documentation:** Include clear docstrings and purpose
3. **Structure:** Follow existing patterns for imports and organization
4. **Results:** Include clear pass/fail criteria and summary output
5. **README:** Update this README with script description

## Troubleshooting

### Import Errors
Ensure scripts run from project root or validation directory with proper path setup:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### InfluxDB Connection Issues
Check InfluxDB configuration in `../src/config.py`:
- INFLUX_URL
- INFLUX_TOKEN  
- INFLUX_ORG
- INFLUX_BUCKET

### Model Loading Errors
Verify model files exist:
- `../unified_thermal_state.json` (production state)
- `../physics_model_validation.pkl` (validation model)

## Related Documentation

- [Unit Tests](../tests/) - Automated component testing
- [Source Code](../src/) - Main application code
- [Notebooks](../notebooks/) - Data analysis and exploration
- [Documentation](../docs/) - Project documentation
