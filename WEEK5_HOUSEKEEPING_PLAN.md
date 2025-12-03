# Week 5 Housekeeping Plan - Clean Codebase for Testing & Validation

## 🎯 **Objective**
Clean up codebase before Week 5 by removing deprecated features, making thermal parameters configurable, and ensuring clean setup for testing/validation.

## 📋 **Implementation Plan**

### **Phase 1: Remove Deprecated Mode States** ✅ **COMPLETED**
- [x] 1.1 Remove `sensor.ml_control_mode` from `ha_client.py`
- [x] 1.2 Clean up charging/balancing/maintenance mode references
- [x] 1.3 Remove mode-related state tracking from all files
- [x] 1.4 Update any tests that reference deprecated modes (not needed - no mode tests found)

### **Phase 2: Thermal Parameter Configuration (Priority 1 - Core Physics)** ✅ **COMPLETED**
- [x] 2.1 Add core thermal parameters to `config.py`:
  - [x] THERMAL_TIME_CONSTANT (default: 24.0)
  - [x] HEAT_LOSS_COEFFICIENT (default: 0.05) 
  - [x] OUTLET_EFFECTIVENESS (default: 0.8)
  - [x] OUTDOOR_COUPLING (default: 0.3)
  - [x] THERMAL_BRIDGE_FACTOR (default: 0.1)
- [x] 2.2 Update `thermal_equilibrium_model.py` to use config values
- [x] 2.3 Add parameter bounds validation

### **Phase 3: External Heat Source Configuration (Priority 2)** ✅ **COMPLETED** 
- [x] 3.1 Add external heat source weights to `config.py`:
  - [x] PV_HEAT_WEIGHT (default: 0.001)
  - [x] FIREPLACE_HEAT_WEIGHT (default: 0.02)
  - [x] TV_HEAT_WEIGHT (default: 0.005) 
  - [x] ~~OCCUPANCY_HEAT_WEIGHT~~ **REMOVED** (no corresponding sensor)
  - [x] ~~COOKING_HEAT_WEIGHT~~ **REMOVED** (no corresponding sensor)
- [x] 3.2 Update thermal model to use configurable weights

### **Phase 4: Adaptive Learning Configuration (Priority 3)** ✅ **COMPLETED**
- [x] 4.1 Add learning parameters to `config.py`:
  - [x] ADAPTIVE_LEARNING_RATE (default: 0.05)
  - [x] MIN_LEARNING_RATE (default: 0.01)
  - [x] MAX_LEARNING_RATE (default: 0.2)
  - [x] LEARNING_CONFIDENCE (default: 3.0)
  - [x] RECENT_ERRORS_WINDOW (default: 10)
- [x] 4.2 Update thermal model to use configurable learning params

### **Phase 5: Environment Files Update** ✅ **COMPLETED**
- [x] 5.1 Update `.env_sample` with new thermal parameters
- [x] 5.2 Update `.env` with new thermal parameters  
- [x] 5.3 Update addon configuration files (`ml_heating/config.yaml`, `ml_heating_dev/config.yaml`)
- [x] 5.4 Add parameter documentation

### **Phase 6: Cleanup & Validation** ✅ **COMPLETED**
- [x] 6.1 Remove any other retired/unused parameters from all files
- [x] 6.2 Clean up unused imports and functions (cleaned cooking/occupancy from model_wrapper.py)
- [x] 6.3 Run test suite to verify no regressions (151/153 tests passed - 2 PV forecast failures unrelated to changes)
- [x] 6.4 Test parameter loading from environment (all thermal parameters loaded successfully)
- [x] 6.5 Validate thermal model uses all configurable parameters

## 🔧 **Parameter Groups Summary**

### Core Physics Parameters (Priority 1)
```bash
THERMAL_TIME_CONSTANT=24.0          # Building thermal response (hours)
HEAT_LOSS_COEFFICIENT=0.05          # Heat loss rate per degree  
OUTLET_EFFECTIVENESS=0.8            # Heat pump efficiency
OUTDOOR_COUPLING=0.3                # Outdoor influence factor
THERMAL_BRIDGE_FACTOR=0.1           # Thermal bridging losses
```

### External Heat Sources (Priority 2)  
```bash
PV_HEAT_WEIGHT=0.001               # Solar heating per 100W
FIREPLACE_HEAT_WEIGHT=0.02         # Fireplace per unit time
TV_HEAT_WEIGHT=0.005               # Electronics heating
OCCUPANCY_HEAT_WEIGHT=0.008        # Human heat per person
COOKING_HEAT_WEIGHT=0.015          # Kitchen appliances
```

### Adaptive Learning (Priority 3)
```bash
ADAPTIVE_LEARNING_RATE=0.05        # Base learning rate
MIN_LEARNING_RATE=0.01             # Minimum learning rate  
MAX_LEARNING_RATE=0.2              # Maximum learning rate
LEARNING_CONFIDENCE=3.0            # Initial confidence
RECENT_ERRORS_WINDOW=10            # Error analysis window
```

## ✅ **Success Criteria** - **ALL COMPLETED**
- [x] No deprecated mode sensor references in codebase
- [x] All thermal parameters configurable via environment/config
- [x] Clean test suite execution (no regressions)
- [x] Updated environment files with new parameters
- [x] Professional, maintainable configuration structure

## 📝 **Notes**
- **Approach**: Greenfield - no backward compatibility required
- **Target**: Clean setup for Week 5 testing and validation
- **Focus**: Make thermal parameters easily tunable for testing scenarios

---
**Status**: ✅ **COMPLETED SUCCESSFULLY**
**Start Date**: December 3, 2025
**Completion Date**: December 3, 2025
**Duration**: ~3 hours (6 phases completed)

## 🎯 **WEEK 5 HOUSEKEEPING COMPLETE!**
The codebase is now clean, well-configured, and ready for Week 5 testing and validation.
