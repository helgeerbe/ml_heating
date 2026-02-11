# ML Heating System - Technical Roadmap Tracker

This document tracks the operationalization of architectural findings and code quality improvements for the ML Heating system.

## 🏗️ Phase 1: Architectural Refactoring & Code Quality
*Focus: Reducing technical debt, improving maintainability, and enhancing type safety.*

### 1.1 Centralize Configuration
- [x] Create `src/thermal_constants.py` (or expand existing) to house all "magic numbers" currently in `main.py` and `thermal_equilibrium_model.py`.
    - [x] Move learning rates (e.g., `0.001`, `0.01`).
    - [x] Move time constants (e.g., `300s` retry delays).
    - [x] Move physics bounds (e.g., `CLAMP_MIN_ABS`, `CLAMP_MAX_ABS`).
- [x] Refactor `src/main.py` to import constants from the new centralized file.
- [x] Refactor `src/thermal_equilibrium_model.py` to import constants.

### 1.2 Type-Safe State Management
- [x] Define a `SystemState` dataclass in `src/state_manager.py` to replace dictionary-based state.
    ```python
    @dataclass
    class SystemState:
        last_run_features: Dict
        last_indoor_temp: float
        last_final_temp: float
        last_is_blocking: bool
        # ...
    ```
- [x] Update `load_state` and `save_state` in `src/state_manager.py` to serialize/deserialize this dataclass.
- [x] Update `src/main.py` to use `SystemState` objects instead of `state.get("key")`.

### 1.3 Refactor `main.py` (The "God Object")
- [x] Extract sensor retry/validation logic from `main.py` into `SensorDataManager` in `src/heating_controller.py`.
- [x] Move `poll_for_blocking` logic fully into `BlockingStateManager` in `src/heating_controller.py`.
- [x] Ensure `main.py` primarily calls high-level methods on controller classes.

### 1.4 Remove Singleton Pattern
- [x] Refactor `ThermalEquilibriumModel` to remove the `__new__` singleton implementation.
- [x] Update `src/main.py` to instantiate `ThermalEquilibriumModel` once and pass it where needed.
- [x] Update `src/model_wrapper.py` to accept a model instance rather than retrieving the singleton.

## 🧪 Phase 2: Testing Improvements
*Focus: Increasing test reliability and coverage.*

### 2.1 Fix Brittle Integration Tests
- [x] Refactor `tests/integration/test_main.py` to use fewer mocks.
- [x] Create a "Sociable Unit Test" for `HeatingController` that uses a real `SensorDataManager` but mocks the `HAClient`.
- [x] Verify that `test_main.py` survives the refactoring of `main.py` in Phase 1.3.

### 2.2 Improve Unit Tests
- [x] Update `tests/unit/test_thermal_equilibrium_model.py` to remove the `clean_model` fixture (once Singleton is removed).
- [x] Add property-based tests (using `hypothesis`) for `ThermalEquilibriumModel` to verify physics bounds across a wider range of inputs.
- [x] Fix `InfluxDBClient` teardown issues in tests (added `close()` method and global cleanup fixture).

## 🚀 Phase 3: Feature Implementation (From Improvement Roadmap)
*Focus: Implementing high-value features identified in `IMPROVEMENT_ROADMAP.md`.*

### 3.1 Monitoring & Alerts
- [ ] Implement `PerformanceMonitor` class to track RMSE/MAE trends.
- [ ] Add alerting logic to `main.py` to trigger HA notifications when model accuracy degrades.

### 3.2 Seasonal Adaptation
- [ ] Implement calendar-aware training weights in `ThermalEquilibriumModel`.
- [ ] Add `season` field to `SystemState` to track and persist seasonal context.

## 📉 Known Issues & Bugs
- [ ] **State 4 (No Data) Flakiness:** Investigate intermittent "missing sensors" errors even when sensors are available (likely race condition in `ha_client.py`).
- [ ] **Parameter Drift:** Monitor `heat_loss_coefficient` for unbounded drift during long periods of stable temperatures.
