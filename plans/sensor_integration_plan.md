# Sensor Integration & Physics-Informed ML Plan

This document outlines the technical implementation plan for integrating critical physical sensors (Inlet Temperature, Flow Rate, Power Consumption) into the `ml_heating` control system.

## 1. Configuration Infrastructure & Environment Management

**Objective:** Enable the application to securely and reliably access new sensor streams via environment variables, adhering to the project's configuration standards.

- [ ] **Update `.env_sample`**
    - Add `INLET_TEMP_ENTITY_ID` (Default: `sensor.hp_inlet_temp`)
    - Add `FLOW_RATE_ENTITY_ID` (Default: `sensor.hp_current_flow_rate`)
    - Add `POWER_CONSUMPTION_ENTITY_ID` (Default: `sensor.power_wp`)
    - Add `SPECIFIC_HEAT_CAPACITY` (Default: `4.186` - kJ/kg·K)

- [ ] **Update `src/config.py`**
    - Implement `os.getenv` retrieval for the new variables.
    - Enforce type casting (strings for Entity IDs, floats for constants).
    - Add validation logic to warn or fail if critical entity IDs are missing or malformed.

- [ ] **Update `src/thermal_constants.py`**
    - Define `PhysicsConstants.SPECIFIC_HEAT_WATER = 4.186` (J/g°C or kJ/kg°C) to centralize the physics constant.
    - Define `PhysicsConstants.MIN_FLOW_RATE` and `PhysicsConstants.MAX_FLOW_RATE` for data validation boundaries.

## 2. Advanced Feature Engineering & Physics Integration

**Objective:** Transform raw sensor data into high-value thermodynamic features that allow the ML model to learn efficiency (COP) and true thermal load.

- [ ] **Update `src/physics_features.py` - Raw Ingestion**
    - Modify `build_physics_features` signature or internal logic to fetch:
        - `inlet_temp`
        - `flow_rate`
        - `power_consumption`
    - Implement `_get_safe_float` helper to handle `unavailable` or `unknown` states from Home Assistant, defaulting to `NaN` or safe fallbacks.

- [ ] **Update `src/physics_features.py` - Derived Features**
    - **Delta T (`delta_t`):**
        - Logic: `outlet_temp - inlet_temp`
        - Significance: Represents the temperature drop across the heating circuit.
    - **Thermal Power Output (`thermal_power_kw`):**
        - Logic: `flow_rate (L/h) * delta_t * 4.186 / 3600`
        - Significance: The actual heat energy delivered to the building.
    - **Real-time COP (`cop_realtime`):**
        - Logic: `thermal_power_kw / (power_consumption_watts / 1000)`
        - Handling: Handle division by zero (if power is 0).
    - **System Load (`system_load_index`):**
        - Logic: `power_consumption_watts / MAX_RATED_POWER` (if max power is known/configured).

- [ ] **Data Cleaning & Normalization**
    - Implement rolling average (e.g., 5-minute window) for `flow_rate` and `power_consumption` to smooth out sensor noise before calculation.
    - Filter out "Disinfection Mode" data points (high temp, unusual flow) to prevent model skew.

## 3. Telemetry, Observability, & Data Persistence

**Objective:** Ensure new metrics are visible in Home Assistant for debugging and persisted in InfluxDB for long-term model training.

- [ ] **Update `src/ha_client.py`**
    - Add methods to publish derived sensors back to HA (optional but recommended for dashboarding):
        - `sensor.ml_heating_cop_realtime`
        - `sensor.ml_heating_thermal_power`
    - Define attributes (unit_of_measurement, state_class) for these new sensors.

- [ ] **Update `src/influx_service.py`**
    - Modify `write_training_data` (or equivalent method) to include:
        - `inlet_temp`
        - `flow_rate`
        - `power_watts`
        - `thermal_power_kw`
        - `cop_realtime`
    - Ensure field types are explicitly set to `float` to prevent InfluxDB schema conflicts.

## 4. Test Suite Expansion & Validation

**Objective:** Verify that the new inputs are correctly loaded, calculated, and stored, ensuring system stability.

- [ ] **Unit Tests (`tests/unit/test_config.py`)**
    - Verify that new environment variables are loaded correctly.
    - Test fallback values when env vars are missing.

- [ ] **Unit Tests (`tests/unit/test_physics_features.py`)**
    - **Calculation Accuracy:** Test `thermal_power_kw` and `cop_realtime` calculations against known manual values.
    - **Boundary Handling:** Test behavior when `flow_rate` is 0, `power` is 0, or `inlet_temp` > `outlet_temp` (physics violation/defrost).
    - **NaN Handling:** Ensure the system doesn't crash if a sensor returns `None`.

- [ ] **Integration Tests (`tests/integration/test_influx_service.py`)**
    - Verify that the new fields are actually written to the mock InfluxDB client.
    - Check data structure matches the expected schema.
