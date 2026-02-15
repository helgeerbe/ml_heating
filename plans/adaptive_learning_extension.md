# Plan: Adaptive Learning Extension for External Heat Sources

This plan outlines the steps to extend the `ThermalEquilibriumModel` to learn weights for external heat sources (TV, PV) and to integrate the `AdaptiveFireplaceLearning` module for more accurate fireplace heat estimation.

## 1. Analysis of Source Attribution

The goal is to disentangle the thermal contributions of different heat sources. A naive gradient descent on all parameters simultaneously can lead to "crosstalk" where one source absorbs the error caused by another (e.g., attributing fireplace heat to the TV if they are often on together).

### Strategy: Conditional Gradient Updates (Masking)

We will employ a **conditional update strategy** where specific parameter weights are only updated when their corresponding source is active and "dominant" or at least isolated from other confounding variables.

*   **PV Heat Weight (`pv_heat_weight`):**
    *   **Condition:** `pv_power > threshold` (e.g., 500W) AND `fireplace_on == 0`.
    *   **Rationale:** Solar gain is best learned during the day when the fireplace is off. TV usage is typically low or negligible compared to solar gain during peak hours.
    *   **Gradient:** Calculated w.r.t `pv_heat_weight`.

*   **TV Heat Weight (`tv_heat_weight`):**
    *   **Condition:** `tv_on > 0` AND `fireplace_on == 0` AND `pv_power < threshold`.
    *   **Rationale:** TV heat is subtle. We should only attempt to learn it when the fireplace (a massive heat source) is off and solar gain is minimal (evening/night).
    *   **Gradient:** Calculated w.r.t `tv_heat_weight`.

*   **Fireplace:**
    *   **Strategy:** Instead of a simple linear weight, we will use the `AdaptiveFireplaceLearning` module. This module uses a more complex internal model (learning efficiency, distribution, etc.) based on temperature differentials.
    *   **Integration:** The `ThermalEquilibriumModel` will accept a pre-calculated `fireplace_power_kw` from the wrapper, which uses the adaptive module. The `ThermalEquilibriumModel` will *not* learn a linear weight for this input (or will keep it fixed at 1.0), relying instead on the `AdaptiveFireplaceLearning` module's own internal learning process.

## 2. Implementation Steps

### Phase 1: Extend `ThermalEquilibriumModel`

We need to modify `src/thermal_equilibrium_model.py` to support learning of `external_source_weights`.

1.  **Add Gradient Calculations:**
    *   Implement `_calculate_pv_weight_gradient(recent_predictions)`
    *   Implement `_calculate_tv_weight_gradient(recent_predictions)`
    *   These will use the existing generic `_calculate_parameter_gradient` method.

2.  **Update `_adapt_parameters_from_recent_errors`:**
    *   Add logic to calculate gradients for PV and TV weights.
    *   Apply the **Conditional Update Logic** defined above.
    *   Apply updates to `self.external_source_weights`.
    *   Ensure new weights are clipped to reasonable bounds (e.g., TV weight shouldn't be negative or absurdly high).

3.  **Update Persistence:**
    *   Ensure `_save_learning_to_thermal_state` saves the changes to `external_source_weights`.
    *   Ensure `_load_thermal_parameters` correctly loads these learned weights.

4.  **Support Direct Power Input:**
    *   Modify `predict_equilibrium_temperature` and `predict_thermal_trajectory` to accept an optional `fireplace_power_kw` argument.
    *   Logic: `heat_from_fireplace = fireplace_power_kw if fireplace_power_kw is not None else (fireplace_on * weight)`.

### Phase 2: Integrate `AdaptiveFireplaceLearning` into `EnhancedModelWrapper`

We need to modify `src/model_wrapper.py` to use the dedicated fireplace learner.

1.  **Initialization:**
    *   Initialize `self.fireplace_learner = AdaptiveFireplaceLearning()` in `__init__`.

2.  **Online Learning (Observation):**
    *   In `learn_from_prediction_feedback` (or a new dedicated method called periodically), call `self.fireplace_learner.observe_fireplace_state(...)`.
    *   This allows the fireplace module to update its internal coefficients based on observed temperature differentials.

3.  **Prediction Integration:**
    *   In `_extract_thermal_features` (or directly in prediction methods):
        *   Call `self.fireplace_learner.get_enhanced_fireplace_features(...)`.
        *   Extract `fireplace_heat_contribution_kw`.
    *   Pass this `fireplace_heat_contribution_kw` to `self.thermal_model.predict_equilibrium_temperature` (and trajectory methods) as the new `fireplace_power_kw` argument.

## 3. Verification

### Unit Tests
*   **`test_thermal_equilibrium_model.py`:**
    *   Test that `tv_heat_weight` updates when TV is on and Fireplace is off.
    *   Test that `tv_heat_weight` does *not* update when Fireplace is on (masking).
    *   Test that `pv_heat_weight` updates correctly.
    *   Test that passing `fireplace_power_kw` overrides the simple `fireplace_on` logic.

### Integration Verification
*   **Simulated Scenario:**
    *   Run a sequence of predictions where "actual" temperature behaves as if TV emits 0.2kW.
    *   Verify that `tv_heat_weight` converges towards a value that produces ~0.2kW (given the TV signal).
*   **Fireplace Integration Check:**
    *   Verify that `EnhancedModelWrapper` correctly instantiates the fireplace learner.
    *   Verify that `fireplace_power_kw` is being passed to the thermal model during prediction.

## 4. Todo List

- [ ] **Modify `ThermalEquilibriumModel`**
    - [ ] Add `_calculate_pv_weight_gradient` and `_calculate_tv_weight_gradient`.
    - [ ] Update `_adapt_parameters_from_recent_errors` with conditional logic.
    - [ ] Update `predict_equilibrium_temperature` to accept `fireplace_power_kw`.
    - [ ] Update persistence (load/save) for external weights.
- [ ] **Modify `EnhancedModelWrapper`**
    - [ ] Initialize `AdaptiveFireplaceLearning`.
    - [ ] Call `observe_fireplace_state` in the learning loop.
    - [ ] Use `get_enhanced_fireplace_features` to get kW and pass to model.
- [ ] **Testing**
    - [ ] Create unit tests for conditional learning.
    - [ ] Verify integration.
