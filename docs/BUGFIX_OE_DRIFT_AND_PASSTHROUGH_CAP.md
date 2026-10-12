# Bug Fix: Outlet-Effectiveness Drift, Equilibrium Gradient & Passthrough Safety Cap

## Issue Description
The ML heating model was commanding **dangerously high outlet temperatures (~53 °C)** when the room only needed **~32 °C** to hold the 21.5 °C target. The 53 °C command fed back into the heat-curve entity, so the passthrough fallback *also* commanded 53 °C — there was no safety margin anywhere in the loop.

## Root Cause Analysis
The drift was traced through a four-link chain:

1. **Asymmetric cold-weather OE damping ratchet** — in `src/thermal_equilibrium_model.py`, outlet-effectiveness (OE) updates were damped asymmetrically during cold weather. Over ~15,000 cycles this ratcheted `outlet_effectiveness` from its baseline **1.107 → 0.358**.
2. **Equilibrium collapse** — with OE=0.358, the equilibrium math says a 32 °C outlet only achieves a ~17.9 °C room. To hit the 21.5 °C target the model therefore demanded an extreme outlet (~53 °C).
3. **Feedback-loop contamination** — the commanded 53 °C was written to the heat-curve sensor entity, which the passthrough fallback reads as its anchor. Passthrough then echoed 53 °C, reinforcing the loop.
4. **No passthrough ceiling** — the passthrough guard capped the outlet at `_MAX_OUTLET_TEMP` (65 °C), so 53 °C passed straight through.

A secondary, compounding defect was also found:

5. **Trajectory gradient is essentially zero** — the OE self-correction signal comes from a 30-minute trajectory prediction. With a thermal time constant τ≈29 h and a 30-min cycle, the approach factor per cycle is `1 - exp(-0.5/29) ≈ 0.017`. A 4 °C equilibrium error produces only a ~0.07 °C trajectory error → a gradient of ~0.0003 per cycle → an estimated **40+ day** recovery. The model literally could not "see" the equilibrium error through the short-horizon trajectory, so OE never self-corrected.

## Resolution
Four coordinated changes, all syntax-verified and live-verified:

### 1. Reset corrupted OE (state file)
`unified_thermal_state.json`: `outlet_effectiveness_delta` → 0.0, `heat_loss_coefficient` delta → 0.0, cleared stale `prediction_history` and `parameter_history` (from the corrupted-OE era), restored `total_predictions` → 50. Runtime OE returned to baseline **1.107** (equilibrium ~22.5 °C at 32 °C outlet). Note: `total_predictions` is recomputed live from `len(prediction_metrics.predictions)` (see `model_wrapper.py:2306`), so clearing the history is what actually resets the effective count.

### 2. Symmetric cold-weather OE damping
`src/thermal_equilibrium_model.py`: removed the asymmetric ratchet; all OE updates are now damped equally during cold weather so OE can no longer drift in one direction across a season.

### 3. Equilibrium-based OE gradient (T0.5 fix)
`src/thermal_equilibrium_model.py`: new `_calculate_equilibrium_based_oe_gradient()` method (~line 1253). When the room is thermally stable (indoor gradient < 0.15 °C/cycle), the actual indoor temperature *is* the equilibrium. The method computes the predicted equilibrium directly and uses `actual − predicted equilibrium` as the error signal — a signal **~50× stronger** than the trajectory-only gradient. Wired into the main OE gradient calculation at ~line 851 as a supplement to (not a replacement for) the trajectory gradient.

### 4. Passthrough safety cap
- `src/config.py`: added `PASSTHROUGH_MAX_SAFE_OUTLET` (default 50.0 °C, env-configurable).
- `src/model_wrapper.py`: the passthrough guard now caps at `config.PASSTHROUGH_MAX_SAFE_OUTLET` instead of `_MAX_OUTLET_TEMP` (65 °C), rejecting 53 °C feedback-loop contamination. The upper bound for the uncalibrated path was also replaced with `passthrough_max`.

### 5. Re-engagement configuration (`.env`)
- `CONFIDENCE_THRESHOLD=2.0` — lowered so the model can re-engage after the parameter reset (code default is 0.2; was previously overridden higher in `.env`).
- `MIN_PREDICTIONS_FOR_TRUST=5` — `src/model_wrapper.py` now reads `MIN_PREDICTIONS_FOR_TRUST` from the env (default 50, set to 5 in `.env`) so deployments with a known-good baseline can re-engage faster after a reset. `import os` added to `model_wrapper.py`.

### 6. Deterministic regression tests
`tests/unit/test_untrained_model_guard.py`: the `wrapper` fixture now pins `MIN_PREDICTIONS_FOR_TRUST = 50` so the untrained-model-guard regression tests encode their original safety intent deterministically, independent of any env override in `.env`.

## Verification
- **Live**: service restarted; ML control **ACTIVE** at cycle 9290 — `is_trusted: True`, `ml_control_mode: 'active'`, `outlet_source: 'ml_predicted'`, **Suggested 28.7 °C / Applied 28.0 °C** (down from 53 °C), OE=1.107 stable, prediction error 0.207 °C, confidence 4.83.
- **Unit tests**: 309 passed, 0 failed.

## Impact
- **Immediate**: outlet commands back in the safe 28–35 °C range; OE stable at baseline; no more 53 °C feedback loop.
- **Long-term**: the equilibrium-based gradient gives OE a strong self-correction signal (recovery from hours/days instead of 40+ days), and the 50 °C passthrough cap prevents heat-curve contamination from ever again commanding dangerous outlets.