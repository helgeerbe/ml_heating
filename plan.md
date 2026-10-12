# ML Heating — Architecture Remediation Plan

> Living tracking document for the deep architecture review (2026-10-05).
> Source findings: architecture review + the 65°C untrained-model incident.
> Single source of truth for remediation tasks — update status inline as work
> progresses. Companion docs: `memory-bank/progress.md`, `IMPROVEMENT_ROADMAP.md`.

## How to use this document
- Each task has a stable ID (`T<phase>.<n>`), priority, status, and acceptance criteria.
- Status legend: `[ ]` Not started · `[~]` In progress · `[x]` Done · `[!]` Blocked · `[-]` Won't do
- Priorities: **P0** = safety/power/production-critical · **P1** = high · **P2** = medium · **P3** = low

---

## Executive summary
- The physics model (2-node RC heat balance with equilibrium inversion + Newton-cooling trajectory) is conceptually sound and is the right architecture for this problem.
- The **system around the model** is the problem: an ill-conditioned hand-rolled online learner, an accreting control wrapper, hand-validated JSON state that repeatedly corrupts, duplicated bounds, docs that describe a non-existent system, and a test suite that passes in a parameter regime that doesn't match production.
- Evidence of fragility: **128 of 295 commits (~43%)** are `fix`/`overshoot`/`drop`/`clamp`/`jump` patches; **8 unit tests currently fail** while docs claim "100% passing / zero tech debt."
- Highest-impact live incident: an **untrained model commands 65°C** (vs heat curve ~38°C), wasting power. Root cause: no uncalibrated-state safeguard + hyperbolic OE sensitivity + silent 65°C clamp.

---

## CRITICAL INCIDENT #1 — Untrained model outputs 65°C (power waste / safety)
**Root cause:** `_calculate_equilibrium_outlet_temperature` inverts the heat balance as
`outlet = target + (HLC/OE)·(target − outdoor) − ext/OE`. When `HLC/OE >= ~3`
(untrained or poisoned state), `equilibrium_outlet` exceeds the hardcoded
`max_outlet = 65.0` (`thermal_equilibrium_model.py:1663`) and is silently clamped
to 65. There is no gate that falls back to the heat curve when the model is
uncalibrated, even though `main.py:1288` already reads `heat_curve_temp`.

| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T0.1 | Emit a warning + HA flag when `equilibrium_outlet` is clamped to `max_outlet` (model in untrustworthy regime) | P0 | `[ ]` | `thermal_equilibrium_model.py:1656-1665` |
| T0.2 | Add an "is_calibrated / is_trustworthy" gate: until the model has N confirmed predictions within tolerance, command a conservative outlet derived from the heat-curve sensor, not the raw inversion | P0 | `[ ]` | `model_wrapper.py`, `main.py:1288-1301` |
| T0.3 | Add a lower-bound sanity guard for OE (not just `OE <= 0`): if `OE < OE_SAFE_MIN` or `HLC/OE > RATIO_MAX`, return a heat-curve-anchored outlet and log | P0 | `[ ]` | `thermal_equilibrium_model.py:1650-1651` |
| T0.4 | Unify the "max outlet" constant to ONE source (`thermal_config.outlet_temp_max`) and remove the hardcoded `65.0`, `CLAMP_MAX_ABS=55`, `MAX_OUTLET_TEMP=70` duplicates | P0 | `[ ]` | `thermal_equilibrium_model.py:1663`, `config.py:356`, `thermal_constants.py:33`, `thermal_config.py:32` |
| T0.5 | Write a regression test: untrained model (defaults / low-OE state) must NOT exceed a conservative ceiling and must log the fallback | P0 | `[ ]` | `tests/unit/` (new) |

**Acceptance:** A fresh/untrained model run produces an outlet <= heat-curve-derived value (not 65°C), and publishes an `ml_model_uncalibrated` flag to HA.

---

## Phase 0 — Stop the bleeding (low risk, high value)
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T0.6 | Reconcile memory-bank docs with code: delete/correct "River framework", "7-stage pipeline", `RealisticPhysicsModel` references (none exist in `src/`) | P1 | `[ ]` | `projectbrief.md`, `productContext.md`, `systemPatterns.md` |
| T0.7 | Correct false claims "Zero Technical Debt" / "294 tests 100% success" in `progress.md`/`activeContext.md`; reflect real status | P1 | `[ ]` | `memory-bank/progress.md`, `activeContext.md` |
| T0.8 | Fix or quarantine the 8 failing `test_trajectory_correction` tests | P1 | `[ ]` | `tests/unit/test_trajectory_correction.py` |
| T0.9 | Delete dead/orphaned `multi_heat_source_physics.py` (833 lines, not imported by `main.py`/`model_wrapper.py`) after confirming no dynamic import | P2 | `[ ]` | `src/multi_heat_source_physics.py` |
| T0.10 | Remove commented-out `GradualTemperatureControl` block sitting in the live path of `main.py` | P2 | `[ ]` | `src/main.py` |
| T0.11 | Consolidate outlet bounds to a single source of truth (covered by T0.4) | P0 | `[ ]` | as above |

---

## Phase 1 — Type-safe state & configuration (kills the "poisoned state" bug class)
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T1.1 | Replace `unified_thermal_state.json` dict access with a `pydantic`/`dataclass` schema with bounded, validated fields | P1 | `[ ]` | `unified_thermal_state.py`, `state_manager.py`, `thermal_state_validator.py` |
| T1.2 | With T1.1 done, retire `_detect_parameter_corruption` heuristic (HLC>0.6 AND Eff<0.35) — schema validation replaces it | P2 | `[ ]` | `thermal_equilibrium_model.py` |
| T1.3 | Centralize all magic numbers (`0.001`, `100.0`, `exp(-4.6·surplus)`, `5.0`, `0.2x` dampening) into `thermal_constants.py` with names + rationale | P2 | `[ ]` | `thermal_equilibrium_model.py`, `model_wrapper.py` |
| T1.4 | Remove singleton pattern (`_enhanced_model_wrapper_instance`, `get_thermal_state_manager()`) in favor of dependency injection | P2 | `[ ]` | `model_wrapper.py`, `unified_thermal_state.py` |
| T1.5 | Remove dual-import `try: from . import config except ImportError: import config` by making the package `pip install -e .`-able | P3 | `[ ]` | all `src/*.py` |

---

## Phase 2 — Replace the online learner (fixes the root cause of recurring instability)
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T2.1 | Replace finite-difference perturbation learning with a real optimizer on the differentiable model (analytic gradients or `scipy.optimize`/Adam on a Huber loss over a windowed prediction error) | P1 | `[ ]` | `thermal_equilibrium_model.py:850-1003` |
| T2.2 | Remove manual per-parameter scaling hacks (`pv_update *= 0.001`, `solar_lag_update *= 100.0`) once a normalized optimizer exists | P1 | `[ ]` | `thermal_equilibrium_model.py:908,915` |
| T2.3 | Remove error-clip/cold-weather-damping/catastrophic-error-block band-aids once T2.1 stabilizes the learner | P2 | `[ ]` | `thermal_equilibrium_model.py` |
| T2.4 | Add bounded-transform parameterization so the learner cannot push params outside physical ranges by construction | P1 | `[ ]` | `thermal_equilibrium_model.py` |
| T2.5 | Add a validation/holdout window + drift guard; freeze learning when validation error degrades | P2 | `[ ]` | `thermal_equilibrium_model.py` |

---

## Phase 3 — Simplify the control wrapper
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T3.1 | Collapse `_verify_trajectory_and_correct` + `_calculate_physics_based_correction` + "dynamic floor" into one well-named control law | P2 | `[ ]` | `model_wrapper.py` |
| T3.2 | Replace empirical `exp(-4.6·surplus)` / `25 + deficit·5` curves with a principled anti-windup/rate-limiter | P2 | `[ ]` | `model_wrapper.py` |
| T3.3 | Decompose `main()` (1454 lines) into `CycleRunner` orchestrating `SensorDataManager`, `BlockingStateManager`, `Predictor`, `Publisher` | P2 | `[ ]` | `main.py` |
| T3.4 | Narrow bare `except Exception` (16 in `model_wrapper.py`, 15 in `main.py`) to specific exceptions; let dev fail loudly | P2 | `[ ]` | `model_wrapper.py`, `main.py`, `influx_service.py` |

---

## Phase 4 — Test strategy
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T4.1 | Run tests against production-equivalent parameters (conftest pins OE=0.04 while prod uses ~0.8; ~20x sensitivity difference) | P1 | `[ ]` | `tests/conftest.py`, `thermal_config.py` |
| T4.2 | Convert the 30 `validation/reproduce_*.py`/`verify_*.py` scripts into permanent regression tests or delete them | P3 | `[ ]` | `validation/` |
| T4.3 | Add sociable integration tests (less mocking) per `IMPROVEMENT_ROADMAP.md` | P2 | `[ ]` | `tests/integration/` |
| T4.4 | Add CI gate so the suite can't report "passing" while tests fail | P1 | `[ ]` | CI config |

---

## Phase 5 — Hygiene (low priority)
| ID | Task | Pri | Status | Files |
|----|------|-----|--------|-------|
| T5.1 | Reduce emoji-laden production logging to structured log fields | P3 | `[ ]` | `model_wrapper.py`, `main.py` |
| T5.2 | Audit/trim `.hypothesis/` and stale `__pycache__` artifacts from the tree | P3 | `[ ]` | repo root |

---

## Master task tracker
| ID | Pri | Status | Short description |
|----|-----|--------|-------------------|
| T0.1 | P0 | `[ ]` | Warn + flag when outlet clamped to 65 |
| T0.2 | P0 | `[ ]` | Uncalibrated -> heat-curve fallback gate |
| T0.3 | P0 | `[ ]` | OE lower-bound / HLC:OE ratio sanity guard |
| T0.4 | P0 | `[ ]` | Unify max-outlet constant (1 source) |
| T0.5 | P0 | `[ ]` | Regression test: untrained != 65C |
| T0.6 | P1 | `[ ]` | Reconcile docs with code (River/7-stage) |
| T0.7 | P1 | `[ ]` | Correct false "zero debt / 100% tests" claims |
| T0.8 | P1 | `[ ]` | Fix 8 failing trajectory-correction tests |
| T0.9 | P2 | `[ ]` | Delete dead multi_heat_source_physics.py |
| T0.10 | P2 | `[ ]` | Remove dead GradualTemperatureControl block |
| T1.1 | P1 | `[ ]` | pydantic/dataclass state schema |
| T1.2 | P2 | `[ ]` | Retire corruption-detection heuristic |
| T1.3 | P2 | `[ ]` | Centralize magic numbers |
| T1.4 | P2 | `[ ]` | Remove singletons -> DI |
| T1.5 | P3 | `[ ]` | Make package pip-installable (drop dual import) |
| T2.1 | P1 | `[ ]` | Real optimizer on differentiable model |
| T2.2 | P1 | `[ ]` | Remove per-parameter scaling hacks |
| T2.3 | P2 | `[ ]` | Remove error-clip/damping band-aids |
| T2.4 | P1 | `[ ]` | Bounded-transform parameterization |
| T2.5 | P2 | `[ ]` | Validation window + drift guard |
| T3.1 | P2 | `[ ]` | Unify trajectory-correction layers |
| T3.2 | P2 | `[ ]` | Principled anti-windup/rate-limiter |
| T3.3 | P2 | `[ ]` | Decompose main() |
| T3.4 | P2 | `[ ]` | Narrow bare except Exception |
| T4.1 | P1 | `[ ]` | Production-equivalent test params |
| T4.2 | P3 | `[ ]` | Convert reproduce_* scripts to tests |
| T4.3 | P2 | `[ ]` | Sociable integration tests |
| T4.4 | P1 | `[ ]` | CI gate on failing tests |
| T5.1 | P3 | `[ ]` | Structured logging (drop emoji) |
| T5.2 | P3 | `[ ]` | Trim stale cache artifacts |

---

## Appendix — evidence / line references
- Outlet inversion + hardcoded 65C ceiling: `thermal_equilibrium_model.py:1638-1665`
- Parameter defaults/bounds: `thermal_config.py:31-68` (OE default 0.5, bounds (0.2, 2.0); HLC default 0.4, bounds (0.01, 1.2))
- Conflicting max-outlet constants: `config.py:356` (55), `thermal_constants.py:33` (70), `thermal_config.py:32` (65)
- Online learner (perturbation heuristic, manual scaling): `thermal_equilibrium_model.py:850-1003`
- Heat-curve sensor already read: `main.py:1288-1301`
- Test/production param mismatch: `tests/conftest.py` (OE=0.04) vs production (~0.8)
- Dead module: `src/multi_heat_source_physics.py` (833 lines, not imported by main/wrapper)
- 8 failing tests: `tests/unit/test_trajectory_correction.py`
- Git signal: 128/295 commits are fix/overshoot/drop/clamp/jump patches
