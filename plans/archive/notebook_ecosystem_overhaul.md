# Notebook Ecosystem Overhaul Proposal

## 1. Executive Summary
The current notebook ecosystem relies on fragile workarounds (`notebook_imports.py`, `sys.path` hacks) and duplicated logic to function. This creates a disconnect between the research environment (notebooks) and the production environment (`src`), leading to "works in notebook, fails in prod" issues and high maintenance overhead.

This proposal outlines a strategy to treat notebooks as first-class citizens by standardizing the environment, eliminating path hacks, and creating a shared analysis library.

## 2. Current Pain Points

| Pain Point | Description | Impact |
| :--- | :--- | :--- |
| **Path Hacking** | Notebooks rely on `sys.path.append('../')` and `notebook_imports.py` to find modules. | Fragile imports; moving a notebook breaks it; IDEs struggle with autocomplete. |
| **Mock Drift** | `notebook_imports.py` mocks `config`, `metrics`, and `InfluxService`. | Notebooks run against "fake" implementations that may drift from production logic. |
| **Logic Duplication** | Critical logic like `get_feature_names` and `load_model` is redefined in `notebook_imports.py`. | Changes in `src` are not automatically reflected in notebooks, invalidating research. |
| **Inconsistent Data** | Data loading logic is scattered or hacked into the mock Influx service. | No standard way to pull "clean" training datasets; hard to reproduce results. |

## 3. Proposed Architecture

### 3.1. Package-Based Environment
Instead of path hacking, we will structure the project so it can be installed in editable mode.

**Action:**
- Ensure `setup.py` or `pyproject.toml` is correctly configured.
- Developers run `pip install -e .` in the environment.
- **Result:** `import src.config` works anywhere, without `sys.path` modification.

### 3.2. The `ml_heating.analysis` Module
We will retire `notebook_imports.py` and `notebook_fix_helper.py` in favor of a formal submodule within the main package (or a dedicated sibling package) designed for research.

**New Structure:**
```text
src/
  analysis/           # NEW: Tools specifically for notebooks/research
    __init__.py
    data_loader.py    # Standardized InfluxDB data fetching for DataFrames
    plotting.py       # Shared plotting utilities (matplotlib/plotly wrappers)
    model_utils.py    # Helpers to load/inspect models (replacing notebook_fix_helper)
    validation.py     # Standard validation metrics and reports
```

### 3.3. Unified Configuration
Refactor `src/config.py` to be importable without side effects or heavy dependencies.

- **Current:** `notebook_imports.py` attempts to manually load config or mock it.
- **Proposed:** `src.config` should handle environment detection. If running in a notebook (detectable via `os.environ` or simple try/except), it should load a `config.yaml` from a standard location or allow manual override, without crashing if HA is missing.

### 3.4. Data Access Layer (DAL)
Notebooks need historical data in Pandas DataFrames, while the app needs point-in-time values.

**New `src.analysis.data_loader`:**
```python
from src.analysis.data_loader import fetch_training_data

# Returns a clean, pre-processed DataFrame ready for training
df = fetch_training_data(
    start_time="2023-01-01", 
    end_time="2023-12-31", 
    features=['outdoor_temp', 'outlet_temp', ...]
)
```
This encapsulates the InfluxDB query logic, ensuring notebooks and the app (if it ever needs history) use the same query definitions.

## 4. Migration Strategy

### Phase 1: Foundation (Week 1)
1.  **Fix Packaging:** Ensure `pip install -e .` works flawlessly.
2.  **Create `src.analysis`:** Scaffold the directory.
3.  **Port Helpers:** Move `notebook_fix_helper.py` logic to `src.analysis.model_utils`.
4.  **Port Config:** Ensure `src.config` is robust enough to be imported directly.

### Phase 2: Data & Imports (Week 1-2)
1.  **Implement DAL:** Create `src.analysis.data_loader` replacing the mock Influx wrapper.
2.  **Retire `notebook_imports.py`:** Create a "Compatibility Shim" that issues a deprecation warning and redirects to `src.analysis`, then eventually delete it.

### Phase 3: Notebook Migration (Ongoing)
1.  **Standard Template:** Create `notebooks/templates/standard_analysis.ipynb` using the new imports.
2.  **Migrate Active Notebooks:** Update `notebooks/development/` to use the new structure.
3.  **Archive Legacy:** Move broken/old notebooks to `notebooks/archive/legacy_v1/` and leave them as-is (or add a markdown note that they require the old environment).

## 5. Directory Structure Update

```text
/opt/ml_heating
├── src/
│   ├── analysis/       <-- NEW
│   ├── config.py       <-- UPDATED (Safe import)
│   └── ...
├── notebooks/
│   ├── development/    <-- Active work (migrated)
│   ├── templates/      <-- NEW: Standard starting points
│   ├── archive/        <-- Old/Reference
│   └── notebook_imports.py  <-- DEPRECATED (Delete after migration)
├── pyproject.toml      <-- ENSURE (Editable install support)
└── ...
```

## 6. Benefits
- **Reproducibility:** Notebooks use the *exact same* code as production.
- **Developer Experience:** Autocomplete works; no boilerplate setup cells.
- **Stability:** Refactoring `src` automatically updates the analysis tools.
