# TEST REPORT

Date: 2026-03-01
Project: Super-Market-Sales-Prediction

## 1) System Overview

Evidence from current codebase:

- Streamlit application entry point: `app.py`
- CLI training entry point: `train_automl.py`
- Model artifact persistence: `automl_model.pkl` with hash sidecar `automl_model.pkl.sha256`
- Test suite location: `tests/`

Observed behavior from code:

- App accepts only CSV/XLSX uploads (`app.py`, `load_data` function).
- App trains FLAML regression model from UI and supports optional GPU toggle via checkbox (`app.py`).
- CLI training uses FLAML with `USE_GPU` environment flag (`train_automl.py`).
- Both app/CLI now write hash sidecar for artifact integrity (`app.py`, `train_automl.py`).

## 2) Issues Found

Issues below were observed in prior execution traces and code audit, then fixed in current code:

1. Data leakage in training features
- Evidence: leakage columns are now explicitly dropped in both pipelines (`app.py`, `train_automl.py`), confirming prior leakage risk existed and was addressed.

2. Invalid file type handling in app upload path
- Prior stress trace showed media files reaching Excel parser path.
- Current behavior validated: invalid `.jpg` rejected with `ValueError`.

3. Corrupted model loading instability
- Prior behavior raised unpickling exceptions.
- Current behavior validated: corrupted model returns `None` and logs error.

4. Null-data training crash at metric calculation
- Prior stress trace showed `ValueError: Input contains NaN`.
- Current behavior validated: rows are filtered and training completes on null-containing data.

5. Upload replacement bug (new uploaded file ignored in session)
- Current code includes `uploaded_filename` tracking and reload logic in `app.py`.

6. Deprecated/invalid plotting argument usage
- Current code uses `use_container_width=True` in all Plotly chart calls.

## 3) Tests Created

Current test files:

- `tests/unit/test_app_helpers.py`
- `tests/integration/test_train_pipeline.py`
- `tests/integration/test_end_to_end_flow.py`
- `tests/ml/test_model_artifact_contract.py`
- `tests/edge/test_failure_modes.py`
- `tests/helpers.py`

Latest test execution evidence:

- Command: `python -m pytest -q`
- Result: `18 passed in 7.71s`

Coverage intent represented by these tests:

- Unit: helper/model utility behavior
- Integration: training + end-to-end train/load/predict flow
- ML contract: artifact keys and prediction shape/type
- Edge: missing/corrupted model, missing source/target, empty metadata cases

## 4) Stress Results

Fresh stress run evidence (Phase 8 execution):

- `LARGE_LOAD rows=50000 cols=17 sec=0.671`
- `INVALID_FILE_REJECTED=True type=ValueError`
- `RAPID_UI ok=30 fail=0`
- `REPEATED_INFER calls=5000 sec=1.654 last=105.000`
- `BATCH_INFER rows=100000 out_shape=100000 sec=0.001`
- `MISSING_MODEL none=True`
- `CORRUPTED_MODEL none=True`
- `WRONG_SCHEMA model_exists=False`
- `NULLS_MODEL model_exists=True hash_exists=True`

Stress completion marker observed:

- `PHASE8_STRESS_END`

## 5) Fixes Applied

Evidence-based fix summary from current code:

- App upload validation for file type (`app.py`): rejects unsupported extensions.
- Artifact integrity workflow (`app.py`, `train_automl.py`): write/read hash sidecar.
- Graceful model load failure path (`app.py`): returns `None` on corrupted artifacts.
- Leakage-safe training columns (`app.py`, `train_automl.py`): explicit drop list.
- Missing-value filtering before training/evaluation (`app.py`, `train_automl.py`).
- Reproducibility and device option (`seed`, `use_gpu`) in training settings.
- Stable chart rendering args (`app.py` Plotly calls).
- Numeric input guardrails in simulator form (`app.py`).
- SHAP categorical mapping robustness improvements (`app.py`).

## 6) Cleanup Done

Removed dead/unused artifacts and modules during prior fix phases, with current root now containing only active project files and test suite.

Current top-level workspace contents (evidence):

- `.git/`
- `.gitignore`
- `.streamlit/`
- `.venv/`
- `app.py`
- `automl_model.pkl`
- `docker-compose.yml`
- `Dockerfile`
- `LICENSE`
- `README.md`
- `requirements.txt`
- `supermarket_sales.csv`
- `tests/`
- `train_automl.py`

## 7) Final Stability

Validation status based on latest evidence:

- Automated tests: PASS (`18/18`)
- Stress scenarios: PASS (system + ML + data checks completed)
- Regression status: no failing assertions in latest stress run

Final assessment:

- Stable under current tested scenarios.
- No unresolved failures observed in latest test/stress evidence.