# UFC Predictor

An installable, leakage-aware Python project for training UFC fight win-probability
models and predicting matchups from two fighter names. The code supports a standardized
logistic-regression baseline, XGBoost with CPU/CUDA fallback, validation-only calibration,
chronological evaluation, versioned artifacts, and order-symmetric inference.

No model is trained during installation, import, formatting, data validation, or CLI help.
Training and final-test evaluation happen only through their explicit commands.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional bounded Optuna tuner only if you intend to use
`--tune-xgboost`:

```powershell
python -m pip install -e ".[optuna]"
```

Install the optional local web API only if you want to use the React matchup
interface:

```powershell
python -m pip install -e ".[web]"
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Data configuration

The default configuration is [configs/default.yaml](configs/default.yaml). It expects
these local processed assets:

```text
data/processed/ufc_model_ready.csv
data/processed/ufc_fighter_latest_features.csv
data/processed/ufc_fighter_profiles_clean.csv
data/processed/ufc_model_feature_dictionary.csv
```

The checked contract is:

- 8,116 unique fights and 71 finite numeric `feature_` columns
- binary target `target_a_win`
- train: 5,618 fights through 2021-03-06
- validation: 1,249 fights from 2021-03-13 through 2023-08-19
- final test: 1,249 fights from 2023-08-26 through 2026-03-07
- dataset and current fighter-snapshot cutoff: 2026-03-07

Datasets and generated artifacts are intentionally ignored by Git.

Validate all configured assets without fitting a model:

```powershell
ufc-predictor data validate --config configs/default.yaml
```

## Manual training

Train the standardized logistic model and select L2/elastic-net settings by validation
log loss:

```powershell
ufc-predictor train --config configs/default.yaml --model logistic
```

Train the configured XGBoost model with validation early stopping:

```powershell
ufc-predictor train --config configs/default.yaml --model xgboost
```

Fit both families and apply the conservative model-selection guardrails:

```powershell
ufc-predictor train --config configs/default.yaml --model all
```

The `all` workflow selects XGBoost only when it meaningfully improves validation log
loss without exceeding calibration-error or aligned-subgroup regression limits. Logistic
regression wins effective ties. Add `--ablation` to refit fixed selected hyperparameters
for the seven configured feature-group ablations. Add `--tune-xgboost` only after
installing the Optuna extra; the YAML bounds and trial budget are enforced.

Training fits on `train`, chooses a candidate on `validation`, then fits the configured
sigmoid/isotonic calibrator on validation probabilities. It does not score or otherwise
use final-test rows. The raw validation report is the model-selection estimate; the
separate `calibration-fit` report is explicitly labeled as an in-sample calibrator
diagnostic and is not presented as an unbiased held-out calibrated score.

## Final held-out evaluation

Run this once, after all modeling decisions are final:

```powershell
ufc-predictor evaluate-final `
  --config configs/default.yaml `
  --run-dir artifacts/<run_id>
```

This loads the frozen model and calibrator, verifies data and feature fingerprints, scores
only the chronological test split, records the test metrics in the integrity manifest,
and writes JSON/Markdown reports. Replacing an existing final result requires the explicit
`--overwrite` flag.

Reports include log loss, Brier score, ROC-AUC, average precision, accuracy, precision,
recall, F1, confusion-matrix counts, ECE, reliability bins, division results, debut/history
groups, three-plus-fight groups, experience bands, title bouts, and probability bands.

## Fight prediction

After manually training a run:

```powershell
ufc-predictor predict `
  --fighter-a "Fighter A" `
  --fighter-b "Fighter B" `
  --run-dir artifacts/<run_id>
```

Use `--division M_LIGHT` when both snapshots do not imply the same division. Ambiguous
names return candidates; repeat the command with `--fighter-a-id` or `--fighter-b-id`
to select the stable fighter ID. `--include-features` exposes both constructed rows for
auditing.

Each prediction records a UTC `predicted_at` timestamp. Inference normalizes aliases,
selects only snapshots strictly earlier than that timestamp's UTC calendar date, refreshes
age and inactivity from the same date, reconstructs all 71 features in the saved order,
predicts both A-vs-B and B-vs-A, reverses the second probability, and averages the two.
Swapping fighter order therefore swaps the final probabilities.

The response includes `predicted_at`, resolved identities, both probabilities, winner,
prior UFC fight counts, snapshot dates, cutoff, model metadata, orientation disagreement,
applicability confidence, and warnings.

## Confidence and limitations

Confidence describes model applicability, not how certain a displayed probability looks.
It is reduced for debutants, limited UFC history, stale snapshots or inactivity, predictions
made after the cutoff, orientation disagreement, out-of-range features, and unsupported
contexts.

Known debutants are supported only when their static profile and the same zero-history
feature defaults used in training are present in the snapshot data; those predictions
are marked low confidence. Unknown fighters or rows missing a required reconstructable
feature are rejected.

The currently configured snapshot file contains one 2026-03-07 snapshot per fighter.
Predictions use it only after the server's UTC date has passed that cutoff. The lookup
implementation can consume multiple dated snapshots if a future point-in-time snapshot
table is supplied.

Rolling 365/730-day activity counts cannot be fully recomputed from one aggregate snapshot;
the predictor resets them when the recalculated layoff proves a window is empty and otherwise
retains the snapshot count with an after-cutoff warning. A requested division change is
rejected for a fighter with UFC history because the current file stores only the fighter's
latest division-specific Elo state, not a per-division history.

This is a research probability model, not a guarantee or betting recommendation.
A value such as `0.64` means the model estimates a 64% win probability under the available
data and matchup assumptions.

## Local prediction UI

The repository includes a small dark-mode React + TypeScript interface in
[src/frontend/](src/frontend). It is a local companion to the Python model: the browser submits
matchup inputs to a local API, and the API runs the same `predict_fight` workflow used by
the CLI. It never trains a model and the browser cannot choose arbitrary artifact paths.

The frontend requires a Vite-compatible Node.js release (currently Node 20.19+ or
22.12+). First install the optional Python web dependencies from the project root, then
start the API with the trained artifact you want to expose:

```powershell
python -m pip install -e ".[web]"
python -m ufc_predictor serve `
  --config configs/default.yaml `
  --run-dir artifacts/20260723T181548Z-xgboost
```

In a second terminal, start the React development server:

```powershell
cd src/frontend
npm install
npm run dev
```

Open the local URL shown by Vite (normally `http://127.0.0.1:5173`). The development
server proxies `/api` requests to `http://127.0.0.1:8000`, so no frontend environment
variables are needed for local use. The form accepts two fighter names and an optional
division. It displays the server-generated UTC prediction timestamp, predicted winner, both
win probabilities, confidence tier, known warnings, UFC-history counts, and data cutoff.

The prediction service records its current UTC timestamp once per request and uses that
same UTC calendar date for both fighters' age and inactivity features. It never accepts a
browser-provided date, so a client cannot select data from the future.

To make a production frontend bundle after installing its dependencies, run:

```powershell
cd src/frontend
npm run build
```

The generated `src/frontend/dist` directory is intentionally ignored by Git. If the API
reports an error, use the same spelling and optional division rules described in the CLI
prediction section above.

## GPU and CPU behavior

XGBoost uses `tree_method="hist"`. With `device: auto`, an explicit tiny CUDA probe runs
only during an XGBoost training command. CUDA is used when the probe succeeds; otherwise
training falls back to CPU and records the reason and device in the artifact. No
distributed or multi-GPU infrastructure is used.

## Leakage prevention

- Feature order comes from the checked pre-fight feature dictionary.
- Obvious target-derived/post-fight columns, duplicates, missing values, infinities, and
  invalid targets are rejected.
- Event dates cannot cross split boundaries.
- Standardization is inside the logistic pipeline and is fit only on training rows.
- Hyperparameters, model family, calibration, and optional ablations use validation only.
- The test split is reachable only through `evaluate-final`.
- Artifact loading verifies SHA256 integrity, schema order, cutoff, and source fingerprints.
- Inference requires `snapshot_date <` the server-generated UTC reference date.

## Project layout

```text
src/
  ufc_predictor/  installable Python package
    data/         loading, fingerprints, validation, splits, snapshots
    features/     ordered registry and semantic/ablation groups
    models/       logistic, XGBoost, tuning, calibration, selection
    evaluation/   metrics, subgroup segmentation, reports
    inference/    identity lookup, feature reconstruction, confidence
    artifacts/    atomic versioned save/load and integrity manifests
    api.py        local HTTP adapter for the React prediction UI
    cli.py        explicit command registration
    workflows.py  train/validation, final-test, and prediction orchestration
  frontend/       standalone React + TypeScript Vite application
    src/
      components/ reusable prediction form and result UI
configs/         checked YAML configuration
scripts/         lightweight environment checks
reports/         generated reports (ignored except `.gitkeep`)
artifacts/       generated model runs (ignored except `.gitkeep`)
```

## Troubleshooting

- `Data file does not exist`: place the four processed CSVs at the configured paths or
  update `configs/default.yaml`.
- `No fighter matches`: confirm spelling or inspect the suggested candidates.
- `matches multiple IDs`: pass the displayed stable fighter ID.
- `snapshot ... strictly before`: update the snapshot table so it contains data before the
  server's current UTC date.
- CUDA fallback: review the recorded probe reason; CPU training is fully supported.
- `Optuna is optional`: install `.[optuna]` or omit `--tune-xgboost`.
- Command not found after a user-level install: activate the virtual environment or run
  `python -m ufc_predictor ...`.

Run lightweight quality checks:

```powershell
python -m ruff format .
python -m ruff check .
python -m mypy src
python -m compileall src/ufc_predictor
python scripts/check_environment.py --config configs/default.yaml
```
