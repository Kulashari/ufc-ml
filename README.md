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

The production configuration is
[configs/production-rolling-2026.yaml](configs/production-rolling-2026.yaml). It expects
these approved processed assets:

```text
data/processed/ufc_model_ready.csv
data/processed/ufc_fighter_latest_features.csv
data/processed/ufc_fighter_profiles_clean.csv
data/processed/ufc_model_feature_dictionary.csv
```

The checked contract is:

- 8,400 unique fights and 71 finite numeric `feature_` columns
- binary target `target_a_win`
- train: 7,530 fights through 2024-12-31
- validation: 513 fights during 2025
- final test: 357 fights during 2026 through 2026-08-29
- dataset and current fighter-snapshot cutoff: 2026-08-29

Datasets and generated artifacts are intentionally ignored by Git.

The examples below use `python -m ufc_ml_api`, which works even when the user-level
Scripts directory is not on `PATH`. After activating the virtual environment,
`ufc-predictor` is the equivalent installed convenience command.

Validate all configured assets without fitting a model:

```powershell
python -m ufc_ml_api data validate --config configs/production-rolling-2026.yaml
```

## Latest UFCStats data fetcher

The standalone package in
[`src/ufc-ml.latestdatafetcher`](src/ufc-ml.latestdatafetcher) incrementally caches and
normalizes UFCStats pages. It is deliberately separate from both the prediction API and
model training. Install its browser dependencies and Chromium once:

```powershell
python -m pip install -e ".[latestdata]"
python -m playwright install chromium
```

Inspect the events newer than the configured model cutoff without fetching their details:

```powershell
python -m ufc_ml_latestdatafetcher discover
```

Fetch completed events missing from SQLite after that cutoff and through today:

```powershell
python -m ufc_ml_latestdatafetcher refresh
```

The completed-events endpoint can contain a future card. The fetcher filters dates through
today and accepts only terminal `W/L`, `D/D`, or `NC/NC` fight pages. It uses one Playwright
browser context for the current JavaScript challenge, makes serial throttled requests, and
resumes from its local HTML cache. Each run refreshes the completed-events index, compares
eligible UFCStats event IDs with transactionally committed SQLite events, and crawls only
missing IDs. It reports the latest stored fight date but also repairs older gaps, so one
later successful event cannot hide an earlier failed event. Use `--refresh-detail-pages`
to deliberately revisit events already stored. The default refresh also crawls the A-Z fighter indexes
and fetches directory profiles missing from the local baseline (plus incomplete baseline
bios once), while selected-event
profiles are refreshed through the cache. This preserves zero-history debutant coverage.

UFCStats currently serves this crawl over `http://ufcstats.com`; the configured completed
endpoint is `/statistics/events/completed?page=all` (the trailing comma from the original
link is not part of the URL). SHA-256 records provide local reproducibility, not transport
authentication. The idempotent CLI is intended to be invoked by Windows Task Scheduler,
cron, or another external scheduler; it does not run a permanent background service.

The page graph is:

```text
completed-events index
  -> selected event details
     -> every fight detail
        -> both fighter profiles (current bio refresh/cache)
  -> A-Z fighter directories
     -> newly discovered fighter profiles
```

Local outputs are intentionally ignored by Git:

```text
data/raw/ufcstats/html/                  immutable/resumable source HTML
data/raw/ufcstats/manifests/             per-run status and failures
data/interim/ufcstats/ufcstats.sqlite3   transactional local source of truth
data/interim/ufcstats/*.csv              normalized, derived table snapshots
data/candidates/latestdatafetcher/run-*  immutable normalized + compatibility run bundles
```

Validate normalized relationships and the exact 71-feature source mapping without network
access:

```powershell
python -m ufc_ml_latestdatafetcher validate
python -m ufc_ml_latestdatafetcher status
```

`validate` requires complete A-Z fighter coverage by default. A deliberately bounded smoke
repository can be inspected with `--allow-missing-fighter-directory`, but neither a bounded
nor failed refresh publishes candidate files.
Complete refreshes publish their own immutable, run-versioned candidates automatically;
each bundle contains a consistent SQLite backup, normalized CSV views, and legacy-shaped
review files. There is no separate unsafe export step.

For a bounded smoke run, use an explicit date and event limit. `--skip-fighter-directory`
is useful only for testing; normal refreshes should retain directory discovery:

```powershell
python -m ufc_ml_latestdatafetcher refresh `
  --since 2026-08-15 `
  --through 2026-08-15 `
  --max-events 1 `
  --skip-fighter-directory
```

`backfill` crawls all completed history through the configured snapshot cutoff. It retains
draws, no-contests, and nonstandard formats because they affect historical fighter state,
even though they are not eligible binary training labels:

```powershell
python -m ufc_ml_latestdatafetcher backfill
```

The fetcher never overwrites `data/processed`, edits the model configuration, or starts
training. For the events actually crawled, the ID-rich normalized tables contain the raw
primitives used by the current feature families. Run `backfill` to create a complete
ID-rich historical corpus; an incremental refresh alone contains only post-cutoff events.
The legacy-shaped exports are compatibility candidates only: they omit
some normalized identity/provenance fields and are not the authoritative retraining input.
Current fighter-page career summaries are retained in explicitly named `*_current` columns
for inspection and must never backfill historical pre-fight state.

### Build a raw-to-71-feature candidate

The repository now includes a chronological raw-to-71 feature builder. It reads the legacy
raw fight/profile CSVs and the ID-rich SQLite data fetched by `ufc_ml_latestdatafetcher`.
All bouts contribute to each fighter's pre-fight state; labels are emitted only for decisive,
standard three- or five-round bouts from 2001-02-23 onward. It preserves the existing
fighter-A orientation contract, reconstructs the 71 model features, and creates current
fighter snapshots.

Build a reviewable candidate bundle with:

```powershell
python -m ufc_ml_api data build-features --config configs/production-rolling-2026.yaml
```

The command writes only to `data/candidates/featurebuilder/run-*`. By default it retains the
approved 8,400 processed baseline rows verbatim, seeds the feature state from the matching
cutoff snapshot, and appends newer normalized fights. It does not overwrite `data/processed`
or train a model. Each candidate includes its own `candidate-config.yaml`, 71-feature CSV,
current fighter snapshots, cleaned profiles, manifest, and raw-reconstruction regression
report. Validate the candidate before manually training from its config:

```powershell
python -m ufc_ml_api data validate --config data/candidates/featurebuilder/run-<run-id>/candidate-config.yaml
python -m ufc_ml_api train --config data/candidates/featurebuilder/run-<run-id>/candidate-config.yaml --model all
```

`configs/production-rolling-2026.yaml` is the current split policy: train through
2024-12-31, tune and calibrate on all of 2025, and reserve all 2026 completed events
for the final chronological test. It supplies 7,530 training rows, 513 validation rows,
and 357 final-test rows with the current local snapshot. This is deliberately more
recent than the legacy split while keeping a large independent tuning window and an
unseen current-era holdout.

Use `--reconstruct-baseline` only to audit the full historic rebuild. The legacy raw CSV
lacks stable event/card ordering for a small number of historical Elo updates, so its
regression report remains an audit artifact; bootstrap mode avoids changing those trusted
historic rows. The candidate config deliberately updates the dataset cutoff, expected row
count, and split labels. Do not move 2026 bouts into training until after you have evaluated
the selected model on that untouched final-test period.

## Manual training

Train the standardized logistic model and select L2/elastic-net settings by validation
log loss:

```powershell
python -m ufc_ml_api train --config configs/production-rolling-2026.yaml --model logistic
```

Train the configured XGBoost model with validation early stopping:

```powershell
python -m ufc_ml_api train --config configs/production-rolling-2026.yaml --model xgboost
```

Fit both families and apply the conservative model-selection guardrails:

```powershell
python -m ufc_ml_api train --config configs/production-rolling-2026.yaml --model all
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
python -m ufc_ml_api evaluate-final `
  --config configs/production-rolling-2026.yaml `
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
python -m ufc_ml_api predict `
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

The currently configured snapshot file contains one 2026-08-29 snapshot per fighter.
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
[src/ufc-ml.web/](src/ufc-ml.web). It is a local companion to the Python model: the browser submits
matchup inputs to a local API, and the API runs the same `predict_fight` workflow used by
the CLI. It never trains a model and the browser cannot choose arbitrary artifact paths.

The frontend requires a Vite-compatible Node.js release (currently Node 20.19+ or
22.12+). First install the optional Python web dependencies from the project root, then
start the API with the trained artifact you want to expose:

```powershell
python -m pip install -e ".[web]"
python -m ufc_ml_api serve `
  --config configs/production-rolling-2026.yaml `
  --run-dir artifacts/20260901T191756Z-logistic
```

In a second terminal, start the React development server:

```powershell
cd src/ufc-ml.web
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
cd src/ufc-ml.web
npm run build
```

The generated `src/ufc-ml.web/dist` directory is intentionally ignored by Git. If the API
reports an error, use the same spelling and optional division rules described in the CLI
prediction section above.

## Feedback deployment

Deploy the React application and prediction API separately. The public repository contains
only source code and safe deployment templates; processed data, trained artifacts, tokens,
and host-specific values stay outside Git.

### Container build profiles

Portable Docker Compose build definitions live in [builds/](builds/README.md):
`ml.api.yaml` for the FastAPI service, `ml.web.yaml` for the static React site, and
`ml.latestdatafetcher.yaml` for the one-shot scheduled scraper worker. The worker stores
its HTML cache and SQLite data in a persistent volume; none of those data assets are copied
into container images.

```text
Vercel web app  ->  public FastAPI URL  ->  private GitHub asset repository
```

### Private assets

Store the following paths in a separate private repository, preserving their layout:

```text
data/processed/
artifacts/20260901T191756Z-logistic/
```

The API can retrieve that repository at startup when these backend-only variables are set:

```text
GITHUB_ASSETS_TOKEN          fine-grained token with Contents: Read-only
UFC_ML_ASSETS_REPOSITORY     owner/private-assets-repository
UFC_ML_ASSETS_REF            full immutable Git commit SHA for the asset revision
UFC_ML_CORS_ORIGINS          exact Vercel origin, such as https://your-app.vercel.app
```

Get the pinned asset revision from the private repository with `git rev-parse HEAD`. The
bootstrapper downloads that revision using an authorization header, unpacks it outside the
public checkout, validates the expected files, and never writes the token to disk or logs.
It rejects Git LFS pointer files, so verify that a GitHub source archive contains the actual
LFS objects before using LFS-backed assets.

Set these values only in the API host's secret/environment-variable settings. Do not add a
real token to `.env` files, source code, GitHub Actions logs, Vercel, or any `VITE_*`
variable. The browser receives only names and user-facing prediction fields; it does not
receive fighter IDs, artifact paths, model metadata, feature rows, or source fingerprints.

### API on Render

The root [render.yaml](render.yaml) provides a feedback-deployment template for the API.
Create a Render Blueprint from the `main` branch and provide each `sync: false` value in the
Render dashboard. The template installs the API dependencies, binds Uvicorn to Render's
assigned port, downloads the private assets at runtime, and checks `/api/health`.

After the service is live, open:

```text
https://your-api.onrender.com/api/health
```

It should return `"status": "ok"` before you connect the web app.

### Web app on Vercel

Import this repository in Vercel and configure:

```text
Root Directory: src/ufc-ml.web
Build Command:  npm run build
Output Directory: dist
```

Add the following Vercel environment variable for Production (and Preview if desired):

```text
VITE_API_BASE_URL=https://your-api.onrender.com
```

`VITE_*` values are embedded in the browser bundle, so this must be the public API origin
only—never a token, private repository URL with credentials, or other secret. The web app
uses `/api` locally through Vite's proxy and the configured public origin after deployment.

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
  ufc-ml.core/    model and data domain service (Python import: ufc_ml_core)
    ufc_ml_core/
      data/       loading, fingerprints, validation, splits, snapshots
      features/   ordered registry and semantic/ablation groups
      models/     logistic, XGBoost, tuning, calibration, selection
      evaluation/ metrics, subgroup segmentation, reports
      inference/  identity lookup, feature reconstruction, confidence
      artifacts/  atomic versioned save/load and integrity manifests
      workflows.py training/validation, final-test, and prediction orchestration
  ufc-ml.api/     HTTP and command-line service (Python import: ufc_ml_api)
    ufc_ml_api/
      api.py      local FastAPI adapter for the React prediction UI
      assets.py   secure, pinned private-asset bootstrap for deployed API instances
      cli.py      explicit command registration
  ufc-ml.web/     standalone React + TypeScript Vite application
    src/
      components/ reusable prediction form and result UI
configs/         checked YAML configuration
reports/         generated reports (ignored except `.gitkeep`)
artifacts/       generated model runs (ignored except `.gitkeep`)
```

The hyphenated directories are service boundaries used by the repository and deployment
layout. Python imports use underscores (`ufc_ml_core` and `ufc_ml_api`) because hyphens
and dots are not valid Python module names.

Trusted artifacts saved before this reorganization remain loadable through an in-memory
compatibility alias; no legacy source directory is retained.

## Troubleshooting

- `Data file does not exist`: place the four processed CSVs at the configured paths or
  update `configs/production-rolling-2026.yaml`.
- `No fighter matches`: confirm spelling or inspect the suggested candidates.
- `matches multiple IDs`: pass the displayed stable fighter ID.
- `snapshot ... strictly before`: update the snapshot table so it contains data before the
  server's current UTC date.
- CUDA fallback: review the recorded probe reason; CPU training is fully supported.
- `Optuna is optional`: install `.[optuna]` or omit `--tune-xgboost`.
- Command not found after a user-level install: activate the virtual environment or run
  `python -m ufc_ml_api ...`.

Run lightweight quality checks:

```powershell
python -m ruff format .
python -m ruff check .
python -m mypy src
python -m compileall src/ufc-ml.core/ufc_ml_core src/ufc-ml.api/ufc_ml_api
```
