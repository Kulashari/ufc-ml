# Container builds

These are three separate deployment units. Source code is baked into each
image; local data, raw UFCStats HTML, SQLite, trained artifacts, and credentials
are deliberately excluded.

| Build file | Purpose | Runtime state |
|---|---|---|
| `ml.api.yaml` | FastAPI prediction service | Immutable private model assets, downloaded at startup or mounted by the host |
| `ml.web.yaml` | React/Vite static site served by Nginx | None |
| `ml.latestdatafetcher.yaml` | One-shot Playwright scraper job | Persistent `/app/data` volume |

## Build locally

Run these commands from the repository root:

```powershell
docker compose -f builds/ml.api.yaml build
docker compose -f builds/ml.web.yaml build
docker compose -f builds/ml.latestdatafetcher.yaml build
```

The web build needs the public API origin at build time:

```powershell
$env:VITE_API_BASE_URL = "https://api.example.com"
docker compose -f builds/ml.web.yaml build
```

## Run the latest-data worker

The worker is a job, not a continuously running web service. Schedule this
exact command daily with your cloud scheduler or Windows Task Scheduler:

```powershell
docker compose -f builds/ml.latestdatafetcher.yaml run --rm latest-data-fetcher
```

The named `ufc_ml_data` volume persists `data/raw/ufcstats`,
`data/interim/ufcstats/ufcstats.sqlite3`, manifests, and candidate outputs
between job runs. It must be initialized from your private data backup before
the first production refresh; the image intentionally does not contain any
training or scrape data.

The default job requires a complete crawl. Do not add `--allow-partial` to its
scheduled command. It refreshes the completed-event index and A-Z directory,
but does not redownload every known fighter profile unless explicitly asked.

## Run the API

Set the selected artifact path and private-asset bootstrap secrets in the host
environment, then start the service:

```powershell
$env:UFC_ML_API_RUN_DIR = "artifacts/20260901T191756Z-logistic"
$env:UFC_ML_API_CONFIG = "configs/production-rolling-2026.yaml"
docker compose -f builds/ml.api.yaml up -d
```

For local Compose, `data/processed` must contain the approved 8,400-row bundle.
For a cloud API, promote that bundle and the selected artifact into the private
asset repository first. A fetcher run alone never changes the live API's model
or snapshots.
