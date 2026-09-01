# UFC ML latest-data fetcher

This service boundary owns UFCStats acquisition only. It does not import the prediction
API, overwrite active processed assets, or train a model. SQLite is the transactional local
source of truth; normalized CSVs are derived snapshots, and raw HTML is content-addressed.

```text
completed events -> event pages -> terminal fight pages -> transactional SQLite -> CSV views
fighter A-Z indexes -> missing fighter pages -----------^
```

Install and initialize the browser runtime from the repository root:

```powershell
python -m pip install -e ".[latestdata]"
python -m playwright install chromium
```

Common commands:

```powershell
python -m ufc_ml_latestdatafetcher discover
python -m ufc_ml_latestdatafetcher refresh
python -m ufc_ml_latestdatafetcher validate
python -m ufc_ml_latestdatafetcher status
```

`refresh` is incremental at the SQLite event boundary. It always refreshes the completed-events
index, then skips event IDs that already have transactionally committed fights. The newest stored
fight date is exposed as a diagnostic watermark, but event-ID comparison also repairs missing
older events and same-day peers. Pass `--refresh-detail-pages` only when stored event/fight pages
must be revisited intentionally. `--max-events` is applied after stored IDs are removed, so it
limits actual pending work rather than counting events already present.

Only complete, validated runs create versioned bundles containing a SQLite backup,
normalized CSV snapshots, and legacy-shaped review files. For crawled events, the
normalized ID-rich tables are authoritative; run `backfill` for complete
ID-rich history. Fighter-page `*_current` summaries are quarantined
from historical feature construction, which must emit each matchup before applying that
fight's outcome and statistics.

Configuration lives in `configs/latestdatafetcher.yaml`. The CLI separately loads
`configs/default.yaml` for the authoritative dataset cutoff and 71-feature dictionary.
Detailed storage, backfill, and retraining-handoff notes are in the root README.
