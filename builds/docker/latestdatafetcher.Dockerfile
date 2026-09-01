FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src

# The fetcher intentionally has its own image because Playwright/Chromium is
# not needed by prediction serving and would make the API image much larger.
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[latestdata]" \
    && python -m playwright install --with-deps chromium \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m", "ufc_ml_latestdatafetcher", "refresh"]
CMD ["--fetcher-config", "configs/latestdatafetcher.yaml", "--model-config", "configs/rolling-2026.yaml"]
