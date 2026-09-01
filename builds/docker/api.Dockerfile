FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[web]" \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/ufc-ml-assets \
    && chown -R appuser:appuser /app /var/lib/ufc-ml-assets

USER appuser

EXPOSE 8000

ENTRYPOINT ["python", "-m", "ufc_ml_api", "serve"]
CMD ["--config", "configs/production-rolling-2026.yaml", "--run-dir", "artifacts/active", "--host", "0.0.0.0", "--port", "8000"]
