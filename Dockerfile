FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
COPY pyproject.toml /app/
COPY stock_forecasting /app/stock_forecasting
COPY enums /app/enums
COPY utils /app/utils
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . azure-identity azure-storage-blob
COPY configs /app/configs
RUN useradd --create-home forecast && mkdir -p /app/artifacts/models /app/data/raw \
    && chown -R forecast:forecast /app
USER forecast
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "stock_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Place only approved bundles/data in deployment/ before building this target.
FROM base AS packaged
COPY --chown=forecast:forecast deployment/models /app/artifacts/models
COPY --chown=forecast:forecast deployment/data /app/data/raw

FROM base AS azureml
USER root
RUN pip install --no-cache-dir azureml-inference-server-http
COPY infrastructure/azure/score.py /app/score.py
USER forecast
EXPOSE 5001
CMD ["azmlinfsrv", "--entry_script", "/app/score.py", "--port", "5001"]
