FROM python:3.11-slim
ARG ASSISTANT_EXTRAS=assistant,gemini
WORKDIR /app
COPY pyproject.toml /app/
COPY stock_forecasting /app/stock_forecasting
COPY stock_assistant /app/stock_assistant
COPY market_data /app/market_data
COPY enums /app/enums
COPY apps/__init__.py /app/apps/__init__.py
COPY apps/back /app/apps/back
# This service calls remote forecasts; it does not need the training runtime.
RUN python -c "import os,tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; skip=('torch','matplotlib','statsmodels'); req=[r for r in p['dependencies'] if not r.startswith(skip)]; req += [r for extra in os.environ['ASSISTANT_EXTRAS'].split(',') for r in p['optional-dependencies'][extra]]; open('/tmp/requirements.txt','w').write('\n'.join(req))" \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps .
RUN useradd --create-home assistant && mkdir -p /app/artifacts /app/data/raw && chown -R assistant:assistant /app
USER assistant
ENV ASSISTANT_MLFLOW_TRACKING_URI=sqlite:////app/artifacts/assistant.db
EXPOSE 8001
CMD ["uvicorn", "stock_assistant.api:app", "--host", "0.0.0.0", "--port", "8001"]
