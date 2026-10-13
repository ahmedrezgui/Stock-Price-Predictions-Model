FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
COPY stock_forecasting /app/stock_forecasting
COPY stock_assistant /app/stock_assistant
COPY market_data /app/market_data
COPY enums /app/enums
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[assistant,gemini]'
RUN useradd --create-home assistant && mkdir -p /app/artifacts /app/data/raw && chown -R assistant:assistant /app
USER assistant
ENV ASSISTANT_MLFLOW_TRACKING_URI=sqlite:////app/artifacts/assistant.db
EXPOSE 8001
CMD ["uvicorn", "stock_assistant.api:app", "--host", "0.0.0.0", "--port", "8001"]
