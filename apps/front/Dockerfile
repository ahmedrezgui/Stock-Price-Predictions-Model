FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir 'streamlit>=1.40,<2' 'plotly>=5,<7' 'requests-cache>=1,<2' 'python-dotenv>=1,<2' 'pandas>=2.1,<4'
COPY apps/front /app/apps/front
COPY .streamlit /app/.streamlit
RUN useradd --create-home frontend && chown -R frontend:frontend /app
USER frontend
ENV FRONTEND_CACHE_EXPIRATION_TIME=300 WINDOW_SIZE=20
EXPOSE 8501
CMD ["streamlit", "run", "apps/front/home.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]
