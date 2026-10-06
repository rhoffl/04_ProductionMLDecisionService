FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY app ./app
COPY ml ./ml
COPY monitoring ./monitoring
COPY orchestration ./orchestration
COPY scripts ./scripts
RUN pip install --no-cache-dir .
COPY . .
RUN mkdir -p /app/artifacts /app/data /app/reports
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
