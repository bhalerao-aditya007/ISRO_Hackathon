FROM python:3.11-slim

# System deps: GDAL, PROJ, and build tools
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces: /data is a 50GB ephemeral volume that persists across restarts within a session
RUN mkdir -p /data/prism_data /data/prism_outputs /data/prism_models

ENV PRISM_ROOT=/app
ENV PRISM_OUTPUT_DIR=/data/prism_outputs
ENV PRISM_CONFIG=/app/config/pipeline_config.json
ENV DATA_CACHE_DIR=/data/prism_data
ENV PRISM_MODELS_DIR=/data/prism_models
ENV PYTHONPATH=/app

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
