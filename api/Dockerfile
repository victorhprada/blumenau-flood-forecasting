FROM python:3.11-slim

WORKDIR /app

# Install build tools (needed for some torch wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts + vendor framework + api source
COPY vendor/flood-forecasting/googlehydrology ./googlehydrology
COPY models/experiments/exp03_nse_loss/exp03_nse_loss_0906_112336 \
     ./models/experiments/exp03_nse_loss/exp03_nse_loss_0906_112336
COPY configs/training/experiments/03_nse_loss.yml \
     ./configs/training/experiments/03_nse_loss.yml
COPY configs/training/itajai_basins.txt ./configs/training/itajai_basins.txt
COPY api/ ./api/

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
