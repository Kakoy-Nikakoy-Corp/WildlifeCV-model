## Installation

```bash
git clone https://github.com/Kakoy-Nikakoy-Corp/WildlifeCV-model.git
cd WildlifeCV-model

uv sync
```

## Basic usage
> For these commands to work, you must create a valid `.env` file inside root directory. See the template below.

Run training:
```bash
uv run train-model
```

Run mlflow ui (disable redundant functionality in options later?)
```bash
uv run serve-ui
```

## .env template
Assuming we're connected to `192.168.0.104` via WireGuard VPN, 
and PostgreSQL+Garage are deployed there under their respective ports 
```env
# YOLO
MODEL_PATH=models/yolo26n.pt
DATASET_URL=https://dataset.irbis.wild1.net/dataset.ndjson

# MLFlow
EXPERIMENT_NAME=WildlifeCV

# PostgreSQL integration
MLFLOW_TRACKING_URI=postgresql://irbis:<PASSWORD>@192.168.0.104:5432/mlruns

# Garage S3 integration
ARTIFACT_URI=s3://irbis
MLFLOW_S3_ENDPOINT_URL=http://192.168.0.104:3900
AWS_ACCESS_KEY_ID=<16 DIGIT HEX KEY>
AWS_SECRET_ACCESS_KEY=<32 DIGIT HEX KEY>
AWS_DEFAULT_REGION=garage
```
