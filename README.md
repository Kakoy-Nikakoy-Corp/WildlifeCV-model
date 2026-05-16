## installation

```bash
git clone ...
cd WildlifeCV-model

uv sync
```

## basic usage
run training:
```bash
uv run train-model
```

run mlflow ui (disable redundant functionality in options later?)
```bash
uv run serve-ui
```

## .env example
```env
# MLFlow
EXPERIMENT_NAME=WildlifeCV
ARTIFACT_LOCATION=<S3 URI>
MODEL_PATH=models/yolo26n.pt

# PostgreSQL
MLFLOW_TRACKING_URI=<database URI>

# Garage S3
MLFLOW_S3_ENDPOINT_URL=<URL>
AWS_ACCESS_KEY_ID=<16 digit hex key>
AWS_SECRET_ACCESS_KEY=<32 digit hex key>
AWS_DEFAULT_REGION=<region>
```