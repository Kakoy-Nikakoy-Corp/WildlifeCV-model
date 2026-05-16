from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path

import mlflow



# Create only if it doesn't exist
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_URI
    )

mlflow.set_experiment(EXPERIMENT_NAME)
print(mlflow.get_artifact_uri())

settings.update({'datasets_dir': str(Path('./dataset').resolve())})

model = YOLO(MODEL_PATH)
model.train(cfg="data/params.yaml", name=EXPERIMENT_NAME, data=DATASET_URL)
