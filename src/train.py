from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path
import os
import mlflow

experiment_name = os.getenv('EXPERIMENT_NAME')

# Create only if it doesn't exist
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location=os.getenv('ARTIFACT_LOCATION')
    )

mlflow.set_experiment(experiment_name)
print(mlflow.get_artifact_uri())

settings.update({'datasets_dir': str(Path('./dataset').resolve())})

model = YOLO(os.getenv('MODEL_PATH'))
model.train(cfg="data/params.yaml", name=experiment_name)
