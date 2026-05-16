from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path
import os
import mlflow

experiment_name = os.getenv('EXPERIMENT_NAME')
model_path = os.getenv('MODEL_PATH')

# Create only if it doesn't exist
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location=os.getenv('ARTIFACT_URI')
    )

mlflow.set_experiment(experiment_name)
print(mlflow.get_artifact_uri())

settings.update({'datasets_dir': str(Path('./dataset').resolve())})

model = YOLO(model_path)
model.train(cfg="data/params.yaml", name=experiment_name, model=model_path, data=os.getenv('DATASET_URL'))
