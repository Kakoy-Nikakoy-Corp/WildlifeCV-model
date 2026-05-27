from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path
import mlflow

from config import EXPERIMENT_NAME, ARTIFACT_URI, MODEL_PATH, DATASET_URL, RUN

settings.update({
    'datasets_dir': str(Path('./dataset').resolve())
})

if __name__ == '__main__':
    # Create only if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_URI
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    model = YOLO(MODEL_PATH)
    model.train(cfg="data/params.yaml", project=EXPERIMENT_NAME, data=DATASET_URL, name=RUN)

    mlflow.end_run()
    with mlflow.start_run(run_name=RUN):
        results = model.val(
            cfg="data/params.yaml",
            project=EXPERIMENT_NAME,
            name=RUN,
            data=DATASET_URL,
            split="test"
        )

        # Base metrics
        precision = results.box.mp
        recall = results.box.mr

        # F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Preprocess_ms": results.speed["preprocess"],
            "Inference_ms": results.speed["inference"],
            "Postprocess_ms": results.speed["postprocess"],
        }

        mlflow.log_metrics(metrics)
