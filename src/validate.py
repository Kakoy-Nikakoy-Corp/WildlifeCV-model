from ultralytics import YOLO
import mlflow

from config import EXPERIMENT_NAME, DATASET_URL, RUN


if __name__ == '__main__':
    mlflow.set_experiment(EXPERIMENT_NAME)
    model = YOLO(f"runs/detect/{EXPERIMENT_NAME}/{RUN}/weights/best.pt")

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

        conf_thresholds = results.box.px

        # Precision(conf)
        for conf, value in zip(conf_thresholds, results.box.p_curve.mean(axis=0)):
            mlflow.log_metric(
                "Precision_vs_conf",
                float(value),
                step=int(conf * 1000),
            )

        # Recall(conf)
        for conf, value in zip(conf_thresholds, results.box.r_curve.mean(axis=0)):
            mlflow.log_metric(
                "Recall_vs_conf",
                float(value),
                step=int(conf * 1000),
            )

        # F1(conf)
        for conf, value in zip(conf_thresholds, results.box.f1_curve.mean(axis=0)):
            mlflow.log_metric(
                "F1_vs_conf",
                float(value),
                step=int(conf * 1000),
            )

        mlflow.log_artifacts(
            local_dir=results.save_dir,
            artifact_path="yolo_artifacts"
        )
