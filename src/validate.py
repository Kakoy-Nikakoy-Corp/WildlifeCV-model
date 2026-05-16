from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path

from config import EXPERIMENT_NAME, DATASET_URL

settings.update({
    'datasets_dir': str(Path('./dataset').resolve())
})

model = YOLO("models/best/best.pt")

results = model.val(
    cfg="data/params.yaml",
    name=EXPERIMENT_NAME,
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
    "mAP@50": results.box.map50,
    "mAP@50:95": results.box.map,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Preprocess (ms)": results.speed["preprocess"],
    "Inference (ms)": results.speed["inference"],
    "Postprocess (ms)": results.speed["postprocess"],
}

print("\n" + "=" * 50)
print(f"Validation results: {EXPERIMENT_NAME}")
print("=" * 50)

for metric, value in metrics.items():
    print(f"{metric:<20} : {value:.4f}")

print("=" * 50)
