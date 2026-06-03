from ultralytics import YOLO

from config import EXPERIMENT_NAME,  MODEL_PATH, DATASET_URL, RUN


if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    model.tune(
        iterations=30,
        epochs=30,
        patience=10,
        batch=96,
        imgsz=640,
        workers=16,
        compile=True,
        optimizer='AdamW',
        single_cls=True,
        cos_lr=True,
        project=EXPERIMENT_NAME,
        data=DATASET_URL,
        name=RUN,
        plots=False,
        val=True,
        save=False
    )
