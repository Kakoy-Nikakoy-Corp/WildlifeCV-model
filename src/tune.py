from ultralytics import YOLO

from config import EXPERIMENT_NAME,  MODEL_PATH, DATASET_URL, RUN


if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    model.tune(
        iterations=100,
        epochs=50,
        patience=15,
        batch=40,
        imgsz=640,
        optimizer='MuSGD',
        single_cls=True,
        cos_lr=True,
        project=EXPERIMENT_NAME,
        data=DATASET_URL,
        name=RUN,
        plots=False,
        val=True,
        save=False
    )
