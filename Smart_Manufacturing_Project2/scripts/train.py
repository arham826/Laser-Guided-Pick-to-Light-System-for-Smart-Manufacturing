from ultralytics import YOLO
from pathlib import Path

# Project folder: this file lives in <project>/scripts/
PROJECT_DIR = Path(__file__).resolve().parents[1]


def train():

    model = YOLO(str(PROJECT_DIR / "yolov8n.pt"))

    model.train(
        data=str(PROJECT_DIR / "dataset_split" / "dataset.yaml"),
        epochs=60,
        imgsz=640,
        batch=4,          # safer for GTX1650 memory
        device=0,         # GPU
        workers=0,        # IMPORTANT for Windows multiprocessing fix
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        degrees=10,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        project=str(PROJECT_DIR / "runs" / "detect" / "runs"),
        name="parts_detection"
    )


if __name__ == "__main__":
    train()