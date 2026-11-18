from ultralytics import YOLO


def train():

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset_split/dataset.yaml",
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
        project="runs",
        name="parts_detection"
    )


if __name__ == "__main__":
    train()