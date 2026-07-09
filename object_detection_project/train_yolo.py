import os
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# TRAIN YOLO OBJECT DETECTION
# Bài toán: phát hiện bottle, cup, pen trong ảnh
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

PRETRAINED_MODEL = "yolov8n.pt"
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 8

RUNS_DIR = BASE_DIR / "runs" / "detect"
BEST_MODEL_PATH = RUNS_DIR / "train" / "weights" / "best.pt"


def check_dataset():
    # Dataset YOLO cần đủ ảnh, label và file data.yaml.
    required_paths = [
        DATA_YAML,
        BASE_DIR / "dataset" / "images" / "train",
        BASE_DIR / "dataset" / "images" / "val",
        BASE_DIR / "dataset" / "labels" / "train",
        BASE_DIR / "dataset" / "labels" / "val",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu đường dẫn: {path}")


def main():
    check_dataset()

    # data.yaml dùng "path: dataset", nên cần chạy YOLO từ thư mục project.
    os.chdir(BASE_DIR)

    # Load model YOLO pretrained. File yolov8n.pt sẽ được tải tự động nếu chưa có.
    model = YOLO(PRETRAINED_MODEL)

    # Train model custom bằng dataset trong data.yaml.
    model.train(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
    )

    print("Train xong.")
    print(f"Model tốt nhất best.pt nằm tại: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
