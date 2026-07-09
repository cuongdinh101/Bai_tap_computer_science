import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# PREDICT YOLO OBJECT DETECTION
# In class, confidence, bounding box và lưu ảnh kết quả
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
PREDICTIONS_DIR = BASE_DIR / "predictions"


def resolve_source_path(source):
    # Nếu người dùng nhập đường dẫn tương đối, ưu tiên tìm từ thư mục hiện tại.
    # Nếu không thấy, thử tìm từ thư mục project.
    source_path = Path(source).expanduser()
    if not source_path.is_absolute() and not source_path.exists():
        source_path = BASE_DIR / source_path
    return source_path


def predict(source):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy runs/detect/train/weights/best.pt. "
            "Hãy train trước bằng lệnh: python3 train_yolo.py"
        )

    source_path = resolve_source_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {source_path}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BEST_MODEL_PATH))
    results = model.predict(source=str(source_path), conf=0.25, verbose=False)

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("Không phát hiện đối tượng nào.")
        else:
            print("Các object phát hiện được:")

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print("----------------------------")
            print("Class name:", class_name)
            print(f"Confidence: {confidence:.2%}")
            print(f"Bounding box: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")

        # Vẽ bounding box lên ảnh và lưu vào thư mục predictions/.
        annotated_image = result.plot()
        output_path = PREDICTIONS_DIR / f"{source_path.stem}_prediction.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Đã lưu ảnh kết quả tại: {output_path}")

        # Hiển thị ảnh nếu môi trường hiện tại hỗ trợ giao diện.
        try:
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("YOLO Object Detection Result")
            plt.show()
        except Exception as error:
            print(f"Không thể hiển thị ảnh trực tiếp: {error}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict ảnh bằng YOLO Object Detection")
    parser.add_argument("--source", required=True, help="Đường dẫn ảnh cần predict")
    args = parser.parse_args()

    predict(args.source)


if __name__ == "__main__":
    main()
