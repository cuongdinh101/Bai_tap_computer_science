from pathlib import Path

import cv2
import numpy as np


# Script này tạo dataset YOLO mẫu bằng hình vẽ đơn giản.
# Dùng để sinh viên chạy thử pipeline train/predict khi chưa có ảnh thật.

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_SIZE = 640

CLASSES = {
    "bottle": 0,
    "cup": 1,
    "pen": 2,
}


def ensure_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_box(x1, y1, x2, y2):
    # YOLO cần x_center, y_center, width, height được chuẩn hóa về 0-1.
    x_center = ((x1 + x2) / 2) / IMAGE_SIZE
    y_center = ((y1 + y2) / 2) / IMAGE_SIZE
    width = (x2 - x1) / IMAGE_SIZE
    height = (y2 - y1) / IMAGE_SIZE
    return x_center, y_center, width, height


def draw_bottle(image, rng):
    x1 = int(rng.integers(230, 300))
    y1 = int(rng.integers(120, 180))
    w = int(rng.integers(90, 130))
    h = int(rng.integers(300, 380))
    x2 = x1 + w
    y2 = y1 + h

    neck_w = w // 3
    neck_x1 = x1 + w // 2 - neck_w // 2
    neck_x2 = neck_x1 + neck_w
    cv2.rectangle(image, (neck_x1, y1), (neck_x2, y1 + 70), (80, 180, 230), -1)
    cv2.rectangle(image, (x1, y1 + 60), (x2, y2), (80, 180, 230), -1)
    cv2.rectangle(image, (neck_x1, y1 - 25), (neck_x2, y1), (40, 70, 120), -1)
    cv2.rectangle(image, (x1, y1 + 60), (x2, y2), (40, 70, 120), 4)
    cv2.line(image, (x1 + 25, y1 + 90), (x1 + 25, y2 - 30), (255, 255, 255), 6)
    return x1, y1 - 25, x2, y2


def draw_cup(image, rng):
    x1 = int(rng.integers(190, 270))
    y1 = int(rng.integers(180, 250))
    w = int(rng.integers(160, 220))
    h = int(rng.integers(180, 240))
    x2 = x1 + w
    y2 = y1 + h

    color = (70, 160, 240)
    cv2.ellipse(image, ((x1 + x2) // 2, y1), (w // 2, 35), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(image, (x1 + 15, y1), (x2 - 15, y2), color, -1)
    cv2.ellipse(image, ((x1 + x2) // 2, y2), (w // 2 - 20, 25), 0, 0, 360, color, -1)
    cv2.ellipse(image, (x2 + 35, y1 + 90), (45, 70), 0, 0, 360, (60, 60, 60), 10)
    cv2.rectangle(image, (x1 + 15, y1), (x2 - 15, y2), (60, 60, 60), 4)
    return x1, y1 - 35, x2 + 85, y2 + 25


def draw_pen(image, rng):
    x1 = int(rng.integers(110, 190))
    y1 = int(rng.integers(420, 520))
    length = int(rng.integers(330, 430))
    thickness = int(rng.integers(22, 34))
    angle = float(rng.uniform(-0.55, -0.25))

    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))

    color = (40, 80, 220)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5)
    cv2.circle(image, (x1, y1), thickness // 2, (220, 220, 220), -1)
    cv2.circle(image, (x2, y2), thickness // 2, (30, 30, 30), -1)

    pad = thickness + 12
    return min(x1, x2) - pad, min(y1, y2) - pad, max(x1, x2) + pad, max(y1, y2) + pad


def make_image(class_name, index, split):
    rng = np.random.default_rng(abs(hash((class_name, index, split))) % (2**32))
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 245, dtype=np.uint8)

    # Thêm nhiễu nền nhẹ để ảnh không quá giống nhau.
    noise = rng.integers(0, 18, size=image.shape, dtype=np.uint8)
    image = cv2.subtract(image, noise)

    if class_name == "bottle":
        x1, y1, x2, y2 = draw_bottle(image, rng)
    elif class_name == "cup":
        x1, y1, x2, y2 = draw_cup(image, rng)
    else:
        x1, y1, x2, y2 = draw_pen(image, rng)

    x1 = max(0, min(IMAGE_SIZE - 1, x1))
    y1 = max(0, min(IMAGE_SIZE - 1, y1))
    x2 = max(0, min(IMAGE_SIZE - 1, x2))
    y2 = max(0, min(IMAGE_SIZE - 1, y2))

    return image, normalize_box(x1, y1, x2, y2)


def save_sample(split, class_name, index):
    image, box = make_image(class_name, index, split)
    class_id = CLASSES[class_name]

    file_stem = f"{class_name}_{split}_{index:03d}"
    image_path = DATASET_DIR / "images" / split / f"{file_stem}.jpg"
    label_path = DATASET_DIR / "labels" / split / f"{file_stem}.txt"

    cv2.imwrite(str(image_path), image)
    label_path.write_text(
        f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n",
        encoding="utf-8",
    )


def main():
    ensure_dirs()

    for class_name in CLASSES:
        for index in range(1, 11):
            save_sample("train", class_name, index)

        for index in range(1, 4):
            save_sample("val", class_name, index)

    print("Đã tạo dataset YOLO mẫu.")
    print("Train: 10 ảnh/class")
    print("Val: 3 ảnh/class")


if __name__ == "__main__":
    main()
