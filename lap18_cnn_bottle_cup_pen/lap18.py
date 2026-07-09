import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# =========================
# Cấu hình chung
# =========================
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Lấy đường dẫn thư mục chứa chính file lap18.py.
# Nhờ vậy mở ở máy khác vẫn chạy được, miễn là dataset nằm cùng thư mục với file này.
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
MODEL_PATH = BASE_DIR / "cnn_bottle_cup_pen_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
ACCURACY_CHART_PATH = BASE_DIR / "training_accuracy.png"
DEFAULT_CLASS_NAMES = ["bottle", "cup", "pen"]


def train_model():
    # Load ảnh từ thư mục dataset/train.
    # Mỗi thư mục con là một class: bottle, cup, pen.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Load ảnh từ thư mục dataset/validation.
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        str(VALIDATION_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Lấy tên class theo đúng thứ tự nhãn mà Keras tạo ra.
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Danh sách class:", class_names)

    # Lưu tên class để lúc predict biết index 0, 1, 2 ứng với class nào.
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)

    # Chuẩn hóa pixel ảnh từ khoảng 0-255 về khoảng 0-1.
    train_ds = train_ds.map(lambda images, labels: (images / 255.0, labels))
    validation_ds = validation_ds.map(lambda images, labels: (images / 255.0, labels))

    # Prefetch giúp quá trình đọc dữ liệu và train nhanh hơn.
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Xây dựng mô hình CNN đơn giản.
    model = models.Sequential(
        [
            layers.Input(shape=(128, 128, 3)),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model với Adam, loss sparse_categorical_crossentropy và accuracy.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Train model trong 10 epochs.
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
    )

    # Vẽ biểu đồ training accuracy và validation accuracy.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training và Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(ACCURACY_CHART_PATH)
    plt.show()

    # Lưu model thành file .h5.
    model.save(MODEL_PATH)
    print(f"Đã lưu model vào file: {MODEL_PATH}")
    print(f"Đã lưu biểu đồ vào file: {ACCURACY_CHART_PATH}")


def load_class_names():
    # Nếu đã train, file class_names.json sẽ chứa đúng thứ tự class.
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Không tìm thấy class_names.json, dùng thứ tự mặc định.")
        return DEFAULT_CLASS_NAMES


def predict_image(image_path):
    class_names = load_class_names()

    # Load model đã train.
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load ảnh mới, resize về 128x128 giống lúc train.
    image_path = Path(image_path)
    if not image_path.is_absolute() and not image_path.exists():
        image_path = BASE_DIR / image_path

    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)

    # Chuẩn hóa pixel về khoảng 0-1.
    image_array = image_array / 255.0

    # Thêm chiều batch: (128, 128, 3) -> (1, 128, 128, 3).
    image_array = np.expand_dims(image_array, axis=0)

    # Dự đoán class.
    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    predicted_class = class_names[predicted_index]

    print(f"Ảnh được dự đoán là: {predicted_class}")
    print(f"Độ tin cậy: {confidence:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train hoặc predict CNN phân loại bottle, cup, pen"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Đường dẫn ảnh cần dự đoán. Nếu không nhập, chương trình sẽ train model.",
    )
    args = parser.parse_args()

    if args.predict:
        predict_image(args.predict)
    else:
        train_model()


if __name__ == "__main__":
    main()
