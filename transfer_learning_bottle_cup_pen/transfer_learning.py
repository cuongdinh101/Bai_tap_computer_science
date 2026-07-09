import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# Cấu hình chung
# =========================
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
TRANSFER_LEARNING_EPOCHS = 10
FINE_TUNING_EPOCHS = 5
FINE_TUNE_LAST_LAYERS = 30

# Lấy đường dẫn thư mục chứa chính file transfer_learning.py.
# Nhờ vậy project vẫn chạy được khi mở ở máy khác, miễn giữ nguyên cấu trúc thư mục.
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
TRANSFER_MODEL_PATH = BASE_DIR / "transfer_learning_model.h5"
FINE_TUNED_MODEL_PATH = BASE_DIR / "fine_tuned_model.h5"
ACCURACY_CHART_PATH = BASE_DIR / "transfer_learning_accuracy.png"


def load_datasets():
    # Load ảnh từ dataset/train. Mỗi thư mục con là một class riêng.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Load ảnh từ dataset/validation để kiểm tra mô hình trong lúc train.
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        str(VALIDATION_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Keras tự lấy class theo tên thư mục, ví dụ: ['bottle', 'cup', 'pen'].
    class_names = train_ds.class_names
    print("Danh sách class:", class_names)

    # Lưu class names để phần predict biết index nào ứng với class nào.
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)

    # Prefetch giúp CPU chuẩn bị batch tiếp theo trong lúc GPU/CPU đang train batch hiện tại.
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, validation_ds, class_names


def build_model(num_classes):
    # Data augmentation tạo thêm biến thể ảnh khi train,
    # giúp mô hình bớt học thuộc dataset nhỏ.
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # MobileNetV2 pretrained đã học đặc trưng từ ImageNet.
    # include_top=False nghĩa là bỏ phần classifier cũ của ImageNet.
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Giai đoạn đầu chỉ train phần classifier mới, chưa train MobileNetV2.
    base_model.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)

    # MobileNetV2 cần ảnh được preprocess về khoảng phù hợp.
    # Không chia ảnh thủ công /255.0 khi đã dùng preprocess_input.
    x = preprocess_input(x)

    # training=False giữ BatchNorm của MobileNetV2 ổn định trong giai đoạn transfer learning.
    x = base_model(x, training=False)

    # GlobalAveragePooling2D gom mỗi feature map thành một giá trị trung bình,
    # nhẹ hơn Flatten và giảm nguy cơ overfitting.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def fine_tune_base_model(base_model):
    # Mở khóa một số layer cuối để mô hình thích nghi tốt hơn với ảnh bottle/cup/pen.
    base_model.trainable = True

    # Đóng băng các layer đầu, chỉ cho phép train 30 layer cuối.
    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    for layer in base_model.layers[-FINE_TUNE_LAST_LAYERS:]:
        layer.trainable = True


def plot_accuracy(transfer_history, fine_tune_history):
    # Gộp accuracy của cả 2 giai đoạn để vẽ một biểu đồ liền mạch.
    train_acc = transfer_history.history["accuracy"] + fine_tune_history.history["accuracy"]
    val_acc = transfer_history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.axvline(
        TRANSFER_LEARNING_EPOCHS,
        color="gray",
        linestyle="--",
        label="Bắt đầu fine-tuning",
    )
    plt.title("Transfer Learning + Fine-tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(ACCURACY_CHART_PATH)
    plt.show()

    print(f"Đã lưu biểu đồ accuracy vào file: {ACCURACY_CHART_PATH}")


def train_model():
    train_ds, validation_ds, class_names = load_datasets()
    model, base_model = build_model(num_classes=len(class_names))

    print("\n===== Giai đoạn 1: Transfer Learning =====")
    compile_model(model, learning_rate=0.001)
    model.summary()

    transfer_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=TRANSFER_LEARNING_EPOCHS,
    )

    model.save(TRANSFER_MODEL_PATH)
    print(f"Đã lưu model transfer learning vào file: {TRANSFER_MODEL_PATH}")

    print("\n===== Giai đoạn 2: Fine-tuning =====")
    fine_tune_base_model(base_model)

    # Fine-tuning dùng learning rate rất nhỏ để không phá hỏng trọng số pretrained.
    compile_model(model, learning_rate=0.00001)

    fine_tune_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNING_EPOCHS,
    )

    model.save(FINE_TUNED_MODEL_PATH)
    print(f"Đã lưu model fine-tuned vào file: {FINE_TUNED_MODEL_PATH}")

    plot_accuracy(transfer_history, fine_tune_history)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def choose_model_path():
    # Ưu tiên model đã fine-tune. Nếu chưa có thì dùng model transfer learning.
    if FINE_TUNED_MODEL_PATH.exists():
        return FINE_TUNED_MODEL_PATH

    if TRANSFER_MODEL_PATH.exists():
        return TRANSFER_MODEL_PATH

    raise FileNotFoundError(
        "Chưa tìm thấy fine_tuned_model.h5 hoặc transfer_learning_model.h5. "
        "Hãy train model trước bằng lệnh: python transfer_learning.py"
    )


def predict_image(image_path):
    class_names = load_class_names()
    model_path = choose_model_path()
    print(f"Đang load model: {model_path}")

    # compile=False giúp load nhanh hơn khi chỉ dùng để predict.
    model = tf.keras.models.load_model(model_path, compile=False)

    # Nếu người dùng nhập đường dẫn tương đối và file không nằm ở thư mục hiện tại,
    # thử tìm theo thư mục project.
    image_path = Path(image_path).expanduser()
    if not image_path.is_absolute() and not image_path.exists():
        image_path = BASE_DIR / image_path

    # Load ảnh mới, resize về đúng 128x128 như lúc train.
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)

    # Thêm chiều batch: (128, 128, 3) -> (1, 128, 128, 3).
    # Không chia /255.0 vì model đã có preprocess_input bên trong.
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    predicted_class = class_names[predicted_index]

    print(f"Ảnh được dự đoán là: {predicted_class}")
    print(f"Độ tin cậy: {confidence:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Learning + Fine-tuning MobileNetV2 cho bottle, cup, pen"
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
