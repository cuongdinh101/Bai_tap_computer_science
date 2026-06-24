import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
 
print("TensorFlow version:", tf.__version__)
 
# =====================================================================
# 1. TẢI & GIẢI NÉN DATASET cats_and_dogs_filtered
# =====================================================================
_URL = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"

path_to_zip = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=_URL, extract=True)
# TF/Keras cũ trả về đường dẫn file .zip; Keras 3 (TF 2.16+) trả về thư mục
# giải nén ("..._extracted"). Xử lý cả 2 trường hợp để tìm đúng base_dir.
_candidates = [
    os.path.join(path_to_zip, "cats_and_dogs_filtered"),
    os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered"),
]
base_dir = next((c for c in _candidates if os.path.isdir(c)), None)
if base_dir is None:
    raise FileNotFoundError(
        "Không tìm thấy thư mục cats_and_dogs_filtered sau khi giải nén. "
        f"Đã kiểm tra: {_candidates}"
    )
 
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
 
print("Train dir     :", train_dir)
print("Validation dir:", validation_dir)
 
# =====================================================================
# 2. LOAD DỮ LIỆU bằng image_dataset_from_directory
# =====================================================================
BATCH_SIZE = 32
IMG_SIZE = (160, 160)  # 160x160
 
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)
 
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)
 
# Lưu tên class TRƯỚC khi áp dụng prefetch/map (vì các phép biến đổi làm mất .class_names)
class_names = train_dataset.class_names
print("Class names:", class_names)  # ['cats', 'dogs'] -> index 0 = cats, 1 = dogs
 
# ---- Tách 20% validation_dataset thành test_dataset ----
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)        # 20% số batch
validation_dataset = validation_dataset.skip(val_batches // 5)  # 80% còn lại
 
print("Số batch validation:", tf.data.experimental.cardinality(validation_dataset).numpy())
print("Số batch test      :", tf.data.experimental.cardinality(test_dataset).numpy())
 
# =====================================================================
# 3. TỐI ƯU DATASET với AUTOTUNE + prefetch
# =====================================================================
AUTOTUNE = tf.data.AUTOTUNE
 
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
 
# =====================================================================
# 4. DATA AUGMENTATION
# =====================================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
])
 
# =====================================================================
# 5. PRETRAINED MODEL: MobileNetV2
# =====================================================================
IMG_SHAPE = IMG_SIZE + (3,)  # (160, 160, 3)
 
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,       # bỏ phần classifier 1000 lớp của ImageNet
    weights="imagenet",
)
print("Số layers của MobileNetV2:", len(base_model.layers))
 
# =====================================================================
# 6. PREPROCESS dành riêng cho MobileNetV2 (đưa pixel về [-1, 1])
# =====================================================================
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
 
# =====================================================================
# 7. FEATURE EXTRACTION
# =====================================================================
base_model.trainable = False  # đóng băng toàn bộ MobileNetV2
 
# Xây model bằng Functional API
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)             # training=False -> giữ BatchNorm ở chế độ inference
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)         # 1 neuron, KHÔNG sigmoid -> logits
model = tf.keras.Model(inputs, outputs)
 
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
 
# ---- Đánh giá trước khi train (tham khảo) ----
loss0, accuracy0 = model.evaluate(validation_dataset)
print(f"Trước khi train -> loss: {loss0:.2f}, accuracy: {accuracy0:.2f}")
 
# ---- Train 10 epochs ----
initial_epochs = 10
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset,
)
 
# ---- Lưu lịch sử để vẽ ----
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
 
# ---- Vẽ biểu đồ Feature Extraction ----
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Feature Extraction - Training & Validation Accuracy")
 
plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Feature Extraction - Training & Validation Loss")
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig("feature_extraction.png", dpi=120)
plt.show()
 
# =====================================================================
# 8. FINE-TUNING
# =====================================================================
base_model.trainable = True
print("Số layers của base_model:", len(base_model.layers))  # ~154
 
# Đóng băng các layer TRƯỚC layer thứ 100, chỉ train các layer cuối
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
 
# Compile lại với learning rate nhỏ hơn (base_learning_rate / 10) và RMSprop
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
print("Số biến có thể train (trainable variables):", len(model.trainable_variables))
 
# ---- Train tiếp 10 epochs (tổng 20) ----
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
 
history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),  # train tiếp từ epoch 10
    validation_data=validation_dataset,
)
 
# ---- Nối lịch sử của 2 giai đoạn ----
acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]
loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]
 
# ---- Vẽ biểu đồ, đánh dấu mốc bắt đầu fine-tuning ----
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
         label="Start Fine Tuning")
plt.legend(loc="lower right")
plt.title("Training & Validation Accuracy")
 
plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
         label="Start Fine Tuning")
plt.legend(loc="upper right")
plt.title("Training & Validation Loss")
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig("fine_tuning.png", dpi=120)
plt.show()
 
# =====================================================================
# 9. EVALUATE trên test_dataset
# =====================================================================
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")
 
# =====================================================================
# 10. PREDICT trên 1 batch của test_dataset
# =====================================================================
# Lấy 1 batch
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
 
# predict_on_batch -> trả về logits
predictions = model.predict_on_batch(image_batch).flatten()
 
# Output là logits nên cần áp dụng sigmoid để đưa về xác suất [0,1]
predictions = tf.nn.sigmoid(predictions)
# < 0.5 -> class 0 (cats), >= 0.5 -> class 1 (dogs)
predictions = tf.where(predictions < 0.5, 0, 1)
 
print("Predictions:", predictions.numpy())
print("Labels     :", label_batch)
 
# ---- Hiển thị 9 ảnh đầu với nhãn dự đoán ----
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("predictions.png", dpi=120)
plt.show()
 
print("Hoàn tất! Các biểu đồ đã được lưu: "
      "feature_extraction.png, fine_tuning.png, predictions.png")
 