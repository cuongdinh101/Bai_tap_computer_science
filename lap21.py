from ultralytics import YOLO


# ==================================================
# LAB 21 - YOLO IMAGE CLASSIFICATION
# Bài toán: Phân loại ảnh bottle / cup / pen
# ==================================================


# =========================
# BƯỚC 1: LOAD MODEL
# =========================
# Load model YOLO classification đã được pretrained
# -cls nghĩa là model dùng cho bài toán classification
# .pt là file trọng số PyTorch
model = YOLO("yolo26n-cls.pt")


# =========================
# BƯỚC 2: TRAIN MODEL
# =========================
# data: đường dẫn tới dataset
# epochs: số vòng lặp học
# imgsz: kích thước ảnh đầu vào
# batch: số ảnh đưa vào model trong 1 lần train
results = model.train(
    data="my_dataset",
    epochs=50,
    imgsz=224,
    batch=16
)


# =========================
# BƯỚC 3: LOAD MODEL TỐT NHẤT
# =========================
# Sau khi train, YOLO sẽ lưu model tốt nhất ở thư mục runs/classify/train/weights/best.pt
best_model = YOLO("runs/classify/train/weights/best.pt")


# =========================
# BƯỚC 4: VALIDATE MODEL
# =========================
# Kiểm tra độ chính xác của model trên tập validation
metrics = best_model.val()

print("Top-1 Accuracy:", metrics.top1)
print("Top-5 Accuracy:", metrics.top5)


# =========================
# BƯỚC 5: PREDICT ẢNH MỚI
# =========================
# Thay test_image.jpg bằng đường dẫn ảnh bạn muốn dự đoán
predict_results = best_model("test_image.jpg")


# =========================
# BƯỚC 6: IN KẾT QUẢ DỰ ĐOÁN
# =========================
for result in predict_results:
    # result.probs chứa xác suất của tất cả các class
    top1 = result.probs.top1

    # Độ tin cậy của class có xác suất cao nhất
    top1_conf = result.probs.top1conf

    # Tên class dự đoán
    top1_name = result.names[top1]

    print("Ảnh được dự đoán là:", top1_name)
    print("Độ tin cậy:", top1_conf.item())


# =========================
# BƯỚC 7: EXPORT MODEL
# =========================
# Xuất model sang ONNX để có thể dùng ở môi trường khác
best_model.export(format="onnx")