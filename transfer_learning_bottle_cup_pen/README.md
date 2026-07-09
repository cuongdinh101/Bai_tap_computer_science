# Transfer Learning và Fine-tuning phân loại bottle, cup, pen

Project Python dùng TensorFlow/Keras và MobileNetV2 pretrained để phân loại ảnh 3 lớp:

- `bottle`
- `cup`
- `pen`

## 1. Transfer Learning là gì?

Transfer Learning là kỹ thuật dùng lại một mô hình đã được train trước trên dataset lớn, ví dụ ImageNet. Thay vì train CNN từ đầu, ta tận dụng phần trích xuất đặc trưng của mô hình pretrained rồi chỉ train thêm phần phân loại mới cho bài toán của mình.

Trong project này, MobileNetV2 đã học nhiều đặc trưng ảnh cơ bản như cạnh, màu sắc, texture, hình dạng. Ta dùng lại các đặc trưng đó để phân loại `bottle`, `cup`, `pen`.

## 2. Fine-tuning là gì?

Fine-tuning là bước train tiếp một phần nhỏ của mô hình pretrained sau khi đã train classifier mới. Ở đây, sau giai đoạn Transfer Learning, chương trình mở khóa 30 layer cuối của MobileNetV2 và train thêm 5 epochs.

Mục đích là giúp mô hình thích nghi tốt hơn với dataset cụ thể của bài toán.

## 3. Vì sao dùng MobileNetV2 pretrained?

MobileNetV2 nhẹ, nhanh và phù hợp cho bài toán học tập hoặc máy cấu hình vừa phải. Vì đã được pretrained trên ImageNet, mô hình có sẵn khả năng nhận biết nhiều đặc trưng hình ảnh, giúp train nhanh hơn và thường cho kết quả tốt hơn so với train CNN từ đầu khi dataset nhỏ.

## 4. Vì sao ban đầu phải đóng băng base model?

Ban đầu ta đặt:

```python
base_model.trainable = False
```

Lý do là để giữ nguyên kiến thức đã học từ ImageNet. Khi dataset nhỏ, nếu train toàn bộ MobileNetV2 ngay từ đầu, trọng số pretrained có thể bị thay đổi quá mạnh và mô hình dễ học sai hoặc overfitting.

## 5. Vì sao fine-tuning dùng learning rate nhỏ?

Fine-tuning chỉ nên điều chỉnh nhẹ các trọng số pretrained, nên dùng learning rate nhỏ:

```python
learning_rate=0.00001
```

Nếu learning rate quá lớn, mô hình có thể phá hỏng các đặc trưng tốt đã học từ ImageNet.

## 6. Vì sao dùng GlobalAveragePooling2D thay cho Flatten?

`Flatten` biến toàn bộ feature map thành một vector rất dài, dễ tạo ra nhiều tham số và dễ overfitting.

`GlobalAveragePooling2D` lấy trung bình mỗi feature map, giúp mô hình gọn hơn, ít tham số hơn và thường phù hợp hơn khi dùng pretrained CNN.

## 7. Cấu trúc dataset

Đặt ảnh theo cấu trúc:

```text
dataset/
├── train/
│   ├── bottle/
│   ├── cup/
│   └── pen/
│
└── validation/
    ├── bottle/
    ├── cup/
    └── pen/
```

Project đã có sẵn dataset mẫu trong thư mục `dataset`.

## 8. Cài thư viện

Tạo môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Lưu ý: lần train đầu tiên có thể cần internet để TensorFlow tải weights ImageNet của MobileNetV2. Các lần sau nếu weights đã được cache thì không cần tải lại.

## 9. Train model

Chạy:

```bash
python3 transfer_learning.py
```

Chương trình sẽ chạy 2 giai đoạn:

- Giai đoạn 1: Transfer Learning trong 10 epochs, lưu `transfer_learning_model.h5`.
- Giai đoạn 2: Fine-tuning trong 5 epochs, lưu `fine_tuned_model.h5`.
- Lưu class names vào `class_names.json`.
- Lưu biểu đồ accuracy vào `transfer_learning_accuracy.png`.

## 10. Predict ảnh mới

Sau khi train xong, chạy:

```bash
python3 transfer_learning.py --predict dataset/validation/cup/cup_001.png
```

Ví dụ khác:

```bash
python3 transfer_learning.py --predict dataset/validation/bottle/bottle_001.png
python3 transfer_learning.py --predict dataset/validation/pen/pen_001.png
```

Khi predict, chương trình sẽ ưu tiên load `fine_tuned_model.h5`. Nếu chưa có file này thì load `transfer_learning_model.h5`.

## 11. Ghi chú cho sinh viên mới học

Trong code, ảnh không được chia thủ công `/255.0`, vì MobileNetV2 dùng hàm:

```python
tf.keras.applications.mobilenet_v2.preprocess_input
```

Hàm này đã xử lý ảnh đúng kiểu MobileNetV2 cần.
