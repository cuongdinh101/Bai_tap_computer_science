# Lap18 - CNN phân loại bottle, cup, pen

Project này dùng Python + TensorFlow/Keras để train mô hình CNN phân loại ảnh 3 lớp:

- `bottle`
- `cup`
- `pen`

## Cấu trúc thư mục

```text
lap18_cnn_bottle_cup_pen/
├── lap18.py
├── requirements.txt
├── generate_sample_dataset.py
├── cnn_bottle_cup_pen_model.h5
├── class_names.json
├── training_accuracy.png
└── dataset/
    ├── train/
    │   ├── bottle/
    │   ├── cup/
    │   └── pen/
    └── validation/
        ├── bottle/
        ├── cup/
        └── pen/
```

## Cách chạy trên máy khác

Sau khi tải project về từ GitHub, mở Terminal trong thư mục `lap18_cnn_bottle_cup_pen`.

Tạo môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## Train model

Chạy:

```bash
python3 lap18.py
```

Chương trình sẽ:

- Load ảnh từ `dataset/train` và `dataset/validation`.
- Resize ảnh về `128x128`.
- Chuẩn hóa pixel về khoảng `0-1`.
- Train mô hình CNN trong `10` epochs.
- Lưu model vào `cnn_bottle_cup_pen_model.h5`.
- Lưu biểu đồ accuracy vào `training_accuracy.png`.

## Predict ảnh mới

Ví dụ predict ảnh cup:

```bash
python3 lap18.py --predict dataset/validation/cup/cup_001.png
```

Ví dụ predict ảnh bottle:

```bash
python3 lap18.py --predict dataset/validation/bottle/bottle_001.png
```

Ví dụ predict ảnh pen:

```bash
python3 lap18.py --predict dataset/validation/pen/pen_001.png
```

## Tạo lại dataset mẫu

Nếu lỡ xóa ảnh mẫu, có thể tạo lại bằng:

```bash
python3 generate_sample_dataset.py
```

File `lap18.py` dùng đường dẫn tương đối theo vị trí của chính nó, nên khi mở trên máy khác chỉ cần giữ nguyên cấu trúc thư mục là chạy được.
