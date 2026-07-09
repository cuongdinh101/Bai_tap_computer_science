# Object Detection dùng YOLO cho bottle, cup, pen

Project này dùng Python, Ultralytics YOLO, OpenCV và Matplotlib để phát hiện 3 loại vật thể trong ảnh:

- `bottle`
- `cup`
- `pen`

## 1. Object Detection Là Gì?

Object Detection là bài toán vừa dự đoán vật thể thuộc class nào, vừa xác định vị trí vật thể trong ảnh bằng bounding box.

Ví dụ, model không chỉ nói ảnh có `cup`, mà còn vẽ được khung quanh vị trí của `cup`.

## 2. Khác Image Classification Ở Điểm Nào?

Image Classification chỉ dự đoán ảnh thuộc class nào.

Object Detection vừa dự đoán class, vừa xác định vị trí object bằng bounding box. Một ảnh có thể có nhiều object khác nhau.

## 3. Bounding Box Là Gì?

Bounding box là khung chữ nhật bao quanh object trong ảnh.

Khi predict, YOLO thường trả về bounding box theo dạng pixel:

```text
x1, y1, x2, y2
```

Trong đó:

- `x1, y1`: góc trên bên trái
- `x2, y2`: góc dưới bên phải

## 4. YOLO Là Gì?

YOLO là mô hình object detection một bước, nhanh và phù hợp để demo. YOLO xử lý ảnh một lần và dự đoán luôn class + bounding box.

## 5. YOLO Format Là Gì?

Mỗi ảnh cần có một file label `.txt` cùng tên.

Ví dụ:

```text
dataset/images/train/cup_001.jpg
dataset/labels/train/cup_001.txt
```

Mỗi dòng trong file label có dạng:

```text
class_id x_center y_center width height
```

Tất cả tọa độ phải được normalized từ `0` đến `1`.

Class trong bài này:

```text
0: bottle
1: cup
2: pen
```

Ví dụ label:

```text
1 0.500000 0.500000 0.300000 0.400000
```

Nghĩa là object class `cup`, tâm box ở giữa ảnh, box rộng 30% ảnh và cao 40% ảnh.

## 6. Annotate Ảnh Bằng Công Cụ Nào?

Bạn có thể dùng:

- Roboflow: upload ảnh, vẽ bounding box trên web, export YOLO format.
- LabelImg: công cụ local để vẽ bounding box và lưu label YOLO.

Khi annotate, cần đảm bảo mỗi ảnh có file `.txt` cùng tên và label đúng format YOLO.

## 7. Cấu Trúc Dataset

```text
object_detection_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── train_yolo.py
├── predict_yolo.py
├── lap22.py
├── requirements.txt
└── README.md
```

Nội dung `dataset/data.yaml`:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: bottle
  1: cup
  2: pen
```

Project này có sẵn dataset mẫu đơn giản để chạy thử. Khi làm bài thật, bạn nên thay bằng ảnh thật đã annotate.

Nếu muốn tạo lại dataset mẫu, chạy:

```bash
python3 create_sample_dataset.py
```

## 8. Cài Thư Viện

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Train Model

Chạy trong thư mục `object_detection_project`:

```bash
python3 train_yolo.py
```

Hoặc chạy qua file lap22:

```bash
python3 lap22.py --train
```

Cấu hình train:

- Model pretrained: `yolov8n.pt`
- Image size: `640`
- Epochs: `50`
- Batch size: `8`
- Kết quả lưu trong: `runs/detect/train`
- Model tốt nhất: `runs/detect/train/weights/best.pt`

## 10. Predict Ảnh Mới

Sau khi train xong, chạy:

```bash
python3 predict_yolo.py --source test_images/example.jpg
```

Hoặc chạy qua file lap22:

```bash
python3 lap22.py --source test_images/example.jpg
```

Hoặc predict ảnh trong dataset mẫu:

```bash
python3 predict_yolo.py --source dataset/images/val/cup_val_001.jpg
```

Chương trình sẽ:

- Load model từ `runs/detect/train/weights/best.pt`
- In ra class name, confidence, bounding box
- Lưu ảnh có vẽ bounding box vào thư mục `predictions/`

## 11. Ghi Chú Cho Sinh Viên Mới Học

Nếu file label bị sai format hoặc thiếu file label cùng tên ảnh, YOLO sẽ không train đúng.

Ví dụ đúng:

```text
dataset/images/train/bottle_001.jpg
dataset/labels/train/bottle_001.txt
```

Tên file ảnh và file label phải giống nhau, chỉ khác đuôi `.jpg` và `.txt`.
