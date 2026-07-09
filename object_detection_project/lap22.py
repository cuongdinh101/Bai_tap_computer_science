import argparse

from predict_yolo import predict
from train_yolo import main as train_main


# File lap22.py này là bản chạy nhanh trong folder nộp.
# Muốn train:   python3 lap22.py --train
# Muốn predict: python3 lap22.py --source test_images/example.jpg


def main():
    parser = argparse.ArgumentParser(description="Lap22 YOLO Object Detection")
    parser.add_argument("--train", action="store_true", help="Train YOLO model")
    parser.add_argument("--source", type=str, help="Đường dẫn ảnh cần predict")
    args = parser.parse_args()

    if args.train:
        train_main()
    elif args.source:
        predict(args.source)
    else:
        print("Cách dùng:")
        print("  Train:   python3 lap22.py --train")
        print("  Predict: python3 lap22.py --source test_images/example.jpg")


if __name__ == "__main__":
    main()
