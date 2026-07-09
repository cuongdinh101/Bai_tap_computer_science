import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


IMAGE_SIZE = 128
TRAIN_COUNT = 30
VALIDATION_COUNT = 10
DATASET_DIR = Path("dataset")


def random_background(draw):
    # Nền sáng, hơi khác nhau để ảnh không quá giống nhau.
    color = (
        random.randint(225, 255),
        random.randint(225, 255),
        random.randint(225, 255),
    )
    draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=color)

    # Thêm vài chấm nhiễu nhẹ.
    for _ in range(40):
        x = random.randint(0, IMAGE_SIZE - 1)
        y = random.randint(0, IMAGE_SIZE - 1)
        gray = random.randint(200, 245)
        draw.point((x, y), fill=(gray, gray, gray))


def rotate_object_layer(layer):
    # Xoay vật thể một góc nhỏ để tạo thêm biến thể.
    angle = random.uniform(-18, 18)
    return layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


def make_bottle():
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)
    random_background(draw)

    layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    cx = random.randint(55, 73)
    top = random.randint(18, 28)
    body_w = random.randint(34, 46)
    body_h = random.randint(58, 72)
    neck_w = random.randint(14, 20)
    color = random.choice([(70, 150, 220, 220), (80, 190, 150, 220), (160, 120, 230, 220)])

    # Nắp chai
    d.rounded_rectangle(
        (cx - neck_w // 2, top - 8, cx + neck_w // 2, top),
        radius=3,
        fill=(50, 80, 120, 255),
    )

    # Cổ chai
    d.rounded_rectangle(
        (cx - neck_w // 2, top, cx + neck_w // 2, top + 24),
        radius=5,
        fill=color,
        outline=(40, 80, 120, 255),
        width=2,
    )

    # Thân chai
    d.rounded_rectangle(
        (cx - body_w // 2, top + 22, cx + body_w // 2, top + 22 + body_h),
        radius=12,
        fill=color,
        outline=(40, 80, 120, 255),
        width=2,
    )

    # Vệt sáng trên chai
    d.line(
        (cx - body_w // 5, top + 34, cx - body_w // 5, top + body_h),
        fill=(255, 255, 255, 120),
        width=4,
    )

    layer = rotate_object_layer(layer)
    image.paste(layer, (0, 0), layer)
    return image.filter(ImageFilter.SMOOTH_MORE)


def make_cup():
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)
    random_background(draw)

    layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    cx = random.randint(54, 72)
    top = random.randint(35, 48)
    cup_w = random.randint(46, 58)
    cup_h = random.randint(42, 54)
    color = random.choice([(245, 120, 90, 255), (245, 185, 70, 255), (90, 170, 235, 255)])

    # Miệng ly
    d.ellipse(
        (cx - cup_w // 2, top, cx + cup_w // 2, top + 18),
        fill=(255, 255, 255, 255),
        outline=(80, 80, 80, 255),
        width=2,
    )

    # Thân ly
    d.rounded_rectangle(
        (cx - cup_w // 2 + 5, top + 8, cx + cup_w // 2 - 5, top + cup_h),
        radius=8,
        fill=color,
        outline=(80, 80, 80, 255),
        width=2,
    )

    # Tay cầm
    handle_x = cx + cup_w // 2 - 4
    d.ellipse(
        (handle_x - 2, top + 16, handle_x + 28, top + 45),
        outline=(80, 80, 80, 255),
        width=6,
    )
    d.ellipse(
        (handle_x + 5, top + 22, handle_x + 20, top + 39),
        fill=(0, 0, 0, 0),
    )

    # Đáy ly
    d.ellipse(
        (cx - cup_w // 2 + 8, top + cup_h - 8, cx + cup_w // 2 - 8, top + cup_h + 6),
        fill=color,
        outline=(80, 80, 80, 255),
        width=2,
    )

    layer = rotate_object_layer(layer)
    image.paste(layer, (0, 0), layer)
    return image.filter(ImageFilter.SMOOTH_MORE)


def make_pen():
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)
    random_background(draw)

    layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    x1 = random.randint(24, 38)
    y1 = random.randint(78, 98)
    length = random.randint(72, 88)
    angle = random.uniform(-0.8, -0.35)
    x2 = int(x1 + length * math.cos(angle))
    y2 = int(y1 + length * math.sin(angle))
    color = random.choice([(30, 80, 210, 255), (30, 30, 30, 255), (210, 50, 70, 255)])

    # Thân bút
    d.line((x1, y1, x2, y2), fill=color, width=10)
    d.line((x1, y1, x2, y2), fill=(255, 255, 255, 110), width=3)

    # Đầu bút
    tip_x = int(x2 + 12 * math.cos(angle))
    tip_y = int(y2 + 12 * math.sin(angle))
    d.line((x2, y2, tip_x, tip_y), fill=(40, 40, 40, 255), width=6)
    d.ellipse((tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3), fill=(10, 10, 10, 255))

    # Đuôi bút
    d.ellipse((x1 - 7, y1 - 7, x1 + 7, y1 + 7), fill=(230, 230, 230, 255), outline=color, width=2)

    # Kẹp bút
    clip_start_x = int(x1 + 20 * math.cos(angle))
    clip_start_y = int(y1 + 20 * math.sin(angle)) - 8
    clip_end_x = int(x1 + 48 * math.cos(angle))
    clip_end_y = int(y1 + 48 * math.sin(angle)) - 8
    d.line((clip_start_x, clip_start_y, clip_end_x, clip_end_y), fill=(80, 80, 80, 255), width=3)

    image.paste(layer, (0, 0), layer)
    return image.filter(ImageFilter.SMOOTH_MORE)


def save_images(split, class_name, count, maker):
    output_dir = DATASET_DIR / split / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, count + 1):
        random.seed(f"{split}-{class_name}-{index}")
        image = maker()
        image.save(output_dir / f"{class_name}_{index:03d}.png")


def main():
    makers = {
        "bottle": make_bottle,
        "cup": make_cup,
        "pen": make_pen,
    }

    for class_name, maker in makers.items():
        save_images("train", class_name, TRAIN_COUNT, maker)
        save_images("validation", class_name, VALIDATION_COUNT, maker)

    print("Đã tạo xong dataset mẫu:")
    print(f"- Train: {TRAIN_COUNT} ảnh/class")
    print(f"- Validation: {VALIDATION_COUNT} ảnh/class")
    print("Bạn có thể chạy: python3 lap18.py")


if __name__ == "__main__":
    main()
