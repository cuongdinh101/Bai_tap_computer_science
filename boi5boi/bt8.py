import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def convolution(image, kernel):
    h, w = image.shape[:2]
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # padding viền ảnh
    padded = cv.copyMakeBorder(
        image,
        pad_h, pad_h,
        pad_w, pad_w,
        cv.BORDER_CONSTANT,
        value=0
    )

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            if len(image.shape) == 2:
                # ảnh xám
                region = padded[i:i + kh, j:j + kw]
                value = np.sum(region * kernel)
                output[i, j] = np.clip(value, 0, 255)
            else:
                # ảnh màu
                for c in range(3):
                    region = padded[i:i + kh, j:j + kw, c]
                    value = np.sum(region * kernel)
                    output[i, j, c] = np.clip(value, 0, 255)

    return output.astype(np.uint8)


# Đọc ảnh
img = cv.imread("image.jpg")

if img is None:
    print("Khong tim thay anh image.jpg")
    exit()

# Chuyển BGR sang RGB để matplotlib hiển thị đúng màu
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Kernel làm mờ 5x5 tự tạo
kernel = np.ones((5, 5), np.float32) / 25

# 1. Tích chập tự viết
custom_blur = convolution(img_rgb, kernel)

# 2. Dùng filter2D của OpenCV
filter2d = cv.filter2D(img_rgb, -1, kernel)

# 3. Averaging blur
average_blur = cv.blur(img_rgb, (5, 5))

# 4. Gaussian blur
gaussian_blur = cv.GaussianBlur(img_rgb, (5, 5), 0)

# 5. Median blur
median_blur = cv.medianBlur(img_rgb, 5)

# 6. Bilateral filter
bilateral_blur = cv.bilateralFilter(img_rgb, 9, 75, 75)


plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(custom_blur)
plt.title("Custom Convolution")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(filter2d)
plt.title("cv.filter2D")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(average_blur)
plt.title("Averaging Blur")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(bilateral_blur)
plt.title("Bilateral Filter")
plt.axis("off")

plt.tight_layout()
plt.show()