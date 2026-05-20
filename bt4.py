import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. ĐỌC ẢNH
# =========================
# Đổi tên ảnh cho đúng với file của bạn
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Không đọc được ảnh. Kiểm tra lại tên file!")
    exit()

# =========================
# 2. HÀM HIỂN THỊ ẢNH
# =========================
def show_image(title, image):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# =========================
# 3. NEGATIVE TRANSFORMATION
# Công thức: s = 255 - r
# =========================
negative = 255 - img

# =========================
# 4. BRIGHTNESS TRANSFORMATION
# Tăng sáng: s = r + b
# Giảm sáng: s = r - b
# =========================
bright = np.clip(img + 50, 0, 255).astype(np.uint8)
dark = np.clip(img - 50, 0, 255).astype(np.uint8)

# Cách khác bằng OpenCV:
bright_cv = cv2.convertScaleAbs(img, alpha=1, beta=50)
dark_cv = cv2.convertScaleAbs(img, alpha=1, beta=-50)

# =========================
# 5. CONTRAST TRANSFORMATION
# Công thức: s = alpha * r + beta
# alpha > 1: tăng tương phản
# 0 < alpha < 1: giảm tương phản
# =========================
high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

# =========================
# 6. CONTRAST QUANH MỐC 128
# Công thức: s = a * (r - 128) + 128
# =========================
a = 1.5
contrast_128 = a * (img.astype(np.float32) - 128) + 128
contrast_128 = np.clip(contrast_128, 0, 255).astype(np.uint8)

# =========================
# 7. THRESHOLDING
# Nếu r >= T thì s = 255
# Nếu r < T thì s = 0
# =========================
T = 127
_, threshold = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

# =========================
# 8. LOG TRANSFORMATION
# Công thức: s = c * log(1 + r)
# =========================
c = 255 / np.log(1 + np.max(img))
log_transform = c * np.log(1 + img.astype(np.float32))
log_transform = np.clip(log_transform, 0, 255).astype(np.uint8)

# =========================
# 9. GAMMA TRANSFORMATION
# Công thức: s = 255 * (r / 255)^gamma
# gamma < 1: sáng hơn
# gamma > 1: tối hơn
# =========================
gamma = 2.0
gamma_transform = 255 * ((img / 255) ** gamma)
gamma_transform = np.clip(gamma_transform, 0, 255).astype(np.uint8)

gamma2 = 0.5
gamma_bright = 255 * ((img / 255) ** gamma2)
gamma_bright = np.clip(gamma_bright, 0, 255).astype(np.uint8)

# =========================
# 10. HIỂN THỊ KẾT QUẢ
# =========================
show_image("Original Image", img)
show_image("Negative Image", negative)
show_image("Bright Image", bright)
show_image("Dark Image", dark)
show_image("High Contrast", high_contrast)
show_image("Low Contrast", low_contrast)
show_image("Contrast Around 128", contrast_128)
show_image("Threshold Image", threshold)
show_image("Log Transformation", log_transform)
show_image("Gamma = 2.0", gamma_transform)
show_image("Gamma = 0.5", gamma_bright)

# =========================
# 11. LƯU ẢNH RA FILE
# =========================
cv2.imwrite("negative.jpg", negative)
cv2.imwrite("bright.jpg", bright)
cv2.imwrite("dark.jpg", dark)
cv2.imwrite("high_contrast.jpg", high_contrast)
cv2.imwrite("low_contrast.jpg", low_contrast)
cv2.imwrite("contrast_128.jpg", contrast_128)
cv2.imwrite("threshold.jpg", threshold)
cv2.imwrite("log_transform.jpg", log_transform)
cv2.imwrite("gamma_2.jpg", gamma_transform)
cv2.imwrite("gamma_05.jpg", gamma_bright)

print("Đã xử lý xong ảnh!")