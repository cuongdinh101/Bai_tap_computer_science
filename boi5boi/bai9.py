import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Đọc ảnh xám
img = cv.imread("image.jpg", cv.IMREAD_GRAYSCALE)

if img is None:
    print("Khong tim thay anh image.jpg")
    exit()


# 1. Laplacian: phát hiện cạnh theo mọi hướng
laplacian = cv.Laplacian(img, cv.CV_64F)
laplacian_abs = np.uint8(np.absolute(laplacian))


# 2. Sobel X: phát hiện cạnh theo chiều dọc
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobelx_abs = np.uint8(np.absolute(sobelx))


# 3. Sobel Y: phát hiện cạnh theo chiều ngang
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
sobely_abs = np.uint8(np.absolute(sobely))


# 4. Kết hợp Sobel X và Sobel Y
sobel_combined = cv.magnitude(sobelx, sobely)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))


# 5. Scharr X, Scharr Y: giống Sobel nhưng mạnh hơn với kernel 3x3
scharrx = cv.Scharr(img, cv.CV_64F, 1, 0)
scharry = cv.Scharr(img, cv.CV_64F, 0, 1)

scharrx_abs = np.uint8(np.absolute(scharrx))
scharry_abs = np.uint8(np.absolute(scharry))

scharr_combined = cv.magnitude(scharrx, scharry)
scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))


# Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(laplacian_abs, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(sobelx_abs, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(sobely_abs, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Sobel Combined")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(scharrx_abs, cmap="gray")
plt.title("Scharr X")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(scharry_abs, cmap="gray")
plt.title("Scharr Y")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(scharr_combined, cmap="gray")
plt.title("Scharr Combined")
plt.axis("off")

plt.tight_layout()
plt.show()