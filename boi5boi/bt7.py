import cv2
import numpy as np

def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            value = np.sum(region * kernel)
            output[i, j] = value

    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)


image = cv2.imread("image.jpg", 0)

if image is None:
    print("Khong tim thay anh image.jpg")
    exit()

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

result = convolution(image, kernel)

cv2.imshow("Anh goc", image)
cv2.imshow("Anh sau tich chap", result)

cv2.waitKey(0)
cv2.destroyAllWindows()