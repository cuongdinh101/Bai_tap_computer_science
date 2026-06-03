import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đường dẫn dataset
# Ví dụ:
# dataset/
#   cat/
#     1.jpg
#   dog/
#     1.jpg
DATASET_PATH = "dataset"

def extract_histogram(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    # Chuyển sang HSV để lấy histogram màu tốt hơn
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist.flatten()

X = []
y = []

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)

        try:
            features = extract_histogram(image_path)
            X.append(features)
            y.append(label)
        except:
            print("Lỗi ảnh:", image_path)

X = np.array(X)
y = np.array(y)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test 1 ảnh mới
test_image = "test.jpg"
test_feature = extract_histogram(test_image)
prediction = knn.predict([test_feature])

print("Ảnh test thuộc lớp:", prediction[0])