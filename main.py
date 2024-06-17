import math
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


def detect_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Chỉ lấy gương mặt đầu tiên
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        return face


def prepare_image(face):
    img_array = image.img_to_array(face)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)


search_image = "data/test.jpg"


model = get_extract_model()
face = detect_face(search_image)
face = cv2.resize(face, (224, 224))  # Đưa về kích thước cần thiết cho VGG16
img_preprocessed = prepare_image(face)
features = model.predict(img_preprocessed)
feature_vector = features.flatten()
vectors = pickle.load(open("data/face-vectors.pkl", "rb"))
paths = pickle.load(open("data/face-paths.pkl", "rb"))
distance: object = np.linalg.norm(vectors - feature_vector, axis=1)
K = 4
ids = np.argsort(distance)[:K]
nearest_image = [(paths[id], distance[id]) for id in ids]

axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10, 5))

for id in range(K):
    draw_image = nearest_image[id]
    print(draw_image[0])
    axes.append(fig.add_subplot(grid_size, grid_size, id + 1))
    axes[-1].set_title(draw_image[1])
    # print(draw_image[0])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()


image_path = cv2.imread(search_image)
gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image_path, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Face Detection', image_path)
cv2.waitKey(0)
cv2.destroyAllWindows()

