import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
from keras.models import load_model
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


data_folder = "data/img-face"

# Khoi tao model
model = get_extract_model()

vectors = []
paths = []

for image_path in os.listdir(data_folder):
    image_path_full = os.path.join(data_folder, image_path)
    face = detect_face(image_path_full)
    face = cv2.resize(face, (224, 224))  # Đưa về kích thước cần thiết cho VGG16
    img_preprocessed = prepare_image(face)
    features = model.predict(img_preprocessed)
    feature_vector = features.flatten()
    print('vector: ', feature_vector, 'file name:', image_path_full)
    vectors.append(feature_vector)
    paths.append(image_path_full)

# save vao file
vector_file = "data/vectors.pkl"
path_file = "data/paths.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))
