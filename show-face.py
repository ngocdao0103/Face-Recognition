import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model
from PIL import Image


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_path = cv2.imread('data/img-face/MS_03_04.jpg')

gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image_path, (x, y), (x + w, y + h), (0, 255, 0), 2)


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


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


model = get_extract_model()
face = detect_face('data/img-face/MS_03_04.jpg')
if face is not None:
    face = cv2.resize(face, (224, 224))
    img_preprocessed = prepare_image(face)
    features = model.predict(img_preprocessed)
    feature_vector = features.flatten()
    print('vector: ', feature_vector)
    cv2.imshow('Face Detection', image_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected.")
