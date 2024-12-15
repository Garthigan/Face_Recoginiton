import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet

# Initialize
facenet = FaceNet()
faces_embeddings = np.load("faces-embeddings_done_2classes.npz")
Y = faces_embeddings['arr_0']  # Correct embeddings
labels = faces_embeddings['arr_1']  # Labels for the faces
#print(Y)
print(labels)
encoder = LabelEncoder()
print(encoder)
print(encoder.fit(labels))

