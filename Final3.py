import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet
from ultralytics import YOLO

# Initialize the FaceNet model
facenet = FaceNet()

# Load the embeddings and training labels
faces_embeddings = np.load("faces-embeddings_done_2classes.npz")
Y = faces_embeddings['arr_0']  # Assuming arr_0 contains the embeddings (features)
labels = faces_embeddings['arr_1']  # Assuming arr_1 contains the labels (names)

# Check the shape of Y and labels
print("Shape of embeddings:", Y.shape)
print("Labels:", labels)

# Train the encoder with the correct labels (names)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)  # Encode the labels (e.g., 0, 1)

print("Encoded Labels:", encoded_labels)
print("Classes:", encoder.classes_)


model = YOLO("yolov8n-face.pt") 


svm_model = joblib.load('svm_face_recognition_model1.pkl')

# Start capturing video
cap = cv.VideoCapture(0)

# Define a function to calculate similarity between the detected face embedding and the stored embeddings
def calculate_similarity(embedding1, embeddings2):
    similarity = np.dot(embedding1, embeddings2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embeddings2, axis=1))
    return similarity

# Start the video capture loop
while cap.isOpened():
    _, frame = cap.read()
    
    # Convert the frame to RGB for YOLOv8 model
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Perform face detection using YOLOv8
    results = model.predict(source=frame, show=False, save=False)

    # Loop through detected faces (results.boxes contains bounding boxes)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates

            # Extract the face from the frame
            face_img = rgb_img[y1:y2, x1:x2]
            face_img = cv.resize(face_img, (160, 160))  # Resize to match FaceNet's input size
            face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

            # Get the embedding for the detected face
            ypred = facenet.embeddings(face_img)

            # Calculate the similarity between the detected face and the known faces
            similarities = calculate_similarity(ypred, Y)

            # Flatten similarities array (from (1, 20) to (20,))
            similarities = similarities.flatten()

            # Get the index of the best match (highest similarity)
            best_match_idx = np.argmax(similarities)
            max_similarity = similarities[best_match_idx]

            # Check if the max similarity is above a threshold
            if max_similarity > 0.6:  # Adjust the threshold as needed
                face_idx = svm_model.predict(ypred)[0]  # Predicted encoded value

                # Check if the encoded value matches any valid label
                if face_idx in encoded_labels:
                    predicted_name = encoder.inverse_transform([face_idx])[0]
                else:
                    # If the label is not found, classify as "Visitor"
                    predicted_name = "Visitor"
            else:
                predicted_name = "Visitor"
            
            # Draw a rectangle around the face and display the name (or "Visitor")
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 10)
            cv.putText(frame, str(predicted_name), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
