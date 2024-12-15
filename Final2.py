import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet

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

# Load the pre-trained Haar Cascade for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the trained SVM model for face recognition
model = joblib.load('svm_face_recognition_model1.pkl')

# Start capturing video
cap = cv.VideoCapture(0)

# Define a function to calculate similarity between the detected face embedding and the stored embeddings
def calculate_similarity(embedding1, embeddings2):
    # Cosine similarity calculation
    similarity = np.dot(embedding1, embeddings2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embeddings2, axis=1))
    return similarity

# Start the video capture loop
while cap.isOpened():
    _, frame = cap.read()
    
    # Convert the frame to RGB and grayscale for face detection
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        # Extract the face from the frame
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))  # Resize to match FaceNet's input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Get the embedding for the detected face
        ypred = facenet.embeddings(img)
        
        # Calculate the similarity between the detected face and the known faces
        similarities = calculate_similarity(ypred, Y)

        # Flatten similarities array (from (1, 20) to (20,))
        similarities = similarities.flatten()

        # Get the index of the best match (highest similarity)
        best_match_idx = np.argmax(similarities)
        max_similarity = similarities[best_match_idx]

        # Check if the max similarity is above a threshold
        if max_similarity > 0.6:  # Adjust the threshold as needed
            # Predict the label using the SVM model
            face_idx = model.predict(ypred)[0]  # Predicted encoded value

            # Check if the encoded value matches any valid label
            if face_idx in encoded_labels:
                predicted_name = encoder.inverse_transform([face_idx])[0]
            else:
                # If the label is not found, classify as "Visitor"
                predicted_name = "Visitor"
        else:
            predicted_name = "Visitor"
        
        # Draw a rectangle around the face and display the name (or "Visitor")
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
        cv.putText(frame, str(predicted_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

    # Display the frame
    cv.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv.destroyAllWindows()
