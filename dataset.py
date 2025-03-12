import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract features from images
def extract_features(images, labels):
    features = []
    feature_labels = []  # To store labels corresponding to extracted features
    for img, label in zip(images, labels):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if label == 'face':  # Only extract features from face images
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (100, 100))  # Resize to a fixed size
                features.append(face_resized.flatten())  # Flatten the image
                feature_labels.append(label)  # Append the corresponding label
    return np.array(features), np.array(feature_labels)

# Load images from the dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append('face' if 'faces' in folder else 'non_face')  # Assign label based on folder
    return images, labels

# Load the dataset
faces_folder = 'dataset/faces'
non_faces_folder = 'dataset/non_faces'

faces_images, faces_labels = load_images_from_folder(faces_folder)
non_faces_images, non_faces_labels = load_images_from_folder(non_faces_folder)

# Combine images and labels
images = faces_images + non_faces_images
labels = faces_labels + non_faces_labels

# Extract features and labels
X, y = extract_features(images, labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)  # Set n_neighbors to 1
knn.fit(X_train, y_train)

# Test the classifier
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)

        # Check if the face_resized has the correct shape
        if face_resized.shape[1] == X_train.shape[1]:  # Ensure it matches the training data
            label = knn.predict(face_resized)
            # Draw rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and predict faces in the frame
    output_frame = detect_and_predict(frame)
 # Display the resulting frame
    cv2.imshow('Face Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' is pressed
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()