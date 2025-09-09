import cv2
import numpy as np
import os

# Load the better face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Load the trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Load labels
labels = {}
if os.path.exists("labels.txt"):
    with open("labels.txt", "r") as f:
        for line in f:
            id_, name = line.strip().split(',')
            labels[int(id_)] = name

# Define fixed image size (must match training size)
IMG_SIZE = (100, 100)

# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        print("❌ No faces detected.")  # Debugging message

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, IMG_SIZE)  # Resize for proper recognition

        # Recognize face
        label_id, confidence = recognizer.predict(roi_gray)

        name = "Unknown"
        if confidence < 100:  # Increase threshold
            name = labels.get(label_id, "Unknown")

        print(f"✅ Detected {name} with confidence: {confidence}")  # Debugging message

        # Draw rectangle and display name
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"{name} ({int(confidence)})", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
