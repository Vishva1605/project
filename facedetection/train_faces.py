import cv2
import numpy as np
import os

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

dataset_path = "dataset"
images, labels = [], []
label_dict = {}
current_id = 0

# Set fixed size for face images
IMG_SIZE = (100, 100)  

# Loop through dataset folder
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_id] = person  # Assign an ID to the name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, IMG_SIZE)  # Resize face to match training size
            images.append(face_roi)
            labels.append(current_id)

    current_id += 1  # Increment ID for the next person

# Convert lists to numpy arrays
images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Train the recognizer
recognizer.train(images, labels)
recognizer.save('trained_model.yml')

# Save labels
with open("labels.txt", "w") as f:
    for id_, name in label_dict.items():
        f.write(f"{id_},{name}\n")

print("✅ Training complete. Model saved as 'trained_model.yml'.")
