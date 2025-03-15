import cv2
import numpy as np
import os

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognizer (LBPH, Eigenfaces, or Fisherfaces)
recognizer = cv2.face.LBPHFaceRecognizer_create()  # You can change this to EigenFaceRecognizer or FisherFaceRecognizer

# Load training data (images and labels)
def load_training_data(data_dir):
    faces = []
    labels = []
    label_map = {}  # Map label names to numerical IDs
    label_id = 0

    for filename in os.listdir(data_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(data_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            if image is None:
                continue #skip if image is not valid.

            label_name = os.path.splitext(filename)[0] #extract name
            if label_name not in label_map:
                label_map[label_name] = label_id
                label_id += 1
            label = label_map[label_name]

            faces.append(image)
            labels.append(label)

    return faces, np.array(labels), label_map

training_data_dir = "training_faces" # Folder containing training images
if not os.path.exists(training_data_dir):
    print(f"Error: Training data directory '{training_data_dir}' not found. Please create it and add face images.")
    exit()

faces, labels, label_map = load_training_data(training_data_dir)

if len(faces) > 0:
    recognizer.train(faces, labels)
else:
    print("Error: No training faces were loaded")
    exit()

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest (ROI)
        label_id, confidence = recognizer.predict(roi_gray)

        if confidence < 100:  # Adjust confidence threshold as needed
            name = list(label_map.keys())[list(label_map.values()).index(label_id)]
            confidence_str = f"{confidence:.2f}"
            cv2.putText(frame, f"{name} ({confidence_str})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()