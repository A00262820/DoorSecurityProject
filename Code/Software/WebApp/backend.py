# camera_stream_server.py (Receiver now acts as Server)
from flask import Flask, render_template
import asyncio
import websockets
import base64
import cv2
import numpy as np
import threading
import os

app = Flask(__name__)
frame_global = None
connected_clients = set()

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


async def handle_client(websocket):
    global frame_global, connected_clients
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            frame_bytes = base64.b64decode(message)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame_global = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(frame_global, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            recognized_face = False  # Flag to track if a known face is recognized

            for (x, y, w, h) in faces_detected:
                roi_gray = gray[y:y + h, x:x + w]  # Region of interest (ROI)
                label_id, confidence = recognizer.predict(roi_gray)

                if confidence < 100:  # Adjust confidence threshold as needed
                    name = list(label_map.keys())[list(label_map.values()).index(label_id)]
                    confidence_str = f"{confidence:.2f}"
                    cv2.putText(frame_global, f"{name} ({confidence_str})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame_global, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    recognized_face = True  # Set the flag if a face is recognized
                else:
                    cv2.putText(frame_global, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame_global, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Send WebSocket message to turn on relay if a face is recognized
            if recognized_face:
                try:
                    await websocket.send("face_recognized")
                    print(f"Sent 'face_recognized' to {websocket.remote_address}")
                except websockets.exceptions.ConnectionClosedError:
                    print(f"Connection closed while trying to send 'face_recognized' to {websocket.remote_address}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"Error handling client {websocket.remote_address}: {e}")
    finally:
        connected_clients.remove(websocket)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    global frame_global
    if frame_global is not None:
        _, buffer = cv2.imencode('.jpg', frame_global)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    else:
        return ""

def run_websocket():
    async def main():
        async with websockets.serve(handle_client, "localhost", 8765):
            print("WebSocket server started. Waiting for connections...")
            await asyncio.Future()

    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.get_event_loop().run_until_complete(main())

if __name__ == '__main__':
    threading.Thread(target=run_websocket).start()
    app.run(debug=False, use_reloader=False)