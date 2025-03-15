from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*") # Important for cross-origin requests

frame_global = None #global variable to store the frame.

@app.route('/')
def index():
    return render_template('index.html')  # Create an index.html file

@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on('video_stream')
def handle_video_stream(data):
    global frame_global #access global varible.
    try:
        # Decode the base64 encoded image
        img_data = base64.b64decode(data.split(',')[1])  # remove data:image/jpeg;base64,
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame_global = frame #store the frame in global variable.
        else:
            print("Error decoding frame")

    except Exception as e:
        print(f"Error processing video stream: {e}")

def display_video():
    global frame_global #access global varible.
    while True:
        if frame_global is not None:
            cv2.imshow('Received Video', frame_global)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import threading
    display_thread = threading.Thread(target=display_video) #run display on seperate thread.
    display_thread.daemon = True #close thread when main thread closes.
    display_thread.start()
    socketio.run(app, debug=True, host='0.0.0.0') # make accessible from network