# camera_stream_server.py (Receiver now acts as Server)
from flask import Flask, render_template
import asyncio
import websockets
import base64
import cv2
import numpy as np
import threading

app = Flask(__name__)
frame_global = None
connected_clients = set()

async def handle_client(websocket):
    global frame_global, connected_clients
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            frame_bytes = base64.b64decode(message)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame_global = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            # Broadcast the frame to all connected clients
            for client in connected_clients:
                if client != websocket and client.open:
                    await client.send(message)

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