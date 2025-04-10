# camera_stream_client.py (Sender now acts as Client)
import cv2
import asyncio
import websockets
import base64
import time

async def send_camera_stream():
    uri = "ws://localhost:8765"
    cap = cv2.VideoCapture(0)  # Open the camera only once
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server.")

            async def receive_messages():
                try:
                    async for message in websocket:
                        if message == "face_recognized":
                            print("Server says: Face Recognized!")
                            # Add your logic here to trigger an action
                            # For example, turn on a relay, log the event, etc.
                        else:
                            print(f"Received unknown message from server: {message}")
                except websockets.exceptions.ConnectionClosedOK:
                    print("Server closed connection.")
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"Server closed connection unexpectedly: {e}")
                except Exception as e:
                    print(f"Error receiving message from server: {e}")

            # Start receiving messages in a separate task
            receive_task = asyncio.create_task(receive_messages())

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit loop if no frame is read

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                try:
                    await websocket.send(frame_base64)
                    await asyncio.sleep(0.03)
                except websockets.exceptions.ConnectionClosedOK:
                    print("Server closed connection during send.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"Server closed connection unexpectedly during send: {e}")
                    break
                except Exception as e:
                    print(f"Error sending frame: {e}")
                    break

            print("Camera stream ended.")
            # Cancel the receiving task when the sending loop ends
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    except websockets.exceptions.ConnectionClosedOK:
        print("Server closed connection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cap.release()  # Release the camera in the finally block

if __name__ == "__main__":
    asyncio.run(send_camera_stream())