import asyncio
import websockets
import base64
import cv2

async def send_frames():
    uri = "ws://your_websocket_server_address" #replace
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0) #or video file.

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send(frame_base64)
            response = await websocket.recv() #Get processed frame.
            nparr = np.frombuffer(response, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(send_frames())