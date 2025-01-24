# Door Security System

This project is a Door Security System using an ESP32-CAM and a web application for monitoring and controlling access. The system streams video from the ESP32-CAM to a server using WebSocket, while simultaneously performing face recognition in the background to send signals for opening the door.

### Features

- Live Video Streaming: The ESP32-CAM streams real-time video to a server.

- Face Recognition: The server processes the video feed to identify authorized faces.

- Door Control: The system sends a signal to open the door when an authorized face is recognized.

- Web Interface: A web application displays the live video feed and provides a user interface for monitoring.

### Components

#### Hardware

- ESP32-CAM

- Door lock (electronic)

- Power supply

- Relay module (for door lock control)

- Supporting peripherals (e.g., cables, connectors)

#### Software

- Arduino IDE (for programming the ESP32-CAM)

- WebSocket server (for video streaming and communication)

- Web application (for monitoring and control)

### Usage

- Power on the ESP32-CAM and ensure it connects to the configured Wi-Fi.

- Start the WebSocket server.

- Open the web application in a browser to view the live video feed.

- The system will automatically recognize authorized faces and send a signal to open the door.
