#include <WiFi.h>
#include <esp_camera.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <WebSocketsClient.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// WebSocket server details
const char* wsServer = "YOUR_WEBSOCKET_SERVER_IP"; // Replace with your server IP or hostname
const int wsPort = YOUR_WEBSOCKET_SERVER_PORT;     // Replace with your server port
const char* wsPath = "/ws";                       // Replace with your WebSocket path if needed

// Relay pin (adjust to your setup)
const int relayPin = 2;

// Camera configuration
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Global variables
WebSocketsClient webSocket;
bool faceRecognized = false;

// Function prototypes
void setupCamera();
void connectWiFi();
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length);
void sendFrameWebSocket();
void processRecognitionResponse(const String& response);
void controlLock(bool open);

void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32-CAM WebSocket Client");

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW); // Initialize lock as closed (assuming LOW activates the solenoid)

  connectWiFi();
  setupCamera();

  // WebSocket event handler
  webSocket.onEvent(webSocketEvent);

  // Attempt to connect to the WebSocket server
  webSocket.begin(wsServer, wsPort, wsPath);
  Serial.printf("Connecting to WebSocket server: %s:%d%s\n", wsServer, wsPort, wsPath);
}

void loop() {
  webSocket.loop();

  // Capture and send frame periodically
  static unsigned long lastTime = 0;
  unsigned long currentTime = millis();
  if (currentTime - lastTime >= 1000) { // Send frame every 1 second (adjust as needed)
    sendFrameWebSocket();
    lastTime = currentTime;
  }

  // Control the lock based on face recognition status
  controlLock(faceRecognized);

  delay(10); // Small delay to prevent watchdog issues
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG; // Adjust format as needed

  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  // Adjust frame size for balance between speed and detail
  config.frame_size = FRAMESIZE_QVGA; // Example: QVGA (320x240)
  config.jpeg_quality = 10;          // 0-63 lower means better quality, higher means lower size

  // Adjust buffer count (higher might improve stability but use more memory)
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(1000);
    ESP.restart();
  }

  Serial.println("Camera initialized");
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("WebSocket Disconnected!");
      break;
    case WStype_CONNECTED: {
      Serial.printf("WebSocket Connected to: %s:%d%s\n", wsServer, wsPort, wsPath);
      // Optionally send a message upon connection
      // webSocket.sendTXT("ESP32-CAM Client Connected");
    }
    break;
    case WStype_TEXT:
      Serial.printf("WebSocket Text Received: %s\n", (const char*)payload);
      processRecognitionResponse((const char*)payload);
      break;
    case WStype_BIN:
      Serial.printf("WebSocket Binary Received: %u bytes\n", length);
      // Handle binary data if your server sends any
      break;
    case WStype_ERROR:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
      break;
  }
}

void sendFrameWebSocket() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera frame capture failed");
    return;
  }

  Serial.printf("Captured frame: %u bytes\n", fb->len);

  // Send the JPEG frame as binary data over WebSocket
  webSocket.sendBIN(fb->buf, fb->len);

  esp_camera_fb_return(fb);
}

void processRecognitionResponse(const String& response) {
  // Assuming the server sends a JSON response like {"face_recognized": true/false}
  StaticJsonDocument doc;
  DeserializationError error = deserializeJson(doc, response);

  if (error) {
    Serial.print("JSON Deserialization failed: ");
    Serial.println(error.c_str());
    return;
  }

  if (doc.containsKey("face_recognized")) {
    faceRecognized = doc["face_recognized"].as<bool>();
    Serial.printf("Face Recognized: %s\n", faceRecognized ? "Yes" : "No");
  } else {
    Serial.println("Invalid JSON response format");
  }
}

void controlLock(bool open) {
  if (open) {
    Serial.println("Opening solenoid lock");
    digitalWrite(relayPin, HIGH); // Assuming HIGH activates the solenoid to open
  } else {
    Serial.println("Closing solenoid lock");
    digitalWrite(relayPin, LOW);  // Assuming LOW deactivates the solenoid to close
  }
}