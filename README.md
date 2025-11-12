# Anti-Cheat Service

Python-based anti-cheat service for monitoring students during online exams.

## Features

- **Camera Monitoring**: Real-time face detection and eye gaze tracking
- **Voice Detection**: Detects human speech using VAD (Voice Activity Detection)
- **Face Presence Detection**: Monitors if student leaves camera view
- **Eye Gaze Detection**: Detects when student looks away for 5+ seconds
- **WebSocket Notifications**: Real-time violation notifications to frontend
- **REST API**: Endpoints for starting/stopping monitoring and checking violations

## Requirements

- Python 3.8+
- Camera access
- Microphone access
- OpenCV
- MediaPipe
- FastAPI
- WebSocket support

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install system dependencies for PyAudio:
   - **Linux**:
     ```bash
     sudo apt-get update
     sudo apt-get install portaudio19-dev python3-pyaudio
     ```
   - **macOS**:
     ```bash
     brew install portaudio
     ```
   - **Windows**: No additional system packages required.

3. Upgrade pip and install Python dependencies:
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

> If you install new dependencies later, make sure the virtual environment is activated before running `pip`.

## Running the Service

With the virtual environment activated (`source .venv/bin/activate`):
```bash
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8081
```

The service will run on `http://localhost:8081`

### Live Camera Preview

1. Gửi `POST /start-monitoring` với `session_id` cụ thể để yêu cầu backend mở camera.
2. Trên FE, hiển thị video bằng cách trỏ một thẻ `<img>` hoặc phần tử `<video>` sử dụng nguồn `http://localhost:8081/stream/{session_id}` (đường dẫn trả về `multipart/x-mixed-replace`).
3. Khi hoàn thành, gọi `POST /stop-monitoring` để giải phóng camera.

## API Endpoints

### POST /start-monitoring
Start monitoring session.

Request:
```json
{
  "exam_id": 1,
  "student_id": 1,
  "session_id": "session_123",
  "camera_index": 0
}
```

Response:
```json
{
  "status": "started",
  "session_id": "session_123",
  "message": "Monitoring started successfully"
}
```

- Nếu không truyền `camera_index`, service sẽ tự dò camera khả dụng (0..5).  
- Nếu không tìm thấy camera, `status` sẽ là `"camera_unavailable"`.

### GET /cameras
Liệt kê các chỉ số camera khả dụng trên máy backend.

Response:
```json
{
  "available_indices": [0],
  "probed": [0,1,2,3,4,5]
}
```

### POST /stop-monitoring
Stop monitoring session.

Request:
```json
{
  "session_id": "session_123"
}
```

### GET /check-violation/{session_id}
Check violation status for a session.

### GET /health
Health check endpoint.

## WebSocket

### WebSocket Endpoint: /ws/{session_id}

Connect to WebSocket for real-time violation notifications.

Violation event format:
```json
{
  "type": "violation",
  "violation_type": "eye_gaze",
  "timestamp": "2024-01-01T00:00:00Z",
  "message": "Looking away detected for 5 seconds",
  "session_id": "session_123"
}
```

## Violation Types

- `eye_gaze`: Student looking away for 5+ seconds
- `voice`: Human speech detected
- `face_presence`: Student left camera view for 3+ seconds

## Configuration

The service can be configured by modifying the following constants in the code:

- Eye gaze threshold: `LOOK_AWAY_THRESHOLD` (degrees)
- Eye gaze duration: `LOOK_AWAY_DURATION` (seconds)
- Face absence threshold: `face_absence_threshold` (seconds)
- Voice detection sensitivity: VAD aggressiveness mode (0-3)

## Notes

- The service requires direct access to camera and microphone
- For web applications, camera/microphone access is handled by the browser
- The service should run on the same machine as the browser or be accessible via network
- Camera index can be configured (default: 0)

## Troubleshooting

1. **Camera not found**: Check if camera is connected and accessible
2. **Microphone not found**: Check if microphone is connected and permissions are granted
3. **WebSocket connection failed**: Check if service is running on port 8081
4. **Violations not detected**: Check camera/microphone permissions and lighting conditions

