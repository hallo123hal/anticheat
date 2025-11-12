from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, Set
import asyncio
import uuid
import queue
from datetime import datetime
import json
import time

import cv2

from app.models.schemas import (
    StartMonitoringRequest,
    StopMonitoringRequest,
    MonitoringResponse,
    ViolationResponse,
    ViolationType,
    ViolationEvent,
)
from app.monitoring.camera_monitor import CameraMonitor


app = FastAPI(title="Anti-Cheat Service", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active monitoring sessions
monitoring_sessions: Dict[str, CameraMonitor] = {}
websocket_connections: Dict[str, Set[WebSocket]] = {}
session_data: Dict[str, dict] = {}  # Store exam_id, student_id for each session
violation_queues: Dict[str, queue.Queue] = {}  # Store violation queues for each session
violations_history: Dict[str, list] = {}  # Store violation history for each session
def detect_available_cameras(max_index: int = 5) -> Dict[int, bool]:
    """Probe camera indices and return availability map."""
    availability: Dict[int, bool] = {}
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        available = cap.isOpened()
        if available:
            cap.release()
        availability[idx] = available
    return availability

def choose_camera_index(preferred: int | None) -> int | None:
    """Return a camera index to use, preferring the provided index else first available."""
    if preferred is not None:
        cap = cv2.VideoCapture(preferred)
        if cap.isOpened():
            cap.release()
            return preferred
        return None
    availability = detect_available_cameras()
    for idx, ok in availability.items():
        if ok:
            return idx
    return None



async def process_violation_queue(session_id: str, violation_queue: queue.Queue):
    """Process violation queue and send notifications via WebSocket."""
    print(f"[DEBUG] Started violation queue processor for session: {session_id}")
    while session_id in monitoring_sessions:
        try:
            # Wait for violation (with timeout to check if session still exists)
            try:
                violation_type, message = violation_queue.get(timeout=1.0)
                print(f"[DEBUG] Violation detected in queue: {violation_type} - {message}")
            except queue.Empty:
                continue
            
            # Create violation event
            violation_event = ViolationEvent(
                type="violation",
                violation_type=ViolationType(violation_type),
                timestamp=datetime.now(),
                message=message,
                session_id=session_id,
            )
            
            # Store violation in history
            if session_id not in violations_history:
                violations_history[session_id] = []
            violations_history[session_id].append(violation_event.model_dump(mode="json"))
            print(f"[DEBUG] Violation stored in history. Total violations for session {session_id}: {len(violations_history[session_id])}")
            
            # Send to all WebSocket connections for this session
            if session_id in websocket_connections and websocket_connections[session_id]:
                event_payload = violation_event.model_dump(mode="json")
                print(f"[DEBUG] Sending violation to {len(websocket_connections[session_id])} WebSocket connection(s)")
                disconnected = set()
                for websocket in websocket_connections[session_id]:
                    try:
                        await websocket.send_json(event_payload)
                        print(f"[DEBUG] Violation sent successfully via WebSocket")
                    except Exception as e:
                        print(f"[ERROR] Error sending violation to WebSocket: {e}")
                        disconnected.add(websocket)
                
                # Remove disconnected connections
                websocket_connections[session_id] -= disconnected
            else:
                print(f"[WARNING] No WebSocket connections for session {session_id}. Violation not sent: {violation_type} - {message}")
                
        except Exception as e:
            print(f"[ERROR] Error processing violation queue: {e}")
            break


@app.post("/start-monitoring", response_model=MonitoringResponse)
async def start_monitoring(request: StartMonitoringRequest):
    """Start monitoring session."""
    session_id = request.session_id or str(uuid.uuid4())
    
    # Check if session already exists
    if session_id in monitoring_sessions:
        return MonitoringResponse(
            status="already_started",
            session_id=session_id,
            message="Monitoring session already started",
        )
    
    try:
        # Create violation queue
        violation_queue = queue.Queue(maxsize=10)
        violation_queues[session_id] = violation_queue
        
        # Create camera monitor with violation queue
        # Note: Camera access requires the service to run on the user's machine
        # For web applications, camera is accessed via browser, so monitoring
        # may need to be done client-side or via video streaming
        selected_index = choose_camera_index(request.camera_index)
        if selected_index is None:
            return MonitoringResponse(
                status="camera_unavailable",
                session_id=session_id,
                message="No available camera device found",
            )
        monitor = CameraMonitor(camera_index=selected_index, violation_queue=violation_queue)
        
        # Start monitoring (this will try to access camera)
        # If camera access fails, monitoring will continue but camera-based detection won't work
        monitor.start_monitoring()
        if not monitor.is_camera_open():
            return MonitoringResponse(
                status="camera_unavailable",
                session_id=session_id,
                message=f"Could not open camera at index {selected_index}",
            )
        
        # Store session (even if camera access failed)
        monitoring_sessions[session_id] = monitor
        session_data[session_id] = {
            "exam_id": request.exam_id,
            "student_id": request.student_id,
            "started_at": datetime.now().isoformat(),
            "camera_index": selected_index,
        }
        
        # Initialize WebSocket connections set
        if session_id not in websocket_connections:
            websocket_connections[session_id] = set()
        
        # Start violation queue processing task
        asyncio.create_task(process_violation_queue(session_id, violation_queue))
        
        return MonitoringResponse(
            status="started",
            session_id=session_id,
            message="Monitoring started successfully",
        )
    except Exception as e:
        print(f"Error starting monitoring: {e}")
        # Allow service to continue even if camera access fails
        # WebSocket connections will still work
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@app.post("/stop-monitoring", response_model=MonitoringResponse)
async def stop_monitoring(request: StopMonitoringRequest):
    """Stop monitoring session."""
    session_id = request.session_id
    
    if session_id not in monitoring_sessions:
        return MonitoringResponse(
            status="not_found",
            session_id=session_id,
            message="Monitoring session not found",
        )
    
    try:
        # Stop and release monitor
        monitor = monitoring_sessions[session_id]
        monitor.stop_monitoring()
        monitor.release()
        
        # Clean up
        del monitoring_sessions[session_id]
        if session_id in websocket_connections:
            # Close all WebSocket connections
            for websocket in websocket_connections[session_id]:
                try:
                    await websocket.close()
                except Exception:
                    pass
            del websocket_connections[session_id]
        if session_id in session_data:
            del session_data[session_id]
        if session_id in violation_queues:
            del violation_queues[session_id]
        if session_id in violations_history:
            del violations_history[session_id]
        
        return MonitoringResponse(
            status="stopped",
            session_id=session_id,
            message="Monitoring stopped successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@app.get("/check-violation/{session_id}", response_model=ViolationResponse)
async def check_violation(session_id: str):
    """Check violation status for a session."""
    if session_id not in monitoring_sessions:
        return ViolationResponse(
            has_violation=False,
            message="Session not found or monitoring not started",
        )
    
    # Check if there are any violations
    has_violations = session_id in violations_history and len(violations_history[session_id]) > 0
    
    return ViolationResponse(
        has_violation=has_violations,
        message=f"Monitoring active. {'Violations detected.' if has_violations else 'No violations yet.'}",
    )


@app.get("/violations/{session_id}")
async def get_violations(session_id: str):
    """Get all violations for a session."""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found or monitoring not started")
    
    violations = violations_history.get(session_id, [])
    
    return {
        "session_id": session_id,
        "total_violations": len(violations),
        "violations": violations,
        "session_info": session_data.get(session_id, {})
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time violation notifications."""
    await websocket.accept()
    print(f"[DEBUG] WebSocket connection established for session: {session_id}")
    
    # Add to connections set
    if session_id not in websocket_connections:
        websocket_connections[session_id] = set()
    websocket_connections[session_id].add(websocket)
    print(f"[DEBUG] Total WebSocket connections for session {session_id}: {len(websocket_connections[session_id])}")
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to anti-cheat monitoring",
        })
        print(f"[DEBUG] Sent connection confirmation to WebSocket for session {session_id}")
        
        # Keep connection alive and wait for messages
        while True:
            try:
                # Wait for any message (ping/pong for keepalive)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back or process message
                await websocket.send_json({
                    "type": "pong",
                    "data": data,
                })
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat(),
                })
            except WebSocketDisconnect:
                print(f"[DEBUG] WebSocket disconnected for session: {session_id}")
                break
    except Exception as e:
        print(f"[ERROR] WebSocket error for session {session_id}: {e}")
    finally:
        # Remove from connections set
        if session_id in websocket_connections:
            websocket_connections[session_id].discard(websocket)
            print(f"[DEBUG] Removed WebSocket connection. Remaining for session {session_id}: {len(websocket_connections[session_id])}")
            if not websocket_connections[session_id]:
                del websocket_connections[session_id]
                print(f"[DEBUG] No more WebSocket connections for session {session_id}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(monitoring_sessions),
        "timestamp": datetime.now().isoformat(),
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    for session_id, monitor in list(monitoring_sessions.items()):
        try:
            monitor.stop_monitoring()
            monitor.release()
        except Exception:
            pass
    monitoring_sessions.clear()
    websocket_connections.clear()
    session_data.clear()


def frame_stream_generator(session_id: str):
    """Generate multipart JPEG stream for the specified session."""
    boundary = b"--frame"
    frame_count = 0
    print(f"[DEBUG] Starting camera stream for session: {session_id}")
    
    while True:
        if session_id not in monitoring_sessions:
            print(f"[DEBUG] Session {session_id} not found, stopping stream")
            break

        monitor = monitoring_sessions[session_id]
        if not monitor.is_monitoring:
            time.sleep(0.1)
            continue

        frame = monitor.get_latest_frame()
        if frame is None:
            if frame_count == 0:
                print(f"[DEBUG] No frame available yet for session {session_id}")
            time.sleep(0.05)
            continue

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print(f"[ERROR] Failed to encode frame for session {session_id}")
            time.sleep(0.05)
            continue

        frame_bytes = buffer.tobytes()
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames (~1 second at 30fps)
            print(f"[DEBUG] Streaming frame {frame_count} for session {session_id} ({len(frame_bytes)} bytes)")
        
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        time.sleep(0.033)  # ~30 FPS


@app.get("/stream/{session_id}")
async def stream_camera(session_id: str):
    """Stream live camera feed for a monitoring session."""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Monitoring session not found")

    return StreamingResponse(
        frame_stream_generator(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/cameras")
async def list_cameras():
    """List available camera indices on this machine."""
    availability = detect_available_cameras()
    available = [idx for idx, ok in availability.items() if ok]
    return {"available_indices": available, "probed": list(availability.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)

