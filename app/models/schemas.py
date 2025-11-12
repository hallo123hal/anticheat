from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class ViolationType(str, Enum):
    EYE_GAZE = "eye_gaze"
    VOICE = "voice"
    FACE_PRESENCE = "face_presence"


class StartMonitoringRequest(BaseModel):
    exam_id: int
    student_id: int
    session_id: str
    camera_index: Optional[int] = None


class StopMonitoringRequest(BaseModel):
    session_id: str


class MonitoringResponse(BaseModel):
    status: str
    session_id: str
    message: Optional[str] = None


class ViolationResponse(BaseModel):
    has_violation: bool
    violation_type: Optional[ViolationType] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None


class ViolationEvent(BaseModel):
    type: str = "violation"
    violation_type: ViolationType
    timestamp: datetime
    message: str
    session_id: str

