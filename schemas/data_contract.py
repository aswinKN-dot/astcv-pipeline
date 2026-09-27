"""
ASTCV Data Contract Definition (Joint M1 - M2 - M3 Specification)
===================================================================
Defines the strict data schema and telemetry interchange format between:
- Screen Emitter & Challenge Generator (M1: Amal)
- WebRTC Capture Pipeline & Sync Engine (M2: Abhijith)
- Data Storage, Versioning & Metric Logging (M3: Aswin)
"""

from typing import List, Dict, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator
import json
import time


class SpatialZone(str):
    FULL = "FULL_SCREEN"
    FOREHEAD = "FOREHEAD"
    LEFT_CHEEK = "LEFT_CHEEK"
    RIGHT_CHEEK = "RIGHT_CHEEK"
    NOSE_BRIDGE = "NOSE_BRIDGE"
    CHIN = "CHIN"
    CHECKERBOARD_4X4 = "CHECKERBOARD_4X4"


class ScreenChallengePattern(BaseModel):
    """Data emitted by M1's challenge engine and rendered on screen by M2's UI."""
    pattern_id: str = Field(..., description="Unique UUID for this specific challenge block")
    sequence_index: int = Field(..., ge=0, description="Sequential index within the session")
    timestamp_emitted_ms: float = Field(..., description="High-resolution display timestamp (performance.now)")
    color_rgb: Tuple[int, int, int] = Field(..., description="RGB color tuple rendered on screen [0-255]")
    frequency_hz: float = Field(..., ge=0.5, le=30.0, description="Modulation frequency in Hz (photosafety bounded)")
    spatial_zone: str = Field(default=SpatialZone.FULL, description="Targeted facial or screen spatial zone")
    nonce_token: str = Field(..., description="Cryptographic salt/nonce preventing pattern replay/prediction")

    @field_validator("color_rgb")
    def validate_rgb(cls, v):
        if len(v) != 3 or any(c < 0 or c > 255 for c in v):
            raise ValueError("color_rgb must be 3 integers between 0 and 255")
        return v


class FrameCaptureTelemetry(BaseModel):
    """Metadata recorded by M2's WebRTC capture pipeline for each received camera frame."""
    frame_index: int = Field(..., ge=0, description="Sequential captured frame index")
    timestamp_captured_ms: float = Field(..., description="Camera sensor / video frame timestamp (performance.now)")
    exposure_time_ms: Optional[float] = Field(None, description="Camera exposure duration if exposed by browser API")
    reported_fps: float = Field(..., gt=0, description="Instantaneous camera capture FPS")
    frame_width: int = Field(..., gt=0)
    frame_height: int = Field(..., gt=0)
    frame_checksum_sha256: Optional[str] = Field(None, description="SHA256 of raw frame bytes")


class MatchedSyncPair(BaseModel):
    """M2's matched emitter pattern and corresponding camera frame capture."""
    pair_id: str = Field(..., description="Sync match identifier")
    pattern: ScreenChallengePattern
    frame: FrameCaptureTelemetry
    display_to_capture_delta_ms: float = Field(..., description="Delta = timestamp_captured - timestamp_emitted")
    sync_status: Literal["SYNCHRONIZED", "JITTER_WARNING", "DESYNC_DROPPED"] = "SYNCHRONIZED"
    estimated_ambient_lux: Optional[float] = None


class DeviceMetadata(BaseModel):
    """Hardware and environment profile for cross-device calibration validation."""
    device_id: str = Field(..., description="Hardware identifier / test device tag")
    device_type: Literal["laptop", "smartphone", "tablet", "external_webcam", "virtual_cam_test"]
    model_name: str = Field(..., description="e.g. MacBook Pro M2, Dell XPS 15, Pixel 7, iPhone 14")
    os_version: str = Field(..., description="Operating System & version")
    browser: str = Field(..., description="Browser name & version (Chrome, Safari, Firefox)")
    display_refresh_rate_hz: float = Field(..., gt=0, description="Display refresh rate (e.g., 60Hz, 120Hz)")
    camera_sensor_name: Optional[str] = None
    native_camera_resolution: Tuple[int, int] = Field(..., description="Native (width, height)")


class JitterStatistics(BaseModel):
    """Telemetry metrics calculated for Week 3 cross-device jitter deliverable."""
    total_frames_emitted: int = Field(..., ge=0)
    total_frames_captured: int = Field(..., ge=0)
    dropped_frames: int = Field(..., ge=0)
    frame_drop_rate_pct: float = Field(..., ge=0.0, le=100.0)
    mean_latency_delta_ms: float
    std_latency_delta_ms: float
    p95_latency_delta_ms: float
    p99_latency_delta_ms: float
    max_latency_delta_ms: float
    meets_jitter_tolerance: bool = Field(..., description="True if jitter remains under threshold (e.g., < 50ms)")


class SessionManifest(BaseModel):
    """
    Master session manifest stored by M3 for each live captured ASTCV session.
    Stored at: data/sessions/{session_id}/manifest.json
    """
    schema_version: str = Field(default="1.0.0")
    session_id: str = Field(..., description="UUID or unique session identifier")
    created_at_utc: str = Field(..., description="ISO 8601 UTC timestamp")
    device: DeviceMetadata
    participant_id: Optional[str] = Field(None, description="Anonymized volunteer ID for consent tracking")
    consent_recorded: bool = Field(default=False, description="Compliance check: formal consent documented")
    session_type: Literal["genuine_human", "printed_photo_attack", "replay_attack", "virtual_cam_injection", "jitter_test_bench"]
    video_relative_path: str = Field(..., description="Path to recorded session video (MP4/WebM)")
    matched_pairs: List[MatchedSyncPair] = Field(default_factory=list)
    jitter_stats: Optional[JitterStatistics] = None
    calibration_matrix_path: Optional[str] = None
    physicality_score: Optional[float] = None
    tags: List[str] = Field(default_factory=list)

    def to_json_file(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, filepath: str) -> "SessionManifest":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


if __name__ == "__main__":
    # Self-test schema generation
    schema_dict = SessionManifest.model_json_schema()
    with open("schemas/data_contract.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)
    print("ASTCV Data Contract validated and exported to schemas/data_contract.json")
