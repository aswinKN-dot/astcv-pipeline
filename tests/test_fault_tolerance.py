"""
ASTCV — M3 Fault-Tolerance & Stress Test Suite
==============================================
Specifically tests edge cases and error handling for M3's subsystem:
1. Corrupted / 0-byte video handling
2. Extreme lighting / completely black frames face crop resilience
3. Severe network jitter & massive frame drop telemetry
4. Malformed schema and data contract boundary enforcement
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.data_contract import (
    ScreenChallengePattern, FrameCaptureTelemetry, MatchedSyncPair,
    SessionManifest, DeviceMetadata
)
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.sanity_checker import SanityChecker
from src.integration.m2_session_receiver import M2SessionReceiver


def test_corrupted_video_handling(tmp_path):
    """FAULT TEST 1: Video file is empty or corrupted binary junk."""
    extractor = FrameExtractor()
    corrupt_file = tmp_path / "corrupted_video.mp4"
    corrupt_file.write_bytes(b"RANDOM_CORRUPT_BYTES_NOT_A_VALID_H264_STREAM")

    # Should safely return empty generator or handle cleanly without unhandled crash
    frames = list(extractor.extract_from_video(str(corrupt_file)))
    assert len(frames) == 0, "Corrupted video should yield 0 frames without throwing unhandled exception"


def test_extreme_lighting_pitch_black_frame():
    """FAULT TEST 2: Camera feed is completely pitch black (lux=0, no visible face)."""
    detector = FaceDetector(target_size=(256, 256), margin=1.3)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Must not crash; must fall back to safe centered crop of target dimensions
    res = detector.detect_and_crop(black_frame)
    assert res is not None
    assert res.cropped_face.shape == (256, 256, 3)
    assert res.detector_type == "center_crop_fallback"


def test_extreme_noise_frame():
    """FAULT TEST 3: Camera feed is pure static / TV noise."""
    detector = FaceDetector(target_size=(256, 256), margin=1.3)
    noise_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    res = detector.detect_and_crop(noise_frame)
    assert res is not None
    assert res.cropped_face.shape == (256, 256, 3)


def test_severe_jitter_and_packet_loss():
    """FAULT TEST 4: M2 WebRTC session experiences severe lag (>150ms) and dropped frames."""
    receiver = M2SessionReceiver(max_acceptable_jitter_ms=50.0)

    # Simulate severe lag: 180ms latency, high variance
    manifest = receiver.simulate_mock_capture_session(
        device_id="test_stressed_device",
        device_type="smartphone",
        model_name="Stressed Mobile",
        os_version="Android 14",
        browser="Chrome",
        base_latency_ms=180.0,
        jitter_std_ms=45.0,
        simulate_drops=True
    )

    stats = receiver._calculate_jitter_statistics(manifest.matched_pairs)
    assert stats.meets_jitter_tolerance is False, "Severe jitter must fail tolerance check"
    assert stats.p95_latency_delta_ms > 50.0
    assert stats.dropped_frames > 0


def test_data_contract_boundary_enforcement():
    """FAULT TEST 5: M1/M2 send illegal values outside physical safety bounds."""
    # Negative frequency or frequency > 30Hz (epilepsy safety limit)
    with pytest.raises(Exception):
        ScreenChallengePattern(
            pattern_id="unsafe_freq",
            sequence_index=1,
            timestamp_emitted_ms=100.0,
            color_rgb=(255, 255, 255),
            frequency_hz=60.0,  # ILLEGAL: exceeds 30.0 Hz photosafety bound
            nonce_token="nonce"
        )

    # Invalid RGB value (> 255)
    with pytest.raises(Exception):
        ScreenChallengePattern(
            pattern_id="invalid_color",
            sequence_index=2,
            timestamp_emitted_ms=100.0,
            color_rgb=(300, 100, 50),  # ILLEGAL: RGB channel > 255
            frequency_hz=10.0,
            nonce_token="nonce"
        )
