"""
ASTCV Unit & Integration Tests (M3: Aswin K N)
==============================================
Validates:
1. Data Contract Schema & Validation
2. Frame Extraction & Resilient Face Cropping
3. First-Pass Dataset Sanity Checker
4. M2 Jitter Telemetry & Latency Calculation
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.data_contract import (
    ScreenChallengePattern, FrameCaptureTelemetry, MatchedSyncPair,
    DeviceMetadata, SessionManifest, JitterStatistics
)
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.sanity_checker import SanityChecker
from src.integration.m2_session_receiver import M2SessionReceiver


def test_screen_challenge_pattern_validation():
    pattern = ScreenChallengePattern(
        pattern_id="test_pat_01",
        sequence_index=0,
        timestamp_emitted_ms=1050.2,
        color_rgb=(255, 0, 128),
        frequency_hz=10.0,
        spatial_zone="FULL_SCREEN",
        nonce_token="test_nonce_123"
    )
    assert pattern.frequency_hz == 10.0
    assert pattern.color_rgb == (255, 0, 128)

    # Invalid RGB should fail
    with pytest.raises(Exception):
        ScreenChallengePattern(
            pattern_id="invalid",
            sequence_index=1,
            timestamp_emitted_ms=1100.0,
            color_rgb=(300, 0, 0),  # > 255
            frequency_hz=5.0,
            nonce_token="x"
        )


def test_face_detector_fallback():
    detector = FaceDetector(target_size=(256, 256), margin=1.3)
    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    result = detector.detect_and_crop(dummy_frame)
    assert result is not None
    assert result.cropped_face.shape == (256, 256, 3)
    assert result.bbox is not None


def test_jitter_statistics_calculation():
    receiver = M2SessionReceiver()
    pattern = ScreenChallengePattern(
        pattern_id="p1", sequence_index=0, timestamp_emitted_ms=100.0,
        color_rgb=(100, 100, 100), frequency_hz=10.0, nonce_token="n1"
    )
    frame = FrameCaptureTelemetry(
        frame_index=0, timestamp_captured_ms=125.0, reported_fps=30.0,
        frame_width=640, frame_height=480
    )
    pair1 = MatchedSyncPair(pair_id="pair1", pattern=pattern, frame=frame, display_to_capture_delta_ms=25.0)

    pattern2 = ScreenChallengePattern(
        pattern_id="p2", sequence_index=1, timestamp_emitted_ms=133.3,
        color_rgb=(100, 100, 100), frequency_hz=10.0, nonce_token="n2"
    )
    frame2 = FrameCaptureTelemetry(
        frame_index=1, timestamp_captured_ms=161.3, reported_fps=30.0,
        frame_width=640, frame_height=480
    )
    pair2 = MatchedSyncPair(pair_id="pair2", pattern=pattern2, frame=frame2, display_to_capture_delta_ms=28.0)

    stats = receiver._calculate_jitter_statistics([pair1, pair2])
    assert stats.total_frames_captured == 2
    assert stats.dropped_frames == 0
    assert stats.mean_latency_delta_ms == 26.5
    assert stats.meets_jitter_tolerance is True


def test_sanity_checker():
    checker = SanityChecker(processed_root="./data/processed")
    results = checker.audit_all()
    assert isinstance(results, dict)
    # Check that processed datasets are recognized
    for ds_name, stats in results.items():
        assert "total_records" in stats
        assert stats["total_records"] > 0
        assert stats["corrupted_images"] == 0
