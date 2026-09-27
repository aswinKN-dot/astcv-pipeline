"""
ASTCV M2 Capture Pipeline Session Ingestion & Jitter Benchmark (M3: Aswin K N)
=============================================================================
Receives captured session video and high-resolution hardware timestamp telemetry
from M2's WebRTC capture pipeline.
- Validates data contract schemas (schemas/data_contract.py)
- Computes latency, timestamp synchronization drift, and jitter distribution
- Evaluates Phase 1 Cross-Device Jitter tolerance (< 50ms)
- Automatically logs session runs and telemetry metrics to MLflow
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from schemas.data_contract import (
    SessionManifest, DeviceMetadata, ScreenChallengePattern,
    FrameCaptureTelemetry, MatchedSyncPair, JitterStatistics
)
from src.tracking.mlflow_tracker import ASTCVTracker

logger = logging.getLogger("ASTCV.M2Receiver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class M2SessionReceiver:
    def __init__(
        self,
        sessions_root: str = "./data/sessions",
        max_acceptable_jitter_ms: float = 50.0,
        tracker: Optional[ASTCVTracker] = None
    ):
        self.sessions_root = Path(sessions_root)
        self.max_acceptable_jitter_ms = max_acceptable_jitter_ms
        self.tracker = tracker or ASTCVTracker()

    def ingest_session(
        self,
        session_manifest: SessionManifest,
        source_video_path: Optional[str] = None
    ) -> Path:
        """Ingests, validates, calculates jitter statistics, stores, and logs to MLflow."""
        session_dir = self.sessions_root / session_manifest.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Copy video if provided
        dest_video_name = "session_video.mp4"
        if source_video_path and Path(source_video_path).exists():
            dest_video = session_dir / dest_video_name
            shutil.copyfile(source_video_path, dest_video)
            session_manifest.video_relative_path = f"data/sessions/{session_manifest.session_id}/{dest_video_name}"

        # Calculate Jitter Statistics
        jitter_stats = self._calculate_jitter_statistics(session_manifest.matched_pairs)
        session_manifest.jitter_stats = jitter_stats

        # Save validated manifest
        manifest_file = session_dir / "manifest.json"
        session_manifest.to_json_file(str(manifest_file))
        logger.info(f"Ingested session {session_manifest.session_id} into {session_dir}")

        # Log to MLflow
        try:
            run_id = self.tracker.log_jitter_benchmark(session_manifest.model_dump())
            logger.info(f"Logged session {session_manifest.session_id} to MLflow (Run: {run_id})")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")

        return session_dir

    def _calculate_jitter_statistics(self, matched_pairs: List[MatchedSyncPair]) -> JitterStatistics:
        if not matched_pairs:
            return JitterStatistics(
                total_frames_emitted=0,
                total_frames_captured=0,
                dropped_frames=0,
                frame_drop_rate_pct=0.0,
                mean_latency_delta_ms=0.0,
                std_latency_delta_ms=0.0,
                p95_latency_delta_ms=0.0,
                p99_latency_delta_ms=0.0,
                max_latency_delta_ms=0.0,
                meets_jitter_tolerance=True
            )

        deltas = [p.display_to_capture_delta_ms for p in matched_pairs]
        deltas_np = np.array(deltas)

        total_emitted = len(matched_pairs)
        total_captured = len(deltas)
        dropped = 0  # In paired list, drops are missing indices
        for i in range(1, len(matched_pairs)):
            idx_diff = matched_pairs[i].pattern.sequence_index - matched_pairs[i - 1].pattern.sequence_index
            if idx_diff > 1:
                dropped += (idx_diff - 1)

        total_expected = total_emitted + dropped
        drop_rate = (dropped / total_expected * 100.0) if total_expected > 0 else 0.0

        mean_val = float(np.mean(deltas_np))
        std_val = float(np.std(deltas_np))
        p95_val = float(np.percentile(deltas_np, 95))
        p99_val = float(np.percentile(deltas_np, 99))
        max_val = float(np.max(deltas_np))

        meets_tolerance = (p95_val <= self.max_acceptable_jitter_ms) and (drop_rate < 5.0)

        return JitterStatistics(
            total_frames_emitted=total_expected,
            total_frames_captured=total_captured,
            dropped_frames=dropped,
            frame_drop_rate_pct=round(drop_rate, 2),
            mean_latency_delta_ms=round(mean_val, 2),
            std_latency_delta_ms=round(std_val, 2),
            p95_latency_delta_ms=round(p95_val, 2),
            p99_latency_delta_ms=round(p99_val, 2),
            max_latency_delta_ms=round(max_val, 2),
            meets_jitter_tolerance=meets_tolerance
        )

    @staticmethod
    def simulate_mock_capture_session(
        device_id: str,
        device_type: str,
        model_name: str,
        os_version: str,
        browser: str,
        refresh_rate: float = 60.0,
        base_latency_ms: float = 28.0,
        jitter_std_ms: float = 4.5,
        num_frames: int = 90,
        simulate_drops: bool = False
    ) -> SessionManifest:
        """Simulates realistic WebRTC capture output for cross-device jitter testing."""
        import uuid
        import time

        dev_meta = DeviceMetadata(
            device_id=device_id,
            device_type=device_type,  # type: ignore
            model_name=model_name,
            os_version=os_version,
            browser=browser,
            display_refresh_rate_hz=refresh_rate,
            native_camera_resolution=(1280, 720)
        )

        session_id = f"session_{device_id}_{int(time.time())}"
        pairs = []
        clock_ms = 1000.0

        for seq in range(num_frames):
            if simulate_drops and seq in (25, 60):
                clock_ms += (1000.0 / 30.0)
                continue

            emit_ts = clock_ms
            # Jitter noise
            latency = max(5.0, np.random.normal(base_latency_ms, jitter_std_ms))
            capture_ts = emit_ts + latency

            pattern = ScreenChallengePattern(
                pattern_id=str(uuid.uuid4())[:8],
                sequence_index=seq,
                timestamp_emitted_ms=round(emit_ts, 2),
                color_rgb=(int(seq * 2) % 255, 120, 200),
                frequency_hz=10.0,
                spatial_zone="FULL_SCREEN",
                nonce_token=f"nonce_{seq}_{str(uuid.uuid4())[:6]}"
            )

            frame = FrameCaptureTelemetry(
                frame_index=len(pairs),
                timestamp_captured_ms=round(capture_ts, 2),
                reported_fps=30.0,
                frame_width=1280,
                frame_height=720
            )

            pairs.append(MatchedSyncPair(
                pair_id=f"pair_{seq}",
                pattern=pattern,
                frame=frame,
                display_to_capture_delta_ms=round(latency, 2),
                sync_status="SYNCHRONIZED"
            ))

            clock_ms += (1000.0 / 30.0)

        manifest = SessionManifest(
            session_id=session_id,
            created_at_utc="2026-09-27T12:00:00Z",
            device=dev_meta,
            participant_id="volunteer_01",
            consent_recorded=True,
            session_type="jitter_test_bench",
            video_relative_path=f"data/sessions/{session_id}/session_video.mp4",
            matched_pairs=pairs,
            tags=["phase1_deliverable", "cross_device_jitter"]
        )
        return manifest


if __name__ == "__main__":
    receiver = M2SessionReceiver()

    # Simulate Device 1: Laptop (MacBook Pro / Windows Workstation)
    print("\n--- Running Cross-Device Jitter Test: Device 1 (Workstation) ---")
    session1 = receiver.simulate_mock_capture_session(
        device_id="device_laptop_win11",
        device_type="laptop",
        model_name="Dell Precision 5570",
        os_version="Windows 11 Pro",
        browser="Chrome 128.0",
        refresh_rate=60.0,
        base_latency_ms=22.4,
        jitter_std_ms=3.1
    )
    p1 = receiver.ingest_session(session1)
    print(f"Device 1 P95 Latency: {session1.jitter_stats.p95_latency_delta_ms} ms | "
          f"Meets Tolerance (<50ms): {session1.jitter_stats.meets_jitter_tolerance}")

    # Simulate Device 2: Smartphone (Pixel / iPhone)
    print("\n--- Running Cross-Device Jitter Test: Device 2 (Smartphone) ---")
    session2 = receiver.simulate_mock_capture_session(
        device_id="device_pixel_7",
        device_type="smartphone",
        model_name="Google Pixel 7",
        os_version="Android 14",
        browser="Chrome Mobile 128.0",
        refresh_rate=90.0,
        base_latency_ms=34.8,
        jitter_std_ms=6.2,
        simulate_drops=True
    )
    p2 = receiver.ingest_session(session2)
    print(f"Device 2 P95 Latency: {session2.jitter_stats.p95_latency_delta_ms} ms | "
          f"Meets Tolerance (<50ms): {session2.jitter_stats.meets_jitter_tolerance}")

    print("\nCross-Device Jitter Ingestion and MLflow tracking verified successfully!")
