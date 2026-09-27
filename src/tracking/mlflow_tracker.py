"""
ASTCV MLflow Experiment Tracker (M3: Aswin K N)
==============================================
Manages central experiment tracking for:
1. Dataset Ingestion Runs (sample counts, real/fake splits, quality)
2. Cross-Device Jitter Benchmarks (M2 WebRTC capture telemetry)
3. Physicality & Baseline Forensics Evaluation (FAR, FRR, AUC-ROC)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import mlflow

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger("ASTCV.MLflowTracker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ASTCVTracker:
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        # Set MLflow tracking URI & artifact path
        tracking_uri = self.config["mlflow"].get("tracking_uri", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI initialized: {tracking_uri}")

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def log_dataset_ingestion(self, audit_summary: Dict[str, Any], report_md_path: Optional[str] = None) -> str:
        """Logs dataset ingestion and first-pass sanity check to MLflow."""
        exp_name = self.config["mlflow"]["experiments"].get("dataset_ingestion", "ASTCV-Dataset-Ingestion")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name="ingestion_sanity_audit") as run:
            run_id = run.info.run_id

            # Log aggregate parameters
            mlflow.log_param("canonical_resolution", str(self.config["standardization"]["image_size"]))
            mlflow.log_param("color_space", self.config["standardization"]["color_space"])
            mlflow.log_param("target_fps", self.config["standardization"]["target_fps"])

            total_frames = 0
            total_real = 0
            total_fake = 0

            # Log per-dataset metrics
            for ds_name, stats in audit_summary.items():
                mlflow.log_metric(f"{ds_name}_total_frames", stats.get("total_records", 0))
                mlflow.log_metric(f"{ds_name}_real_frames", stats.get("real_count", 0))
                mlflow.log_metric(f"{ds_name}_fake_frames", stats.get("fake_count", 0))
                mlflow.log_metric(f"{ds_name}_avg_blur_score", stats.get("mean_laplacian_variance", 0.0))
                mlflow.log_metric(f"{ds_name}_corrupt_frames", stats.get("corrupted_images", 0))

                total_frames += stats.get("total_records", 0)
                total_real += stats.get("real_count", 0)
                total_fake += stats.get("fake_count", 0)

            mlflow.log_metric("total_standardized_frames", total_frames)
            mlflow.log_metric("total_real_frames", total_real)
            mlflow.log_metric("total_fake_frames", total_fake)

            if report_md_path and Path(report_md_path).exists():
                mlflow.log_artifact(report_md_path, artifact_path="sanity_reports")

            logger.info(f"Dataset ingestion run logged to MLflow [Run ID: {run_id}]")
            return run_id

    def log_jitter_benchmark(self, session_manifest_dict: Dict[str, Any]) -> str:
        """Logs M2's cross-device jitter capture session and telemetry."""
        exp_name = self.config["mlflow"]["experiments"].get("cross_device_jitter", "ASTCV-Cross-Device-Jitter")
        mlflow.set_experiment(exp_name)

        device = session_manifest_dict.get("device", {})
        stats = session_manifest_dict.get("jitter_stats", {})
        session_id = session_manifest_dict.get("session_id", "unknown_session")

        run_name = f"jitter_{device.get('device_type', 'dev')}_{device.get('model_name', 'model')}".replace(" ", "_")

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Parameters
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("device_id", device.get("device_id"))
            mlflow.log_param("device_type", device.get("device_type"))
            mlflow.log_param("model_name", device.get("model_name"))
            mlflow.log_param("os_version", device.get("os_version"))
            mlflow.log_param("browser", device.get("browser"))
            mlflow.log_param("display_refresh_rate_hz", device.get("display_refresh_rate_hz"))
            mlflow.log_param("session_type", session_manifest_dict.get("session_type"))

            # Metrics
            if stats:
                mlflow.log_metric("mean_latency_ms", stats.get("mean_latency_delta_ms", 0.0))
                mlflow.log_metric("std_latency_ms", stats.get("std_latency_delta_ms", 0.0))
                mlflow.log_metric("p95_latency_ms", stats.get("p95_latency_delta_ms", 0.0))
                mlflow.log_metric("p99_latency_ms", stats.get("p99_latency_delta_ms", 0.0))
                mlflow.log_metric("max_latency_ms", stats.get("max_latency_delta_ms", 0.0))
                mlflow.log_metric("dropped_frames", stats.get("dropped_frames", 0))
                mlflow.log_metric("drop_rate_pct", stats.get("frame_drop_rate_pct", 0.0))
                mlflow.log_metric("meets_jitter_tolerance", 1.0 if stats.get("meets_jitter_tolerance") else 0.0)

            logger.info(f"Cross-device jitter run logged to MLflow [Run ID: {run_id}]")
            return run_id


if __name__ == "__main__":
    tracker = ASTCVTracker()
    print("ASTCV MLflow Tracker ready. Tracking URI:", tracker.config["mlflow"]["tracking_uri"])
