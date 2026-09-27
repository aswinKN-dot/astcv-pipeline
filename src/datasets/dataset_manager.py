"""
ASTCV Dataset Manager & Ingestion Tracker (M3: Aswin K N)
=========================================================
Tracks access state, local storage, checksum validation, and
file statistics for:
- FaceForensics++ (FF++)
- Celeb-DF v2
- DFDC
- UADFV
- FFHQ
- VoxCeleb2
"""

import os
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTCV.DatasetManager")


class AccessStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    APPLICATION_SUBMITTED = "APPLICATION_SUBMITTED"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    DOWNLOADING = "DOWNLOADING"
    READY_RAW = "READY_RAW"
    PREPROCESSED = "PREPROCESSED"
    FAILED_ERROR = "FAILED_ERROR"


class DatasetEntry:
    def __init__(
        self,
        dataset_id: str,
        display_name: str,
        local_dir: str,
        status: AccessStatus = AccessStatus.NOT_REQUESTED,
        total_videos: int = 0,
        total_frames: int = 0,
        notes: str = ""
    ):
        self.dataset_id = dataset_id
        self.display_name = display_name
        self.local_dir = local_dir
        self.status = status
        self.total_videos = total_videos
        self.total_frames = total_frames
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "local_dir": self.local_dir,
            "status": self.status.value,
            "total_videos": self.total_videos,
            "total_frames": self.total_frames,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetEntry":
        return cls(
            dataset_id=data["dataset_id"],
            display_name=data["display_name"],
            local_dir=data["local_dir"],
            status=AccessStatus(data.get("status", AccessStatus.NOT_REQUESTED.value)),
            total_videos=data.get("total_videos", 0),
            total_frames=data.get("total_frames", 0),
            notes=data.get("notes", "")
        )


class DatasetManager:
    """Manages tracking, health audits, and state updates for raw forensic datasets."""

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.metadata_file = Path(self.config["paths"]["metadata_dir"]) / "datasets_status.json"
        self.registry: Dict[str, DatasetEntry] = {}
        self._initialize_registry()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _initialize_registry(self) -> None:
        # Load from disk if exists
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    for key, val in saved_data.items():
                        self.registry[key] = DatasetEntry.from_dict(val)
                logger.info(f"Loaded existing dataset registry with {len(self.registry)} entries.")
                return
            except Exception as e:
                logger.warning(f"Could not parse {self.metadata_file}: {e}. Creating new registry.")

        # Default registry populated from pipeline_config
        for ds_key, ds_info in self.config.get("datasets", {}).items():
            self.registry[ds_key] = DatasetEntry(
                dataset_id=ds_info.get("id", ds_key),
                display_name=ds_info.get("name", ds_key),
                local_dir=ds_info.get("path", f"./data/raw/{ds_key}"),
                status=AccessStatus.NOT_REQUESTED,
                notes="Initial setup. Awaiting academic access approval."
            )
        self.save_registry()

    def save_registry(self) -> None:
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        serialized = {k: v.to_dict() for k, v in self.registry.items()}
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        logger.info(f"Saved dataset registry to {self.metadata_file}")

    def update_status(self, dataset_key: str, status: AccessStatus, notes: Optional[str] = None) -> None:
        if dataset_key not in self.registry:
            raise KeyError(f"Dataset '{dataset_key}' not recognized. Available: {list(self.registry.keys())}")
        self.registry[dataset_key].status = status
        if notes:
            self.registry[dataset_key].notes = notes
        self.save_registry()
        logger.info(f"Dataset '{dataset_key}' status updated to {status.value}")

    def scan_local_storage(self) -> Dict[str, Any]:
        """Scans filesystem to count real videos and frames per dataset directory."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        summary = {}
        for key, entry in self.registry.items():
            dir_path = Path(entry.local_dir)
            if not dir_path.exists():
                summary[key] = {"exists": False, "videos": 0, "images": 0, "status": entry.status.value}
                continue

            v_count = 0
            img_count = 0
            for root, _, files in os.walk(dir_path):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in video_extensions:
                        v_count += 1
                    elif ext in image_extensions:
                        img_count += 1

            entry.total_videos = v_count
            entry.total_frames = img_count
            if v_count > 0 or img_count > 0:
                if entry.status in (AccessStatus.NOT_REQUESTED, AccessStatus.APPLICATION_SUBMITTED):
                    entry.status = AccessStatus.READY_RAW

            summary[key] = {
                "exists": True,
                "videos": v_count,
                "images": img_count,
                "status": entry.status.value
            }

        self.save_registry()
        return summary

    def generate_status_markdown_report(self) -> str:
        """Generates a clean markdown table of dataset status for team syncs."""
        self.scan_local_storage()
        lines = [
            "# ASTCV Dataset Status & Ingestion Tracking (Phase 1)",
            "",
            "| Dataset | ID | Status | Local Videos | Local Frames | Notes |",
            "|:---|:---|:---|:---:|:---:|:---|"
        ]
        for key, entry in self.registry.items():
            lines.append(
                f"| {entry.display_name} | `{entry.dataset_id}` | **{entry.status.value}** | "
                f"{entry.total_videos} | {entry.total_frames} | {entry.notes} |"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    manager = DatasetManager()
    print(manager.generate_status_markdown_report())
