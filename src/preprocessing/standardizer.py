"""
ASTCV Dataset Standardization Pipeline (M3: Aswin K N)
======================================================
Harmonizes multi-source forensic datasets into a uniform,
canonical format:
- Canonical Resolution: 256x256 (3-channel RGB)
- Standardized Labels: REAL (0) vs FAKE (1)
- Rich Manipulation Tags: FaceSwap, Deepfakes, Face2Face, NeuralTextures, etc.
- Unified JSONL Metadata Manifest
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2

from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector

logger = logging.getLogger("ASTCV.Standardizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DatasetStandardizer:
    def __init__(
        self,
        raw_root: str = "./data/raw",
        processed_root: str = "./data/processed",
        target_size: tuple = (256, 256),
        target_fps: float = 5.0,
        margin: float = 1.3
    ):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.target_size = target_size
        self.extractor = FrameExtractor(target_fps=target_fps)
        self.detector = FaceDetector(target_size=target_size, margin=margin)

    def process_all(self, max_frames_per_video: int = 10) -> Dict[str, int]:
        """Runs standardization across all available raw datasets."""
        results = {}
        for ds_dir in self.raw_root.iterdir():
            if ds_dir.is_dir() and not ds_dir.name.startswith("."):
                count = self.process_dataset(ds_dir.name, max_frames_per_video=max_frames_per_video)
                results[ds_dir.name] = count
        return results

    def process_dataset(self, dataset_name: str, max_frames_per_video: int = 10) -> int:
        ds_raw_path = self.raw_root / dataset_name
        ds_out_path = self.processed_root / dataset_name
        ds_out_path.mkdir(parents=True, exist_ok=True)
        manifest_path = ds_out_path / "manifest.jsonl"

        logger.info(f"Standardizing dataset: {dataset_name} -> {ds_out_path}")
        records = []
        total_processed_frames = 0

        # Special case: Image dataset (FFHQ)
        if dataset_name == "ffhq":
            records, total_processed_frames = self._process_ffhq(ds_raw_path, ds_out_path)
        else:
            # Video datasets
            video_files = list(ds_raw_path.rglob("*.mp4")) + list(ds_raw_path.rglob("*.avi"))
            for v_path in video_files:
                label_info = self._infer_label(dataset_name, v_path)
                video_stem = v_path.stem
                out_frames_dir = ds_out_path / label_info["label_text"].lower() / video_stem
                out_frames_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx, ts_ms, frame_bgr in self.extractor.extract_from_video(
                    str(v_path), max_frames=max_frames_per_video
                ):
                    crop_res = self.detector.detect_and_crop(frame_bgr)
                    if crop_res is None:
                        continue

                    img_filename = f"frame_{frame_idx:05d}.png"
                    img_out_file = out_frames_dir / img_filename
                    cv2.imwrite(str(img_out_file), crop_res.cropped_face)

                    record = {
                        "dataset": dataset_name,
                        "video_source": str(v_path.relative_to(self.raw_root)),
                        "relative_image_path": str(img_out_file.relative_to(self.processed_root)),
                        "frame_index": frame_idx,
                        "timestamp_ms": ts_ms,
                        "label_binary": label_info["label_binary"],
                        "label_text": label_info["label_text"],
                        "manipulation_type": label_info["manipulation_type"],
                        "detector_used": crop_res.detector_type,
                        "detector_confidence": crop_res.confidence,
                        "width": self.target_size[0],
                        "height": self.target_size[1]
                    }
                    records.append(record)
                    total_processed_frames += 1

        # Write manifest.jsonl
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        logger.info(f"Finished {dataset_name}: {total_processed_frames} standardized frames generated.")
        return total_processed_frames

    def _process_ffhq(self, raw_dir: Path, out_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
        records = []
        count = 0
        img_files = list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpg"))
        real_dir = out_dir / "real"
        real_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            crop_res = self.detector.detect_and_crop(img)
            out_file = real_dir / img_path.name
            cv2.imwrite(str(out_file), crop_res.cropped_face)

            rec = {
                "dataset": "ffhq",
                "video_source": str(img_path.relative_to(self.raw_root)),
                "relative_image_path": str(out_file.relative_to(self.processed_root)),
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "label_binary": 0,
                "label_text": "REAL",
                "manipulation_type": "original",
                "detector_used": crop_res.detector_type,
                "detector_confidence": crop_res.confidence,
                "width": self.target_size[0],
                "height": self.target_size[1]
            }
            records.append(rec)
            count += 1
        return records, count

    def _infer_label(self, dataset_name: str, video_path: Path) -> Dict[str, Any]:
        """Infers ground truth label from dataset directory hierarchy and naming."""
        path_str = str(video_path).lower()

        if dataset_name == "faceforensics":
            if "original" in path_str:
                return {"label_binary": 0, "label_text": "REAL", "manipulation_type": "original"}
            for manip in ["deepfakes", "faceswap", "face2face", "neuraltextures"]:
                if manip in path_str:
                    return {"label_binary": 1, "label_text": "FAKE", "manipulation_type": manip}
            return {"label_binary": 1, "label_text": "FAKE", "manipulation_type": "unknown_manip"}

        elif dataset_name == "celeb_df_v2":
            if "synthesis" in path_str:
                return {"label_binary": 1, "label_text": "FAKE", "manipulation_type": "celeb_synthesis"}
            return {"label_binary": 0, "label_text": "REAL", "manipulation_type": "original"}

        elif dataset_name == "dfdc":
            # Check metadata.json if present
            meta_json = video_path.parent / "metadata.json"
            if meta_json.exists():
                try:
                    with open(meta_json, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    vid_meta = meta.get(video_path.name, {})
                    label = vid_meta.get("label", "REAL").upper()
                    return {
                        "label_binary": 1 if label == "FAKE" else 0,
                        "label_text": label,
                        "manipulation_type": "dfdc_synthesis" if label == "FAKE" else "original"
                    }
                except Exception:
                    pass
            # Fallback
            is_fake = "fake" in path_str
            return {
                "label_binary": 1 if is_fake else 0,
                "label_text": "FAKE" if is_fake else "REAL",
                "manipulation_type": "dfdc_synthesis" if is_fake else "original"
            }

        elif dataset_name == "uadfv":
            is_fake = "fake" in path_str
            return {
                "label_binary": 1 if is_fake else 0,
                "label_text": "FAKE" if is_fake else "REAL",
                "manipulation_type": "uadfv_fake" if is_fake else "original"
            }

        elif dataset_name == "voxceleb2":
            return {"label_binary": 0, "label_text": "REAL", "manipulation_type": "original"}

        return {"label_binary": 0, "label_text": "REAL", "manipulation_type": "unspecified"}


if __name__ == "__main__":
    standardizer = DatasetStandardizer()
    results = standardizer.process_all(max_frames_per_video=8)
    print("\nStandardization Complete. Output frames per dataset:")
    for ds, count in results.items():
        print(f"  {ds}: {count} standardized frames")
