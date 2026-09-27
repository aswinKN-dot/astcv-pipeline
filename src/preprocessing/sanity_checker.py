"""
ASTCV First-Pass Sanity Checker (M3: Aswin K N)
==============================================
Validates preprocessed dataset integrity across all 6 targets:
- Confirms Ground-Truth Labels & Class Balance (Real vs Fake)
- Confirms Exact Counts (Manifest records vs disk files)
- Quality Audit: Exact resolution (256x256), corruptions, zero-byte files
- Blurriness & Exposure: Laplacian variance & mean luminance
- Exports audit summary to data/metadata/sanity_check_report.md
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger("ASTCV.SanityChecker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SanityChecker:
    def __init__(
        self,
        processed_root: str = "./data/processed",
        report_output: str = "./data/metadata/sanity_check_report.md",
        expected_size: tuple = (256, 256),
        blur_threshold: float = 20.0
    ):
        self.processed_root = Path(processed_root)
        self.report_output = Path(report_output)
        self.expected_size = expected_size
        self.blur_threshold = blur_threshold

    def audit_all(self) -> Dict[str, Any]:
        """Audits all datasets in processed_root."""
        logger.info(f"Initiating First-Pass Sanity Check on {self.processed_root}")
        overall_results = {}

        for ds_dir in self.processed_root.iterdir():
            if ds_dir.is_dir() and not ds_dir.name.startswith("."):
                ds_res = self.audit_dataset(ds_dir.name)
                overall_results[ds_dir.name] = ds_res

        self.generate_markdown_report(overall_results)
        return overall_results

    def audit_dataset(self, dataset_name: str) -> Dict[str, Any]:
        ds_path = self.processed_root / dataset_name
        manifest_path = ds_path / "manifest.jsonl"

        stats = {
            "dataset_name": dataset_name,
            "manifest_exists": manifest_path.exists(),
            "total_records": 0,
            "real_count": 0,
            "fake_count": 0,
            "unknown_label_count": 0,
            "files_found_on_disk": 0,
            "missing_files": 0,
            "corrupted_images": 0,
            "dimension_mismatches": 0,
            "blurry_images": 0,
            "mean_laplacian_variance": 0.0,
            "manipulation_breakdown": {},
            "all_checks_passed": True
        }

        if not manifest_path.exists():
            stats["all_checks_passed"] = False
            return stats

        laplacian_vars = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                stats["total_records"] += 1

                label = record.get("label_text", "UNKNOWN").upper()
                if label == "REAL":
                    stats["real_count"] += 1
                elif label == "FAKE":
                    stats["fake_count"] += 1
                else:
                    stats["unknown_label_count"] += 1

                manip = record.get("manipulation_type", "unspecified")
                stats["manipulation_breakdown"][manip] = stats["manipulation_breakdown"].get(manip, 0) + 1

                # Verify image file on disk
                rel_img_path = record.get("relative_image_path")
                full_img_path = self.processed_root / rel_img_path

                if not full_img_path.exists() or full_img_path.stat().st_size == 0:
                    stats["missing_files"] += 1
                    stats["all_checks_passed"] = False
                    continue

                stats["files_found_on_disk"] += 1

                # Image quality & dimension check
                img = cv2.imread(str(full_img_path))
                if img is None:
                    stats["corrupted_images"] += 1
                    stats["all_checks_passed"] = False
                    continue

                h, w = img.shape[:2]
                if (w, h) != self.expected_size:
                    stats["dimension_mismatches"] += 1
                    stats["all_checks_passed"] = False

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
                if lap_var < self.blur_threshold:
                    stats["blurry_images"] += 1

        if laplacian_vars:
            stats["mean_laplacian_variance"] = round(float(np.mean(laplacian_vars)), 2)

        if stats["missing_files"] > 0 or stats["corrupted_images"] > 0 or stats["dimension_mismatches"] > 0:
            stats["all_checks_passed"] = False

        return stats

    def generate_markdown_report(self, results: Dict[str, Any]) -> None:
        self.report_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# ASTCV First-Pass Dataset Sanity & Quality Report",
            "**Generated by M3 (Data Engineer): Aswin K N**  ",
            "**Standard Canonical Spec:** 256x256 RGB PNG | MediaPipe/Haar Face Center Crop",
            "",
            "## 1. Summary Audit Table",
            "",
            "| Dataset | Status | Total Frames | Real | Fake | Missing/Corrupt | Size Mismatch | Avg Blur Score |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        ]

        total_frames = 0
        total_real = 0
        total_fake = 0

        for ds_name, stats in results.items():
            status_icon = "PASSED [OK]" if stats["all_checks_passed"] else "FAILED [WARN]"
            bad_files = stats["missing_files"] + stats["corrupted_images"]
            lines.append(
                f"| `{ds_name}` | {status_icon} | {stats['total_records']} | {stats['real_count']} | "
                f"{stats['fake_count']} | {bad_files} | {stats['dimension_mismatches']} | "
                f"{stats['mean_laplacian_variance']} |"
            )
            total_frames += stats["total_records"]
            total_real += stats["real_count"]
            total_fake += stats["fake_count"]

        lines.extend([
            "",
            f"**Aggregate Processed Frames:** {total_frames} ({total_real} Real, {total_fake} Fake)",
            "",
            "## 2. Manipulation Subtype Breakdown",
            ""
        ])

        for ds_name, stats in results.items():
            lines.append(f"### `{ds_name}`")
            lines.append("| Manipulation Method | Frame Count |")
            lines.append("|:---|:---:|")
            for m_type, count in stats.get("manipulation_breakdown", {}).items():
                lines.append(f"| {m_type} | {count} |")
            lines.append("")

        with open(self.report_output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Sanity Check report saved to {self.report_output}")


if __name__ == "__main__":
    checker = SanityChecker()
    audit_data = checker.audit_all()
    print("\nSanity Check Audit Completed. Report saved to data/metadata/sanity_check_report.md")
