"""
ASTCV Synthetic Mock Dataset Generator (M3: Aswin K N)
======================================================
Generates realistic sample videos and images representing the 6 target datasets:
- FaceForensics++ (Deepfakes, FaceSwap, Face2Face, original)
- Celeb-DF v2 (real, fake)
- DFDC (videos with diverse aspect ratios & JSON metadata)
- UADFV (sample real & fake clips)
- FFHQ (genuine high-res portraits)
- VoxCeleb2 (speaker video segments)

Allows the team (M1, M2, M3) to develop and test preprocessing,
alignment, sanity checks, and MLflow tracking without waiting for
academic approvals.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path


def create_synthetic_face_frame(
    width: int = 640,
    height: int = 480,
    face_color: tuple = (180, 200, 230),
    is_fake: bool = False,
    frame_idx: int = 0
) -> np.ndarray:
    """Draws a synthetic face target with simulated skin reflectance and movement."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    for y in range(height):
        img[y, :] = (int(30 + 20 * (y / height)), int(40 + 10 * (y / height)), int(50))

    # Center coords with subtle oscillation simulating head movement
    cx = int(width // 2 + 15 * np.sin(frame_idx * 0.1))
    cy = int(height // 2 + 10 * np.cos(frame_idx * 0.1))
    axes = (int(width * 0.18), int(height * 0.28))

    # Face ellipse (base skin)
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, face_color, -1)

    # Eyes
    eye_offset_x = int(axes[0] * 0.45)
    eye_offset_y = int(axes[1] * 0.25)
    cv2.circle(img, (cx - eye_offset_x, cy - eye_offset_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_offset_x, cy - eye_offset_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (cx - eye_offset_x, cy - eye_offset_y), 4, (40, 20, 10), -1)
    cv2.circle(img, (cx + eye_offset_x, cy - eye_offset_y), 4, (40, 20, 10), -1)

    # Nose bridge
    cv2.line(img, (cx, cy - 10), (cx, cy + 20), (140, 160, 190), 3)

    # Mouth
    cv2.ellipse(img, (cx, cy + int(axes[1] * 0.45)), (25, 8), 0, 0, 180, (120, 130, 200), -1)

    # If simulated fake: add subtle boundary artifact or chromatic aberration
    if is_fake:
        cv2.rectangle(img, (cx - axes[0] - 2, cy - axes[1] - 2),
                      (cx + axes[0] + 2, cy + axes[1] + 2), (0, 0, 180), 1)

    # Timestamp tag
    cv2.putText(img, f"Frame {frame_idx:04d}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def generate_synthetic_video(
    output_path: Path,
    num_frames: int = 60,
    fps: int = 30,
    resolution: tuple = (640, 480),
    is_fake: bool = False
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    for i in range(num_frames):
        frame = create_synthetic_face_frame(
            width=resolution[0],
            height=resolution[1],
            is_fake=is_fake,
            frame_idx=i
        )
        writer.write(frame)

    writer.release()


def generate_all_mock_datasets(base_dir: str = "./data/raw") -> None:
    base = Path(base_dir)
    print("Generating synthetic mock samples for all 6 target datasets...")

    # 1. FaceForensics++ (Deepfakes, FaceSwap, Face2Face, original)
    ffpp_dir = base / "faceforensics"
    for category in ["original", "Deepfakes", "Face2Face", "FaceSwap"]:
        for vid_id in ["001_002", "003_004"]:
            is_fake = (category != "original")
            out_file = ffpp_dir / category / "c23" / "videos" / f"{vid_id}.mp4"
            generate_synthetic_video(out_file, num_frames=45, fps=30, is_fake=is_fake)
    print("  [OK] FaceForensics++ mock samples created")

    # 2. Celeb-DF v2 (Celeb-real, Celeb-synthesis, YouTube-real)
    celeb_dir = base / "celeb_df_v2"
    generate_synthetic_video(celeb_dir / "Celeb-real" / "id0_0000.mp4", num_frames=50, fps=30, is_fake=False)
    generate_synthetic_video(celeb_dir / "Celeb-synthesis" / "id0_id1_0000.mp4", num_frames=50, fps=30, is_fake=True)
    generate_synthetic_video(celeb_dir / "YouTube-real" / "00001.mp4", num_frames=40, fps=25, is_fake=False)
    print("  [OK] Celeb-DF v2 mock samples created")

    # 3. DFDC (Deepfake Detection Challenge)
    dfdc_dir = base / "dfdc"
    generate_synthetic_video(dfdc_dir / "train_sample_videos" / "aagfhgtpmv.mp4", num_frames=60, fps=30, is_fake=False)
    generate_synthetic_video(dfdc_dir / "train_sample_videos" / "abarnvbtwb.mp4", num_frames=60, fps=30, is_fake=True)
    dfdc_meta = {
        "aagfhgtpmv.mp4": {"label": "REAL", "split": "train"},
        "abarnvbtwb.mp4": {"label": "FAKE", "split": "train", "original": "aagfhgtpmv.mp4"}
    }
    with open(dfdc_dir / "train_sample_videos" / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(dfdc_meta, f, indent=2)
    print("  [OK] DFDC mock samples and metadata.json created")

    # 4. UADFV
    uadfv_dir = base / "uadfv"
    generate_synthetic_video(uadfv_dir / "real" / "0001.mp4", num_frames=45, fps=30, is_fake=False)
    generate_synthetic_video(uadfv_dir / "fake" / "0001_fake.mp4", num_frames=45, fps=30, is_fake=True)
    print("  [OK] UADFV mock samples created")

    # 5. FFHQ (High-Res Images)
    ffhq_dir = base / "ffhq" / "images1024x1024"
    ffhq_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img = create_synthetic_face_frame(width=1024, height=1024, is_fake=False, frame_idx=i)
        cv2.imwrite(str(ffhq_dir / f"{i:05d}.png"), img)
    print("  [OK] FFHQ mock 1024x1024 images created")

    # 6. VoxCeleb2
    vox_dir = base / "voxceleb2" / "mp4" / "id00012" / "21Uxsk56VDQ"
    vox_dir.mkdir(parents=True, exist_ok=True)
    generate_synthetic_video(vox_dir / "00001.mp4", num_frames=60, fps=25, resolution=(720, 480), is_fake=False)
    print("  [OK] VoxCeleb2 mock samples created")

    print("\nAll mock dataset samples successfully created in", base.resolve())


if __name__ == "__main__":
    generate_all_mock_datasets()
