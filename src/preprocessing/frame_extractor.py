"""
ASTCV Frame Extraction Engine (M3: Aswin K N)
==============================================
Extracts video frames at controlled FPS / stride, standardizing frame rates
across datasets with differing source frame rates (24, 25, 29.97, 30, 60 FPS).
"""

import cv2
import logging
from pathlib import Path
from typing import List, Generator, Tuple, Optional

logger = logging.getLogger("ASTCV.FrameExtractor")


class FrameExtractor:
    def __init__(self, target_fps: Optional[float] = None, stride: int = 1):
        """
        :param target_fps: If provided, samples frames to match this effective FPS.
        :param stride: Alternatively, extract every Nth frame.
        """
        self.target_fps = target_fps
        self.stride = stride

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, cv2.typing.MatLike], None, None]:
        """
        Yields (frame_index, timestamp_ms, frame_bgr)
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video stream: {video_path}")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0
        if self.target_fps and self.target_fps > 0 and source_fps > 0:
            frame_interval = max(1.0, source_fps / self.target_fps)
        elif self.stride > 1:
            frame_interval = float(self.stride)

        frame_idx = 0
        extracted_count = 0
        next_target_frame = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx >= int(next_target_frame):
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    yield (extracted_count, timestamp_ms, frame)
                    extracted_count += 1
                    next_target_frame += frame_interval

                    if max_frames and extracted_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

    def extract_and_save(
        self,
        video_path: str,
        output_dir: str,
        file_prefix: str = "frame",
        max_frames: Optional[int] = None
    ) -> List[str]:
        """Extracts frames from video and writes them to output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for count, ts, frame in self.extract_from_video(video_path, max_frames=max_frames):
            frame_filename = f"{file_prefix}_{count:05d}.png"
            dest_file = out_path / frame_filename
            cv2.imwrite(str(dest_file), frame)
            saved_paths.append(str(dest_file))

        return saved_paths


if __name__ == "__main__":
    extractor = FrameExtractor(stride=2)
    sample_vid = "data/raw/dfdc/train_sample_videos/aagfhgtpmv.mp4"
    if Path(sample_vid).exists():
        saved = extractor.extract_and_save(sample_vid, "data/scratch/extracted_sample", max_frames=5)
        print(f"Extracted {len(saved)} frames to data/scratch/extracted_sample")
