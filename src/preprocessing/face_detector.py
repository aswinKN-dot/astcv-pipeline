"""
ASTCV Face Detection & Alignment Module (M3: Aswin K N)
======================================================
Detects facial regions, applies configurable margin padding (to preserve
forehead and jawline for M1's spatial reflectance analysis), and crops
faces to canonical square dimensions.
Designed to be multi-detector resilient (YuNet, MediaPipe, Haar, Fallback).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class FaceCropResult:
    def __init__(
        self,
        cropped_face: np.ndarray,
        bbox: Tuple[int, int, int, int],  # x, y, w, h
        confidence: float,
        detector_type: str
    ):
        self.cropped_face = cropped_face
        self.bbox = bbox
        self.confidence = confidence
        self.detector_type = detector_type


class FaceDetector:
    def __init__(self, target_size: Tuple[int, int] = (256, 256), margin: float = 1.3):
        self.target_size = target_size
        self.margin = margin
        self._init_detectors()

    def _init_detectors(self):
        # 1. Check CascadeClassifier
        self.haar_cascade = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.haar_cascade = None

        # 2. Check MediaPipe
        self.mp_face_detection = None
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        except Exception:
            self.mp_face_detection = None

    def detect_and_crop(self, frame_bgr: np.ndarray) -> FaceCropResult:
        """
        Detects primary face, applies margin, square-crops, and resizes to target_size.
        Guarantees a valid FaceCropResult under all conditions.
        """
        h, w = frame_bgr.shape[:2]

        # Method 1: MediaPipe if available
        if self.mp_face_detection:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(frame_rgb)
                if results and results.detections:
                    det = results.detections[0]
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * w)
                    y = int(bb.ymin * h)
                    bw = int(bb.width * w)
                    bh = int(bb.height * h)
                    conf = float(det.score[0]) if det.score else 0.9
                    return self._crop_with_margin(frame_bgr, (x, y, bw, bh), conf, "mediapipe")
            except Exception:
                pass

        # Method 2: Haar Cascade if available
        if self.haar_cascade and not self.haar_cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                )
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda b: b[2] * b[3])
                    x, y, bw, bh = largest_face
                    return self._crop_with_margin(frame_bgr, (x, y, bw, bh), 0.85, "haar_cascade")
            except Exception:
                pass

        # Method 3: Skin-color segmentation heuristic
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            # Standard human skin HSV range
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid_cnts = [c for c in contours if cv2.contourArea(c) > (w * h * 0.05)]
                if valid_cnts:
                    largest = max(valid_cnts, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(largest)
                    return self._crop_with_margin(frame_bgr, (x, y, bw, bh), 0.70, "skin_heuristic")
        except Exception:
            pass

        # Method 4: Centered Region of Interest fallback
        return self._center_crop_fallback(frame_bgr)

    def _crop_with_margin(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        detector_name: str
    ) -> FaceCropResult:
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]

        cx = x + bw // 2
        cy = y + bh // 2
        crop_size = int(max(bw, bh) * self.margin)

        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, cx + crop_size // 2)
        y2 = min(h, cy + crop_size // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._center_crop_fallback(frame)

        ch, cw = crop.shape[:2]
        target_dim = max(ch, cw)
        padded = np.zeros((target_dim, target_dim, 3), dtype=np.uint8)
        pad_x = (target_dim - cw) // 2
        pad_y = (target_dim - ch) // 2
        padded[pad_y:pad_y + ch, pad_x:pad_x + cw] = crop

        resized = cv2.resize(padded, self.target_size, interpolation=cv2.INTER_AREA)
        return FaceCropResult(
            cropped_face=resized,
            bbox=bbox,
            confidence=confidence,
            detector_type=detector_name
        )

    def _center_crop_fallback(self, frame: np.ndarray) -> FaceCropResult:
        h, w = frame.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        crop = frame[y1:y1 + side, x1:x1 + side]
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
        return FaceCropResult(
            cropped_face=resized,
            bbox=(x1, y1, side, side),
            confidence=0.5,
            detector_type="center_crop_fallback"
        )
