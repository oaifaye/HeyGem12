"""Face detection facade (default: OpenCV YuNet)."""

import os
from typing import Optional

import cv2
import numpy as np


class YuNetFaceDetector:
    """OpenCV YuNet face detector (ONNX), robust on full scenes."""

    def __init__(self, model_path: str, input_size: int = 320, score_thresh: float = 0.6):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"yunet model not found: {model_path}")
        self.model_path = model_path
        self.input_size = input_size
        self.score_thresh = score_thresh
        # silence dnn5 "Targets are not supported by the new graph engine" warnings
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            try:
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            except AttributeError:
                pass
        self._detectors = {}  # (w, h) -> FaceDetectorYN, cached per input size

    def _get_detector(self, w: int, h: int):
        key = (w, h)
        if key not in self._detectors:
            detector = cv2.FaceDetectorYN.create(
                self.model_path, "", (w, h), self.score_thresh, 0.3, 5000)
            self._detectors[key] = detector
        return self._detectors[key]

    def detect(self, img: np.ndarray) -> Optional[dict]:
        h, w = img.shape[:2]
        det = self._get_detector(w, h)
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        # faces: (n, 15) -> x, y, w, h, landmarks(10), score
        best = max(faces, key=lambda f: f[14])
        x, y, bw, bh = best[:4].astype(np.float32)
        bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        kps = best[4:14].reshape(5, 2).astype(np.float32)
        return {"bbox": bbox, "score": float(best[14]), "kps": kps}

    def crop_face(self, img: np.ndarray, det: dict, size: int = 112) -> np.ndarray:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) / 2 * 1.2
        h, w = img.shape[:2]
        x1c, x2c = int(max(0, cx - half)), int(min(w, cx + half))
        y1c, y2c = int(max(0, cy - half)), int(min(h, cy + half))
        crop = img[y1c:y2c, x1c:x2c]
        self._last_crop_box = (x1c, y1c, x2c, y2c)
        return cv2.resize(crop, (size, size))
