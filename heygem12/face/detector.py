"""SCRFD face detector (ONNX Runtime).

Note: the scrfd_500m ONNX shipped with HeyGem is unreliable on full scenes
(false positives on background textures, misses the dominant face). We
therefore run it over a small center-crop pyramid and validate candidates
geometrically (aspect ratio + size), which is robust for talking-head videos
where the speaker face is dominant and near the frame center.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

_FEATURE_STRIDES = (8, 16, 32)
NMS_THRESHOLD = 0.4
# geometric sanity: real face boxes are roughly square-ish and not tiny
ASPECT_RANGE = (0.65, 1.55)
MIN_SIZE_RATIO = 0.06


class SCRFD:
    def __init__(self, model_path: str, input_size: int = 640, score_thresh: float = 0.5):
        from heygem12.ortutil import make_session

        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_thresh = score_thresh

    # ------------------------------------------------------------- single shot
    def _detect_on_canvas(self, canvas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on a square canvas; returns scores/boxes/kps in canvas coords."""
        blob = ((canvas.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        outs = self.session.run(None, {self.input_name: blob})
        num_out = len(outs)
        per_stride = num_out // 3
        scores_all, boxes_all, kps_all = [], [], []
        for i in range(per_stride):
            stride = _FEATURE_STRIDES[i]
            scores = outs[i].reshape(-1)
            boxes = outs[i + per_stride].reshape(-1, 4)
            kps = outs[i + per_stride * 2].reshape(-1, 10)
            grid = self.input_size // stride
            ys, xs = np.mgrid[0:grid, 0:grid]
            # prediction layout is INTERLEAVED per cell: (cell0_a0, cell0_a1,
            # cell1_a0, ...) -- anchor-major concatenation was the long-standing
            # bug that made every kps land on background (verified against the
            # real face position on demo1/2/3).
            cx = (xs.reshape(-1) + 0.5) * stride
            cy = (ys.reshape(-1) + 0.5) * stride
            ax = np.repeat(cx, 2)
            ay = np.repeat(cy, 2)
            keep = scores > self.score_thresh
            if not np.any(keep):
                continue
            idx = np.where(keep)[0]
            ax, ay = ax[idx], ay[idx]
            b = boxes[idx] * stride
            k = kps[idx] * stride
            x1 = ax - b[:, 0]
            y1 = ay - b[:, 1]
            x2 = ax + b[:, 2]
            y2 = ay + b[:, 3]
            kp = np.stack([ax[:, None] + k[:, 0::2], ay[:, None] + k[:, 1::2]], axis=2)
            scores_all.append(scores[idx])
            boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
            kps_all.append(kp)
        if not scores_all:
            return (np.zeros(0), np.zeros((0, 4)), np.zeros((0, 5, 2)))
        return (np.concatenate(scores_all),
                np.concatenate(boxes_all),
                np.concatenate(kps_all))

    # ---------------------------------------------------------------- detection
    def detect(self, img: np.ndarray, multi_scale: bool = True) -> Optional[dict]:
        """Returns dict with bbox (x1,y1,x2,y2), score, kps (5,2) in image coords."""
        h, w = img.shape[:2]
        candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []

        def _collect(canvas: np.ndarray, offset: Tuple[float, float], src_size: Tuple[int, int]):
            """canvas is a square 640 image of a src_size (w, h) region at offset (x0, y0)."""
            scores, boxes, kps = self._detect_on_canvas(canvas)
            if len(scores) == 0:
                return
            sx = src_size[0] / self.input_size
            sy = src_size[1] / self.input_size
            scale_xy = np.array([sx, sy, sx, sy], dtype=np.float32)
            kps_scale = np.array([sx, sy], dtype=np.float32)
            off = np.array([offset[0], offset[1]], dtype=np.float32)
            keep = self._nms(boxes, scores)
            for i in keep:
                box = boxes[i] * scale_xy + np.array([offset[0], offset[1]] * 2, dtype=np.float32)
                candidates.append((float(scores[i]), box.astype(np.float32),
                                   kps[i] * kps_scale + off))

        def _letterbox(region: np.ndarray):
            """Aspect-preserving resize of a (h, w) region onto the square canvas.
            Returns (canvas, pad_x, pad_y, scale) mapping canvas coords back to
            region coords: region = (canvas - pad) / scale."""
            rh, rw = region.shape[:2]
            scale = self.input_size / max(rh, rw)
            nw, nh = int(round(rw * scale)), int(round(rh * scale))
            resized = cv2.resize(region, (nw, nh))
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
            canvas[py:py + nh, px:px + nw] = resized
            return canvas, px, py, scale

        def _collect_letterbox(region: np.ndarray, offset: Tuple[float, float]):
            canvas, px, py, sc = _letterbox(region)
            scores, boxes, kps = self._detect_on_canvas(canvas)
            if len(scores) == 0:
                return
            keep = self._nms(boxes, scores)
            off = np.array(offset, dtype=np.float32)
            off4 = np.array([off[0], off[1], off[0], off[1]], dtype=np.float32)
            for i in keep:
                box = (boxes[i] - np.array([px, py, px, py], dtype=np.float32)) / sc + off4
                kp = (kps[i] - np.array([px, py], dtype=np.float32)) / sc + off
                candidates.append((float(scores[i]), box.astype(np.float32),
                                   kp.astype(np.float32)))

        # pass 1: full-frame letterbox (uniform scale; content maps 1/scale)
        _collect_letterbox(img, (0.0, 0.0))

        # pass 2: center-crop pyramid (aspect-preserving; squashing destroys kps)
        if multi_scale:
            for frac in (0.75, 0.5, 0.35):
                ch, cw = int(h * frac), int(w * frac)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                _collect_letterbox(img[y0: y0 + ch, x0: x0 + cw], (float(x0), float(y0)))

        valid = [c for c in candidates if self._valid(c[1], img.shape)]
        if not valid:
            return None
        # prefer the largest valid face
        best = max(valid, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))
        return {"bbox": best[1], "score": best[0], "kps": best[2].astype(np.float32)}

    @staticmethod
    def _valid(bbox: np.ndarray, shape: Tuple[int, ...]) -> bool:
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        if x1 < -0.05 * w or y1 < -0.05 * h or x2 > 1.05 * w or y2 > 1.05 * h:
            return False
        aspect = bw / bh
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            return False
        if min(bw, bh) < MIN_SIZE_RATIO * min(h, w):
            return False
        return True

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(a[:, 0], b[0])
        y1 = np.maximum(a[:, 1], b[1])
        x2 = np.minimum(a[:, 2], b[2])
        y2 = np.minimum(a[:, 3], b[3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / np.maximum(area_a + area_b - inter, 1e-9)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            iou = self._iou(boxes[order[1:]], boxes[i])
            order = order[1:][iou < NMS_THRESHOLD]
        return keep

    def crop_face(self, img: np.ndarray, det: dict, size: int = 112) -> np.ndarray:
        """Square-crop the face region around the detection bbox."""
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) / 2 * 1.15
        h, w = img.shape[:2]
        x1c, x2c = int(max(0, cx - half)), int(min(w, cx + half))
        y1c, y2c = int(max(0, cy - half)), int(min(h, cy + half))
        crop = img[y1c:y2c, x1c:x2c]
        self._last_crop_box = (x1c, y1c, x2c, y2c)
        return cv2.resize(crop, (size, size))

    def detect_refined(self, img: np.ndarray, det: dict, margin: float = 1.6) -> Optional[dict]:
        """Re-run SCRFD on an upscaled crop around ``det`` for precise kps.

        The original pipeline runs SCRFD twice per frame; on a full 1080x1920
        frame the face is ~80 px in the 640 canvas and the raw five points jump
        5-20 px frame to frame.  A second pass on a 1.6x-margin face crop gives
        ~4x the effective resolution (the original's precision) with the same
        detector semantics.  Falls back to ``det`` if the refined pass fails.
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) / 2.0 * margin
        x1c = int(max(0, cx - half)); x2c = int(min(w, cx + half))
        y1c = int(max(0, cy - half)); y2c = int(min(h, cy + half))
        if x2c - x1c < 32 or y2c - y1c < 32:
            return det
        crop = img[y1c:y2c, x1c:x2c]
        # letterbox the crop onto the square canvas (aspect preserved)
        ch, cw = crop.shape[:2]
        scale = self.input_size / max(ch, cw)
        nw, nh = int(round(cw * scale)), int(round(ch * scale))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(crop, (nw, nh))
        scores, boxes, kps = self._detect_on_canvas(canvas)
        if len(scores) == 0:
            return det
        keep = self._nms(boxes, scores)
        if not keep:
            return det
        i = keep[0]
        # prefer the detection nearest the crop center
        cc = np.array([self.input_size / 2.0, self.input_size / 2.0], np.float32)
        best, best_d = None, None
        for j in keep[:8]:
            kp0 = (kps[j][0] - np.array([px, py], np.float32)) / scale
            d = float(np.hypot(float(kp0[0] - cw / 2.0), float(kp0[1] - ch / 2.0)))
            if best_d is None or d < best_d:
                best, best_d = j, d
        box = (boxes[best] - np.array([px, py, px, py], np.float32)) / scale
        box = box + np.array([x1c, y1c, x1c, y1c], np.float32)
        kp = (kps[best] - np.array([px, py], np.float32)) / scale
        kp = kp + np.array([x1c, y1c], np.float32)
        refined = {"bbox": box.astype(np.float32), "score": float(scores[best]),
                   "kps": kp.reshape(5, 2).astype(np.float32)}
        return refined
