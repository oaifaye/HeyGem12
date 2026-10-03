"""PFLD-106 facial landmark model (ONNX Runtime)."""

from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class PFLDLandmarker:
    """98-point facial landmark detector (pfpld_robust_sim).

    The model expects a 112x112 face crop produced by 5-point similarity
    alignment (standard PFLD preprocessing), NOT a raw bbox crop.
    """

    # 5-point reference coordinates for 112x112 alignment
    # (order: left eye, right eye, nose, left mouth, right mouth)
    # The model was trained on crops with slightly more margin than the raw
    # arcface-112 template, so the reference is scaled up by REF_SCALE.
    REF_SCALE = 1.25
    REF_112 = (np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32) - 56.0) * REF_SCALE + 56.0

    # PFLD-98 semantic layout (verified against YuNet 5-point anchors)
    NUM_POINTS = 98
    MOUTH_RANGE = range(76, 96)   # 20 mouth points (outer 76-87 + inner 88-95)
    CONTOUR_RANGE = range(0, 33)
    NOSE_BRIDGE_MID = 53          # dlib-68 point 29 equivalent
    NOSE_TIP = 54                 # dlib-68 point 33 equivalent
    MOUTH_CORNER_L = 76           # dlib-68 point 48 equivalent
    MOUTH_CORNER_R = 82           # dlib-68 point 54 equivalent
    PUPIL_L = 96
    PUPIL_R = 97

    def __init__(self, model_path: str, input_size: int = 112):
        from heygem12.ortutil import make_session

        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def _run(self, aligned: np.ndarray) -> np.ndarray:
        """Run the model on an aligned 112x112 BGR image; returns (98, 2) in that image.

        The ``landms`` head emits 196 interleaved values ``[x0, y0, x1, y1, ...]``
        normalised to [0, 1]; reshaping to (98, 2) is the only parsing that
        reproduces the canonical 98-point layout (contour 0-32, brows 33-50,
        nose 51-59, eyes 60-75, mouth 76-95, pupils 96-97). Splitting the
        vector in half yields scrambled points (verified against YuNet's
        5 keypoints: nose error 2.9px interleaved vs 18.7px split).
        """
        blob = aligned.astype(np.float32)
        blob = blob / 255.0 - 0.5
        blob = blob.transpose(2, 0, 1)[None]
        out_names = [o.name for o in self.session.get_outputs()]
        outs = self.session.run(None, {self.input_name: blob})
        r = None
        if "landms" in out_names:
            r = outs[out_names.index("landms")].reshape(-1)
        else:
            for o in outs:
                rr = o.reshape(-1)
                if rr.size == 2 * self.NUM_POINTS:
                    r = rr
                    break
        if r is None or r.size != 2 * self.NUM_POINTS:
            raise RuntimeError(f"no landmark output found in model outputs: {[o.shape for o in outs]}")
        pts = r.reshape(self.NUM_POINTS, 2).astype(np.float32)
        return pts * self.input_size

    # 5-point set matching the arcface template order
    # (L eye, R eye, nose tip, L mouth corner, R mouth corner), expressed in
    # PFLD-98 indices. Needed because YuNet's own 5 keypoints are unreliable for
    # alignment: on a static talking head their eye-line tilt swings between
    # -0.4 deg and -4.1 deg frame to frame (PFLD pupils stay within 0.6 deg of
    # level), which both misaligns the crop and makes it wobble temporally.
    FIVE_FROM_98 = (96, 97, 54, 76, 82)

    def five_points(self, pts98: np.ndarray) -> np.ndarray:
        """Reduce 98 landmarks to the arcface-order 5-point set."""
        return np.asarray(pts98, np.float32)[list(self.FIVE_FROM_98)]

    def predict_aligned(self, img: np.ndarray, five_kps: np.ndarray) -> np.ndarray:
        """Returns (98, 2) landmarks in original image coordinates.

        Args:
            img: original BGR image.
            five_kps: (5, 2) facial keypoints in image coords, ordered
                (left eye, right eye, nose, left mouth, right mouth).
        """
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(five_kps, dtype=np.float32), self.REF_112, method=cv2.LMEDS)
        if matrix is None:
            raise RuntimeError("failed to estimate alignment transform")
        aligned = cv2.warpAffine(img, matrix, (self.input_size, self.input_size))
        pts = self._run(aligned)
        # map back to original image coordinates
        hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        inv = cv2.invertAffineTransform(matrix)
        return (hom @ inv.T).astype(np.float32)

    def predict(self, face_crop: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Returns (98, 2) landmark coordinates in original image space.

        Args:
            face_crop: square face crop (any size); landmarks are assumed to be
                roughly aligned already (this path is kept for tight crops).
        """
        lms_norm = self._run_on_crop(face_crop)
        x1, y1, x2, y2 = crop_box
        out = lms_norm.copy()
        out[:, 0] = out[:, 0] / self.input_size * (x2 - x1) + x1
        out[:, 1] = out[:, 1] / self.input_size * (y2 - y1) + y1
        return out

    def _run_on_crop(self, face_crop: np.ndarray) -> np.ndarray:
        aligned = cv2.resize(face_crop, (self.input_size, self.input_size))
        return self._run(aligned)
