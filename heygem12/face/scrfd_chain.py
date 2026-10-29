"""SCRFD detection chain — faithful reimplementation of the original HeyGem
``face_lib.face_detect_and_align`` pipeline.

Recovered from ``face_align_5_landmarks.so`` + ``trans_dh_service.so`` by
Ghidra decompilation (see dev/ notes). The original ``FaceDetectThread`` does:

    self.fd = FaceDetect5Landmarks(mode='scrfd_500m')      # tracking=False
    bboxes, kpss = self.fd.get_bboxes(frame, min_bbox_size=64)   # min_bbox_size IGNORED
    if len(bboxes) != 0:
        face_image, mat_rev, roi_box = self.fd.get_single_face(
            crop_size=512, mode='mtcnn_512', apply_roi=True)
        return [frame, face_image, mat_rev, roi_box]

Key recovered facts:
- SCRFD runs through the insightface-style wrapper (aspect-preserving
  letterbox onto the 640x640 canvas, blob mean=127.5 / std=128.0, BGR->RGB
  swap, NMS 0.4, ``metric='default'`` = area - 2*center_dist^2 for top-k).
- ``tracking=False``: every frame is a fresh detection; there is NO temporal
  filter in the original chain. Face selection = argmax bbox AREA.
- ``get_single_face(crop_size=512, mode='mtcnn_512', apply_roi=True)``:
  expands the winning bbox into an ROI, re-expresses the 5 kps in ROI coords,
  then similarity-warps the ROI to the mtcnn 512x512 template.
- cv2box's ``Normalize(x).np_norm()`` is plain L2 normalization; the custom
  ``norm_crop`` is a standard template similarity warp (returned with M).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# mtcnn / arcface share the same 5-point reference template (112x112 canvas).
_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# crop_size / mode -> (output_size, template_scale). 'mtcnn_512' (the mode the
# original FaceDetectThread uses) maps the 112 template onto a 512x512 canvas.
_MODE_TEMPLATES = {
    "default": (112, 1.0),
    "mtcnn_512": (512, 512.0 / 112.0),
    "mtcnn_256": (256, 256.0 / 112.0),
    "arcface_512": (512, 512.0 / 112.0),
    "arcface": (112, 1.0),
    "default_95": (480, 480.0 / 112.0),
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """cv2box.Normalize(x).np_norm() — L2 normalization."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def norm_crop(image_rgb: np.ndarray, kps: np.ndarray, crop_size: int = 512,
              mode: str = "mtcnn_512") -> Tuple[np.ndarray, np.ndarray]:
    """Similarity-warp ``image_rgb`` so the 5 ``kps`` land on the reference
    template of the given mode, on a ``crop_size`` square canvas.

    Returns (aligned_rgb, M) with M the 2x3 frame->crop affine. This mirrors
    the original cv2box.utils.math.norm_crop behaviour (template similarity
    fit + warpAffine).
    """
    out_size, scale = _MODE_TEMPLATES[mode]
    template = _TEMPLATE_112 * scale
    M, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, dtype=np.float32), template)
    aligned = cv2.warpAffine(image_rgb, M, (out_size, out_size))
    return aligned, M


def apply_roi_func(image_rgb: np.ndarray, bbox: np.ndarray,
                   kps: np.ndarray) -> Tuple[np.ndarray, list, np.ndarray]:
    """Crop an expanded ROI around ``bbox`` and re-express ``kps`` in ROI
    coords. Mirrors the original module-level helper (margin 0.5 of the box
    per side; calibrated against the original in A/B dumps).
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    nx1 = int(max(0.0, x1 - bw * 0.5))
    ny1 = int(max(0.0, y1 - bh * 0.5))
    nx2 = int(min(float(w), x2 + bw * 0.5))
    ny2 = int(min(float(h), y2 + bh * 0.5))
    roi = image_rgb[ny1:ny2, nx1:nx2]
    roi_box = [nx1, ny1, nx2, ny2]
    roi_kpss = np.asarray(kps, dtype=np.float32) - np.array([nx1, ny1], np.float32)
    return roi, roi_box, roi_kpss


class InsightfaceSCRFD:
    """SCRFD detector with insightface preprocessing semantics.

    aspect-preserving letterbox onto the input-size canvas, blob
    (x-127.5)/128.0 with BGR->RGB swap, stride 8/16/32 decode, score
    threshold, NMS 0.4, and the insightface top-k rule
    (``metric='default'`` -> area - 2 * center_distance^2).
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 session=None):
        import onnxruntime as ort
        from heygem12.ortutil import make_session
        self.session = session if session is not None else make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size            # (width, height)
        self.input_mean = 127.5
        self.input_std = 128.0
        self.nms_thresh = 0.4
        # scrfd_500m_bnkps: 9 outputs -> fmc=3, 2 anchors, with kps
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self._center_cache = {}

    # ------------------------------------------------------------------ core
    def _forward(self, img_bgr: np.ndarray, threshold: float):
        """img_bgr: already resized+letterboxed onto the canvas (BGR uint8)."""
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        ih, iw = blob.shape[2], blob.shape[3]
        batched = len(net_outs[0].shape) == 3 and net_outs[0].shape[0] == 1
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            if batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                kps_preds = net_outs[idx + self.fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                kps_preds = net_outs[idx + self.fmc * 2] * stride

            height, width = ih // stride, iw // stride
            key = (height, width, stride)
            anchor_centers = self._center_cache.get(key)
            if anchor_centers is None:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1).reshape(-1, 2)
                if len(self._center_cache) < 100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores.ravel() >= threshold)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores.reshape(-1, 1)[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss = _distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
            kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def detect_faces(self, img_rgb: np.ndarray, thresh: float = 0.5,
                     max_num: int = 0, metric: str = "default"
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect on an RGB image; returns (bboxes (N,5) xyxy+score, kpss (N,5,2))."""
        bgr = img_rgb[:, :, ::-1]
        im_ratio = bgr.shape[0] / bgr.shape[1]
        model_ratio = self.input_size[1] / self.input_size[0]
        if im_ratio > model_ratio:
            new_h = self.input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = self.input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / bgr.shape[0]
        resized = cv2.resize(bgr, (new_w, new_h))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), np.uint8)
        det_img[:new_h, :new_w] = resized

        scores_list, bboxes_list, kpss_list = self._forward(det_img, thresh)
        if not scores_list or sum(s.size for s in scores_list) == 0:
            return (np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32))

        scores = np.vstack(scores_list)
        order = np.argsort(-scores.ravel(), kind="stable")
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order]
        kpss = kpss[order]
        keep = self._nms(pre_det)
        det = pre_det[keep]
        kpss = kpss[keep]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = (img_rgb.shape[0] // 2, img_rgb.shape[1] // 2)
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == "max" else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            kpss = kpss[bindex]
        return det, kpss

    def _nms(self, dets: np.ndarray) -> List[int]:
        thresh = self.nms_thresh
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(-scores, kind="stable")
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetect5Landmarks:
    """Faithful reimplementation of the original ``FaceDetect5Landmarks``."""

    def __init__(self, scrfd_model_path: str, mode: str = "scrfd_500m",
                 tracking: bool = False):
        self.mode = mode
        self.tracking = tracking
        self.dis_list: List[float] = []
        self.last_bboxes: list = []
        assert mode in ("scrfd", "scrfd_500m", "mtcnn")
        self.bboxes = None
        self.kpss = None
        self.image = None
        if "scrfd" in mode:
            self.det_model_scrfd = InsightfaceSCRFD(scrfd_model_path)

    # ------------------------------------------------------------------ api
    def get_bboxes(self, image: np.ndarray, nms_thresh: float = 0.5,
                   max_num: int = 0, min_bbox_size: Optional[float] = None):
        """BGR frame in -> cached RGB + bboxes/kpss.

        ``min_bbox_size`` is accepted for interface compatibility but is NOT
        applied — matching the original, where the parameter is ignored.
        """
        self.image = image[:, :, ::-1]          # CVImage(image).rgb()
        if not self.tracking:
            if "scrfd" in self.mode:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
                    self.image, thresh=nms_thresh, max_num=max_num,
                    metric="default")
            return self.bboxes, self.kpss
        if len(self.last_bboxes) == 0:
            self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
                self.image, thresh=nms_thresh, max_num=1, metric="default")
            self.last_bboxes = self.bboxes
            return None
        self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
            self.image, thresh=nms_thresh, max_num=0, metric="default")
        self.bboxes, self.kpss = self.tracking_filter()
        return None

    def bboxes_filter(self, min_bbox_size: float) -> None:
        min_area = np.power(min_bbox_size, 2)
        width = self.bboxes[:, 2] - self.bboxes[:, 0]
        height = self.bboxes[:, 3] - self.bboxes[:, 1]
        area_list = width * height
        min_index = np.where(area_list < min_area)
        self.bboxes = np.delete(self.bboxes, min_index, axis=0)
        self.kpss = np.delete(self.kpss, min_index, axis=0)

    def tracking_filter(self):
        if len(self.bboxes) > 0:
            for i in range(len(self.bboxes)):
                r = self.bboxes[i]
                dis = np.linalg.norm(
                    _l2_normalize(r) - _l2_normalize(self.last_bboxes[0]))
                self.dis_list.append(dis)
            best_index = int(np.argmin(np.array(self.dis_list)))
            self.dis_list = []
            self.last_bboxes = [self.bboxes[best_index]]
            return self.last_bboxes, [self.kpss[best_index]]

    def get_single_face(self, image: Optional[np.ndarray] = None,
                        crop_size: int = 512, mode: str = "mtcnn_512",
                        apply_roi: bool = False, best_index: Optional[int] = None):
        """Returns (align_img BGR, mat_rev, roi_box) for the selected face.

        ``best_index`` overrides the default largest-area selection -- the
        original's rule is argmax detection score, chosen by the caller.
        """
        assert mode in ("default", "mtcnn_512", "mtcnn_256", "arcface_512",
                        "arcface", "default_95")
        if image is not None:
            self.get_bboxes(image)
        bboxes = self.bboxes
        if len(bboxes) > 0:
            if best_index is None:
                if self.tracking:
                    best_index = int(np.argmin(np.array(self.dis_list)))
                else:
                    area_list = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    best_index = int(np.argmax(np.array(area_list)))
            bbox = bboxes[best_index]
            kps = self.kpss[best_index]
            if apply_roi:
                src, roi_box, roi_kpss = apply_roi_func(self.image, bbox, kps)
            else:
                src, roi_box, roi_kpss = self.image, None, kps
            align_img, M = norm_crop(src, roi_kpss, crop_size=crop_size, mode=mode)
            mat_rev = cv2.invertAffineTransform(M)
            align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
            return align_img, mat_rev, roi_box
