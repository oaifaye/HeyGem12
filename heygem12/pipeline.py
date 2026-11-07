"""End-to-end lip-sync pipeline: video + audio -> lip-synced video.

Faithful port of the original HeyGem inference chain (behaviours recovered by
executing the original compiled modules under HG/ + the cv2box 0.5.9 sources;
see HANDOFF.md). Per frame:

    1. SCRFD(640) fresh detection, tracking OFF; best face = argmax det_score
       (InsightfaceSCRFD - matches the original custom wrapper to <0.4 px).
    2. chaofen_before (the original's GFPGAN pre-restore), EXACT chain:
         roi        = frame_rgb[bbox]              (plain bbox crop, no margin)
         roi_kpss   = kps - roi_origin
         M512       = estimateAffinePartial2D(roi_kpss, MTCNN_512_TEMPLATE)
         face512    = warpAffine(roi, M512, 512)   (RGB -> BGR for GFPGAN)
         restored   = GFPGAN(face512)              (blob 1/127.5, mean 127.5,
                                                    swapRB; (out+1)/2 -> [0,1])
         recovered  = recover_from_reverse_matrix(restored, roi_u8, inv(M512),
                       common_face_mask(roi_size))  (cv2box exact)
       The recovered frame replaces the roi in the frame; GFPGAN content never
       reaches the final composite directly - only through DINet's render.
    3. PFLD-98 run on the ORIGINAL's 112 framing (T112_REF, dump-fitted)
       -> five points (96,97,54,76,82) mapped back to the frame
       -> fit to TEMPLATE_CROP (dump-fitted) -> 256x256 RGB crop.
    4. DINetV1: source = crop*input_mask, ref = crop, audio = WeNet window
       (T=20) -> fake; fuse = fake*stackBlur17(fuse_mask) + crop*(1-mask).
    5. Composite (original's additive form): frame += oval * M^-1(fuse-crop).
       The delta is exactly 0 outside the fuse feather, so non-face pixels
       are bit-identical to the input (no double-resampling shimmer); the
       oval (ellipse (128,128) axes (104,116), sigma 6) only bounds the
       mouth update inside the face contour. Oval center (129,154) axes
       (91,66) sigma 6, dump-fitted in this crop geometry.

No temporal filtering anywhere (the original has none); no chaofen_after.
"""

import os
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from heygem12.audio.frontend import WeNetFeatureExtractor
from heygem12.face.scrfd_chain import InsightfaceSCRFD
from heygem12.models import load_dinet_hd

# measured from the original norm_crop (inv(M) applied to the fitted points)
MTCNN_512_TEMPLATE = np.array([
    [183.94, 246.06],
    [330.45, 245.53],
    [250.60, 313.57],
    [193.34, 392.73],
    [322.16, 391.04],
], dtype=np.float32)

# --- DINet-crop geometry, recovered by fitting against the ORIGINAL pipeline's
# runtime dumps (hooked DINetV1.forward arg1 crops, three persons).
# Chain (all fixed constants, no per-video fitting):
#   112 crop  = warp(frame, fit(scrfd_kps -> T112))          (the original's
#               PFLD input framing; T112 fitted so our PFLD on it reproduces
#               the original's dumped 98-pt landmarks to ~1.3 px)
#   five      = PFLD-98(112 crop)[96,97,54,76,82] mapped back to frame coords
#   DINet crop= warp(frame, fit(five -> T_UNIV))  (256 RGB)
# Dump-verified render equivalence (DINet out MAE vs original's dumps):
#   woman/man1 ~0.053 full / ~0.03-0.06 mouth (was 0.147/0.098 with the old
#   PFLD-REF_112x1.25 + TEMPLATE_256 crop on the woman).
# TEMPLATE_256 (the old SIFT-fitted template) is NOT the template the original
# uses: fitted per-person it lands within ~4 px of T_UNIV for the shared
# nose/mouth but puts the eyes ~3-4 px off, and cross-person it collapses.
T112_REF = np.array([
    [35.75, 48.34],
    [78.04, 44.60],
    [57.28, 70.63],
    [35.66, 98.14],
    [73.68, 98.94],
], dtype=np.float32)

TEMPLATE_CROP = np.array([
    [85.65, 55.30],
    [161.70, 56.15],
    [125.26, 96.86],
    [94.80, 136.45],
    [156.05, 138.40],
], dtype=np.float32)


def common_face_mask(mask_shape: Tuple[int, int]) -> np.ndarray:
    """cv2box.utils.util.common_face_mask, exact."""
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 256), (220, 160), 90, 0, 360, (255, 255, 255), -1)
    thres = 20
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    mask = cv2.stackBlur(mask, (201, 201))
    mask = mask / 255.0
    mask = cv2.resize(mask, mask_shape)
    return mask[..., np.newaxis]


def norm_crop(image_rgb: np.ndarray, landmark: np.ndarray,
              crop_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Original norm_crop('mtcnn_512'): similarity warp of image_rgb so the
    5 landmarks land on MTCNN_512_TEMPLATE. warpAffine(upscale) uses the
    FORWARD fit A; the INVERSE M is what recover() consumes (verified
    against the original compiled function: it returns inv(A))."""
    A, _ = cv2.estimateAffinePartial2D(
        np.asarray(landmark, np.float32), MTCNN_512_TEMPLATE)
    aligned = cv2.warpAffine(image_rgb, A, (crop_size, crop_size),
                             borderMode=cv2.BORDER_REPLICATE)
    M = cv2.invertAffineTransform(A)
    return aligned, M


def recover_from_reverse_matrix(img_fg01: np.ndarray, img_bg_u8: np.ndarray,
                                M: np.ndarray,
                                img_fg_mask: np.ndarray) -> np.ndarray:
    """cv2box CVImage.recover_from_reverse_matrix, exact."""
    h, w = img_bg_u8.shape[:2]
    fg_t = cv2.warpAffine(img_fg01, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = img_fg_mask * (fg_t * 255.0) + (1.0 - img_fg_mask) * img_bg_u8
    return out.astype(np.uint8)


class LipSyncPipeline:
    def __init__(
        self,
        checkpoints_dir: str,
        device: str = "cpu",
        restore: bool = False,        # chaofen_after (original default OFF)
        restore_source: bool = False, # chaofen_before (original gates it on a
                                      # blur/quality check; runtime dumps show it
                                      # OFF for sharp sources -> default OFF)
        fuse_mode: str = "original",  # CLI compat; only the original scheme remains
        kps_stabilize: bool = False,  # CLI compat; ignored
        render_ema: float = 0.0,      # CLI compat; ignored
        audio_device: Optional[str] = None,  # WeNet device; None follows device
    ):
        self.device = device
        # crop geometry: "orig" replicates the original HeyGem chain
        # (landmark boundingRect -> x1=xc-0.75w, y1=y-0.15w, side=1.5w,
        # temporally smoothed, rect crop -> 266 INTER_CUBIC -> [5:261] = 256
        # DINet input, paste mask = convexHull(jaw+nose) + bottom half
        # ellipse + dilate 6x6). "v24" keeps the old five-point affine warp.
        self.geom_mode = os.environ.get("HEYGEM12_GEOM", "orig")
        dinet_path = os.path.join(checkpoints_dir, "dinet_v1_20240131.pth")
        print("[heygem12] loading DINet_HD ...")
        self.model = load_dinet_hd(dinet_path, device=device)
        checkpoint = torch.load(dinet_path, map_location="cpu", weights_only=False)
        # fuse_mask with the original 17x17 stackBlur feather (float in-place)
        fm = np.asarray(checkpoint["fuse_mask"]).astype(np.float32)
        fm = cv2.stackBlur(fm, (17, 17))
        self.fuse_mask = np.clip(fm, 0.0, 1.0)[:, :, None]        # (256,256,1)
        self.input_mask = np.asarray(
            checkpoint["input_mask"]).astype(np.float32)[:, :, None]
        print("[heygem12] dinet_v1 input %s | input_mask zero frac %.2f"
              % (checkpoint.get("model_input_size"),
                 float((self.input_mask[..., 0] == 0).mean())))

        print("[heygem12] loading WeNet frontend ...")
        self.audio_frontend = WeNetFeatureExtractor(
            os.path.join(checkpoints_dir, "wenetmodel.pt"), device=audio_device or device)

        from heygem12.face.landmarks import PFLDLandmarker
        self.pfld = PFLDLandmarker(
            os.path.join(checkpoints_dir, "pfpld_robust_sim_bs1_8003.onnx"))
        self.five_idx = (96, 97, 54, 76, 82)

        # face mask in the 256 DINet-crop space (paste-back limit). Fitted from
        # the original's runtime dump in THIS crop geometry (TEMPLATE_CROP):
        # warp original output frames back to crop space and solve
        # alpha = (out_crop - ref) / (fused - ref); the original's paste mask
        # 0.5-contour is the ellipse center (129,154) semi-axes (91,66) --
        # deliberately SMALLER than the fuse feather support (y83..245): the
        # original trims the feather at the jaw/neck boundary, which is why
        # its non-face regions never flicker. (crop_mask_probe.py)
        c = np.zeros((256, 256), np.float32)
        cv2.ellipse(c, (129, 154), (91, 66), 0, 0, 360, 1.0, -1)
        self._face_fg_crop = cv2.GaussianBlur(c, (0, 0), 6.0)

        scrfd_path = os.path.join(checkpoints_dir, "scrfd_500m_bnkps_shape640x640.onnx")
        print("[heygem12] loading SCRFD (insightface semantics) ...")
        self.det = InsightfaceSCRFD(scrfd_path)

        self.restore = restore
        self.restore_source = restore_source
        self.restorer = None
        if restore or restore_source:
            try:
                gfp = os.path.join(checkpoints_dir, "GFPGANv1.4.onnx")
                if os.path.exists(gfp):
                    from heygem12.ortutil import make_session, ort_providers

                    self.restorer = make_session(gfp)
                    print(f"[heygem12] GFPGAN face restore: enabled ({ort_providers(self.restorer)})")
                else:
                    print(f"[heygem12] GFPGAN onnx not found at {gfp}, restore disabled")
            except Exception as e:  # pragma: no cover
                print(f"[heygem12] GFPGAN unavailable ({e}), restore disabled")

    # ---------------------------------------------------------------- detect
    def _detect_faces(self, frames: List[np.ndarray]) -> List[Optional[Tuple]]:
        """Per frame: (bbox, kps) of the best face (argmax det_score), or None.
        Fresh detection every frame (the original has no temporal filter)."""
        out: List[Optional[Tuple]] = []
        for i, frame in enumerate(frames):
            rgb = frame[:, :, ::-1]
            bboxes, kpss = self.det.detect_faces(rgb, thresh=0.5, max_num=0,
                                                 metric="default")
            if bboxes is None or len(bboxes) == 0:
                if i > 0 and out[-1] is not None:
                    print(f"[heygem12] WARNING: no face in frame {i}, reusing previous")
                    out.append(out[-1])
                    continue
                raise RuntimeError(f"no face detected in frame {i}")
            best = int(np.argmax(bboxes[:, -1]))
            out.append((bboxes[best].astype(np.float32), kpss[best].astype(np.float32)))
        return out

    # ---------------------------------------------------------------- restore
    def _chaofen(self, face512_bgr: np.ndarray) -> np.ndarray:
        """GFPGAN restore. Original: CVImage.blob(input_size=(512,512),
        mean=127.5, std=127.5, rgb=True) = blobFromImage(swapRB) ->
        (out+1)/2 -> [0,1] float RGB (HWC)."""
        blob = cv2.dnn.blobFromImage(face512_bgr, 1.0 / 127.5, (512, 512),
                                     (127.5, 127.5, 127.5), swapRB=True)
        out = self.restorer.run(None, {"input": blob})[0][0]    # (3,512,512)
        restored = (out.transpose(1, 2, 0) + 1.0) / 2.0         # RGB [0,1] HWC
        return np.clip(restored, 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- pipeline
    @torch.no_grad()
    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> str:
        t0 = time.perf_counter()
        video_path, audio_path = self._normalize_inputs(video_path, audio_path)
        frames, fps = self._read_frames(video_path, max_frames)
        n = len(frames)
        h0, w0 = frames[0].shape[:2]
        print(f"[heygem12] {n} frames @ {fps:.2f} fps | {w0}x{h0}")

        print("[heygem12] detecting faces (SCRFD, per frame) ...")
        t = time.perf_counter()
        dets = self._detect_faces(frames)
        t_det = time.perf_counter() - t
        print(f"[heygem12] detection done: {t_det:.1f}s ({t_det / n * 1000:.0f} ms/frame)")

        print("[heygem12] extracting audio features (WeNet) ...")
        t = time.perf_counter()
        windows = self.audio_frontend.windows_for_video(audio_path, num_frames=n, fps=fps)
        n = windows.shape[0]
        t_audio = time.perf_counter() - t
        print(f"[heygem12] audio features done: {t_audio:.1f}s")

        # -- original geometry: precompute landmarks + temporally smoothed
        # crop rects (the original's preprocess is non-causal: it sees the
        # whole video before synthesis, which is why its crops don't jitter).
        lm98: List[Optional[np.ndarray]] = [None] * n
        rects = np.full((n, 3), np.nan)
        if self.geom_mode == "orig":
            print("[heygem12] landmarks pass (PFLD-98, orig crop geometry) ...")
            t = time.perf_counter()
            for i in range(n):
                det = dets[i]
                if det is None:
                    continue
                try:
                    lm = self._landmarks68(frames[i], det[0])
                except Exception:
                    continue
                lm98[i] = lm
                rects[i] = self._crop_rect(lm)
            valid_idx = np.where(~np.isnan(rects[:, 0]))[0]
            for i in range(n):                      # nearest-valid fill
                if np.isnan(rects[i, 0]) and len(valid_idx):
                    rects[i] = rects[valid_idx[np.argmin(np.abs(valid_idx - i))]]
            sm_rects = self._smooth_rects(rects) if len(valid_idx) else rects
            print(f"[heygem12] landmarks done: {time.perf_counter() - t:.1f}s")

        m5 = 5   # crop border removed before resize (int(256/256*5))
        dump_dir = os.environ.get("HEYGEM12_DUMP_DIR")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            print(f"[heygem12] dumping DINet I/O (first 40 frames) to {dump_dir}")
        print(f"[heygem12] synthesizing (chaofen_before={bool(self.restorer and self.restore_source)}, "
              f"chaofen_after={bool(self.restorer and self.restore)}) ...")
        t_gen0 = time.perf_counter()
        out_frames: List[np.ndarray] = []
        for i in range(n):
            det = dets[i]
            frame = frames[i]
            if det is None:
                out_frames.append(frame)
                continue
            bbox, kps = det

            # -- chaofen_before: the original GFPGAN-restores the frame FIRST
            # (roi -> mtcnn_512 -> GFPGAN -> recover_from_reverse_matrix) and
            # crops DINet's input from the RESTORED frame (its dumps show the
            # crop carries the restoration's sharpness). The restore canvas is
            # driven by the SCRFD five points, not the PFLD geometry, so the
            # restored content (face fullness!) matches the original's.
            frame_bgr = np.ascontiguousarray(frame)
            src_img = frame_bgr
            if self.restorer is not None and self.restore_source:
                src_img = self._chaofen_frame(frame_bgr, bbox, kps)

            if self.geom_mode == "orig" and lm98[i] is not None:
                out_frames.append(self._run_frame_orig(
                    src_img, frame, lm98[i], sm_rects[i], windows[i],
                    dump_dir, i))
                if (i + 1) % 50 == 0 or i == n - 1:
                    el = time.perf_counter() - t_gen0
                    print(f"  frame {i + 1}/{n} | {el:.1f}s elapsed, {el / (i + 1) * 1000:.0f} ms/frame")
                continue

            # -- DINet crop: five points from our PFLD run on the ORIGINAL's
            # 112 framing (T112_REF, dump-fitted), warped to the ORIGINAL's
            # crop template (TEMPLATE_CROP, dump-fitted). See constants above.
            five = self._five_points(src_img, kps)
            M, _ = cv2.estimateAffinePartial2D(
                five, TEMPLATE_CROP, method=cv2.LMEDS)
            if M is None:
                out_frames.append(frame)
                continue
            rgb = cv2.warpAffine(src_img, M, (256, 256),
                                 borderMode=cv2.BORDER_REPLICATE)
            rgb = rgb[:, :, ::-1].astype(np.float32) / 255.0      # RGB [0,1]

            src_t = self._to_tensor(rgb * self.input_mask)
            ref_t = self._to_tensor(rgb)
            aud_t = torch.from_numpy(windows[i]).permute(1, 0).unsqueeze(0).float().to(self.device)
            pred = self.model(src_t, ref_t, aud_t)
            fake = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            if dump_dir and i < 40:
                np.save(os.path.join(dump_dir, "arg0_%03d.npy" % i),
                        src_t.squeeze(0).cpu().numpy())
                np.save(os.path.join(dump_dir, "arg1_%03d.npy" % i),
                        ref_t.squeeze(0).cpu().numpy())
                np.save(os.path.join(dump_dir, "arg2_%03d.npy" % i),
                        aud_t.squeeze(0).cpu().numpy())
                np.save(os.path.join(dump_dir, "out_%03d.npy" % i), fake)

            fuse_res = fake * self.fuse_mask + (1.0 - self.fuse_mask) * rgb

            # -- composite (original semantics): out = src + oval*(render-src).
            # The delta is exactly 0 outside the fuse feather, so non-face
            # pixels stay bit-identical to the input frame -- a replacement
            # composite (warped_fuse*M + frame*(1-M)) would instead paste
            # double-resampled background there and shimmer as M jitters.
            delta = (fuse_res - rgb).astype(np.float32)[:, :, ::-1]   # BGR
            h, w = frame.shape[:2]
            inv = cv2.invertAffineTransform(M)
            warped_delta = cv2.warpAffine(delta * 255.0, inv, (w, h))
            mask_frame = cv2.warpAffine(self._face_fg_crop, inv, (w, h))[:, :, None]
            out = frame.astype(np.float32) + warped_delta * mask_frame
            out_frames.append(np.clip(out, 0, 255).astype(np.uint8))

            if (i + 1) % 50 == 0 or i == n - 1:
                el = time.perf_counter() - t_gen0
                print(f"  frame {i + 1}/{n} | {el:.1f}s elapsed, {el / (i + 1) * 1000:.0f} ms/frame")
        t_gen = time.perf_counter() - t_gen0
        print(f"[heygem12] synthesis done: {t_gen:.1f}s ({t_gen / n * 1000:.0f} ms/frame)")

        t = time.perf_counter()
        ret = self._write_video(out_frames, fps, audio_path, output_path)
        print(f"[heygem12] write video done: {time.perf_counter() - t:.1f}s | "
              f"total {time.perf_counter() - t0:.1f}s")
        return ret

    def _run_frame_orig(self, src_img: np.ndarray, frame: np.ndarray,
                        lm: np.ndarray, rect: np.ndarray,
                        audio_win: np.ndarray,
                        dump_dir: Optional[str], i: int) -> np.ndarray:
        """One frame with the original's exact crop/paste geometry.

        rect crop (x1,y1,side, temporally smoothed) -> resize 266 INTER_CUBIC
        -> DINet input = crop266[5:261] (the original's [5:-5] slice) -> DINet
        -> fuse -> put back into the 266 canvas -> resize to side -> paste
        into the frame through the landmark hull+ellipse mask.
        """
        x1, y1, side = int(round(rect[0])), int(round(rect[1])), int(round(rect[2]))
        side = max(side, 32)
        h, w = frame.shape[:2]
        pad = side
        fp = cv2.copyMakeBorder(src_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = fp[y1 + pad:y1 + side + pad, x1 + pad:x1 + side + pad]
        crop266 = cv2.resize(crop, (266, 266), interpolation=cv2.INTER_CUBIC)
        lm266 = (lm - np.array([x1, y1], np.float32)) * (266.0 / side)

        rgb = crop266[5:261, 5:261][:, :, ::-1].astype(np.float32) / 255.0
        src_t = self._to_tensor(rgb * self.input_mask)
        ref_t = self._to_tensor(rgb)
        aud_t = torch.from_numpy(audio_win).permute(1, 0).unsqueeze(0).float().to(self.device)
        pred = self.model(src_t, ref_t, aud_t)
        fake = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if dump_dir and i < 40:
            np.save(os.path.join(dump_dir, "arg0_%03d.npy" % i),
                    src_t.squeeze(0).cpu().numpy())
            np.save(os.path.join(dump_dir, "arg1_%03d.npy" % i),
                    ref_t.squeeze(0).cpu().numpy())
            np.save(os.path.join(dump_dir, "arg2_%03d.npy" % i),
                    aud_t.squeeze(0).cpu().numpy())
            np.save(os.path.join(dump_dir, "out_%03d.npy" % i), fake)

        fuse_res = fake * self.fuse_mask + (1.0 - self.fuse_mask) * rgb

        rendered266 = crop266.astype(np.float32)
        rendered266[5:261, 5:261] = fuse_res[:, :, ::-1] * 255.0   # RGB->BGR
        rendered = cv2.resize(rendered266, (side, side), interpolation=cv2.INTER_LINEAR)

        mask266 = self._paste_mask266(lm266)
        mask = cv2.resize(mask266, (side, side), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[:, :, None]

        out = frame.copy()
        vx1, vy1 = max(0, x1), max(0, y1)
        vx2, vy2 = min(w, x1 + side), min(h, y1 + side)
        if vx2 > vx1 and vy2 > vy1:
            rx1, ry1 = vx1 - x1, vy1 - y1
            roi_m = mask[ry1:ry1 + (vy2 - vy1), rx1:rx1 + (vx2 - vx1)]
            roi_new = rendered[ry1:ry1 + (vy2 - vy1), rx1:rx1 + (vx2 - vx1)]
            roi_old = frame[vy1:vy2, vx1:vx2].astype(np.float32)
            out[vy1:vy2, vx1:vx2] = np.clip(
                roi_old * (1.0 - roi_m) + roi_new * roi_m, 0, 255).astype(np.uint8)
        return out

    # ----------------------------------------------------------------- pieces
    @staticmethod
    def _normalize_inputs(video_path: str, audio_path: str):
        """Match the original's format step: re-encode video to h264 crf 15
        and audio to 16 kHz mono pcm. Set HEYGEM12_SKIP_FORMAT=1 to skip."""
        if os.environ.get("HEYGEM12_SKIP_FORMAT"):
            return video_path, audio_path
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return video_path, audio_path
        workdir = tempfile.mkdtemp(prefix="heygem12_fmt_")
        vout = os.path.join(workdir, "format.mp4")
        aout = os.path.join(workdir, "format.wav")
        r1 = subprocess.call([ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
                              "-c:v", "libx264", "-crf", "15", "-an", vout])
        r2 = subprocess.call([ffmpeg, "-y", "-loglevel", "error", "-i", audio_path,
                              "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", aout])
        if r1 == 0 and os.path.exists(vout):
            video_path = vout
        if r2 == 0 and os.path.exists(aout):
            audio_path = aout
        return video_path, audio_path

    @staticmethod
    def _read_frames(video_path: str, max_frames: Optional[int]):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        if not frames:
            raise RuntimeError(f"cannot read video: {video_path}")
        return frames, fps

    @staticmethod
    def _write_video(frames: List[np.ndarray], fps: float,
                     audio_path: str, output_path: str) -> str:
        h, w = frames[0].shape[:2]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.call([
                ffmpeg, "-y", "-loglevel", "error",
                "-i", tmp_video, "-i", audio_path,
                "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                "-movflags", "+faststart",
                output_path,
            ])
        except Exception:
            import shutil
            shutil.move(tmp_video, output_path)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
        return output_path

    def _landmarks98(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """PFLD-98 landmarks (frame coords) on the original's 112 framing.

        The original feeds its PFLD a 112 crop warped by fit(scrfd_kps ->
        T112_REF) (recovered from its runtime dumps). Our PFLD on that same
        framing reproduces the original's dumped 98-pt landmarks to ~1.3 px;
        the points are mapped back to frame coordinates here.
        """
        m, _ = cv2.estimateAffinePartial2D(
            np.asarray(kps, np.float32), T112_REF, method=cv2.LMEDS)
        if m is None:
            raise RuntimeError("failed to estimate 112 alignment")
        aligned = cv2.warpAffine(img_bgr, m, (112, 112))
        pts = self.pfld._run(aligned)                          # (98,2) in 112
        hom = np.concatenate(
            [pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)
        return (hom @ cv2.invertAffineTransform(m).T).astype(np.float32)

    def _five_points(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return self._landmarks98(img_bgr, kps)[list(self.five_idx)]

    # ------------------------------------------------- original crop geometry
    _C68_IDX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                33, 34, 35, 36, 37, 42, 43, 44, 45, 46,
                51, 52, 53, 54, 55, 56, 57, 58, 59]

    @staticmethod
    def convert98to68(pts98: np.ndarray) -> np.ndarray:
        """The original's exact PFLD-98 -> 68 conversion (recovered by
        wrapping face_detect.convert98to68 at runtime and solving its
        input/output pairs; eye points are 2-point means):
        jaw 0-16 = 98[0,2,...,32]; brows = 98[33..37] + 98[42..46];
        nose = 98[51..59]; eyes: 60, avg(60,62), avg(62,64), 64, avg(64,66),
        avg(60,66) | 68, avg(68,70), avg(70,72), 72, avg(72,74), avg(68,74);
        mouth = 98[76..95]."""
        p = np.asarray(pts98, np.float64)
        out = [p[i] for i in LipSyncPipeline._C68_IDX]
        out += [p[60], (p[60] + p[62]) / 2, (p[62] + p[64]) / 2, p[64],
                (p[64] + p[66]) / 2, (p[60] + p[66]) / 2]
        out += [p[68], (p[68] + p[70]) / 2, (p[70] + p[72]) / 2, p[72],
                (p[72] + p[74]) / 2, (p[68] + p[74]) / 2]
        out += [p[i] for i in range(76, 96)]
        return np.array(out, np.float32)

    def _landmarks68(self, img_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """The original's landmark chain (runtime-verified):
        SCRFD bbox (x1,y1,x2,y2) -> crop [x1-0.1w, y1, x2+0.1w, y2+0.1h]
        (10% side/bottom margin, top unchanged; template-match verified)
        -> resize 112 anisotropic -> PFLD-98 -> convert98to68 -> frame coords.
        """
        h, w = img_bgr.shape[:2]
        bx1, by1, bx2, by2 = [float(v) for v in bbox[:4]]
        bw, bh = bx2 - bx1, by2 - by1
        cx1 = int(bx1 - 0.1 * bw)
        cy1 = int(by1)
        cx2 = int(bx2 + 0.1 * bw)
        cy2 = int(by2 + 0.1 * bh)
        crop = img_bgr[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
        pad = (max(0, -cy1), max(0, cy2 - h), max(0, -cx1), max(0, cx2 - w))
        if any(pad):
            crop = cv2.copyMakeBorder(crop, pad[0], pad[1], pad[2], pad[3],
                                      cv2.BORDER_REPLICATE)
        inp = cv2.resize(crop, (112, 112))
        pts112 = self.pfld._run(inp)                       # (98,2) in 112 px
        scale = np.array([cx2 - cx1, cy2 - cy1], np.float32) / 112.0
        pts98 = pts112 * scale + np.array([cx1, cy1], np.float32)
        return self.convert98to68(pts98)

    @staticmethod
    def _crop_rect(lm: np.ndarray) -> Tuple[int, int, int]:
        """The original's loc_crop_face formula, verbatim (decompiled from
        preprocess_audio_and_3dmm.so, template-match verified (342,556,437)):
            x, y, w, h = cv2.boundingRect(landmarks68)
            x1 = x_c - 0.75w ; y1 = y - 0.15w ; side = 1.5w (square)
        """
        x, y, w, _h = cv2.boundingRect(lm.astype(np.int32))
        xc = x + w / 2.0
        x1 = int(xc - w * 0.75)
        y1 = int(y - w * 0.15)
        side = int(1.5 * w)
        return x1, y1, side

    @staticmethod
    def _smooth_rects(rects: np.ndarray) -> np.ndarray:
        """The original's smooth_ (preprocess_audio_and_3dmm): centered
        window-5 moving average with symmetric boundary, then a de-spike:
        keep the RAW value where it deviates more than 12 px from the
        smoothed one (genuine fast motion passes through), use the smoothed
        value otherwise (sub-pixel detection jitter is removed)."""
        if len(rects) < 3:
            return rects
        rp = np.pad(rects.astype(np.float64), ((2, 2), (0, 0)), mode="symmetric")
        k = np.ones(5) / 5.0
        sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, rp)
        return np.where(np.abs(rects - sm) > 12.0, rects, sm)

    def _paste_mask266(self, lm266: np.ndarray) -> np.ndarray:
        """The original's DigitalHumanModel.get_face_mask on the 266 canvas,
        black-box identified by calling the original function directly
        (IoU 1.0 against its runtime dump):
          wanted = landmarks68[2:15] (jaw, temples excluded) + landmarks68[29]
          (nose tip); mask = fillConvexPoly(convexHull(wanted), 255)
          + bottom-half ellipse 0-180deg:
              cx = jaw wanted center x, cy = chin(8).y - 34, axes = jaw width * 0.35
          + dilate 6x6 x1.
        """
        wanted = np.concatenate([lm266[2:15], lm266[29:30]], axis=0)
        hull = cv2.convexHull(wanted.astype(np.int32))
        mask = np.zeros((266, 266), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        jx = wanted[:, 0]
        cx = int(round((jx.min() + jx.max()) / 2.0))
        chin_y = float(lm266[8, 1])
        cy = int(round(chin_y - 34.0))
        ax = int(round((jx.max() - jx.min()) * 0.35))
        cv2.ellipse(mask, (cx, cy), (ax, ax), 0, 0, 180, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask.astype(np.float32) / 255.0

    def _chaofen_frame(self, frame_bgr: np.ndarray, bbox: np.ndarray,
                       kps: np.ndarray) -> np.ndarray:
        """The original's chaofen_before: GFPGAN-restore the frame and return
        the recovered frame (uint8 BGR). Exact cv2box chain:

            roi        = frame_rgb[by1:by2, bx1:bx2]   (plain bbox, no margin)
            roi_kpss   = kps - [bx1, by1]
            M512       = estimateAffinePartial2D(roi_kpss, MTCNN_512_TEMPLATE)
            face512    = warpAffine(roi, M512, 512)
            restored   = GFPGAN(face512 BGR)           ([0,1] float RGB)
            recovered  = recover_from_reverse_matrix(restored, roi_u8, M512,
                         common_face_mask((roi_w, roi_h)))
            frame[roi] = recovered
        """
        img_rgb = frame_bgr[:, :, ::-1]
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        roi = img_rgb[y1:y2, x1:x2]
        roi_kpss = np.asarray(kps, np.float32) - np.array([x1, y1], np.float32)
        face512_rgb, M512 = norm_crop(roi, roi_kpss, 512)
        face512_bgr = np.ascontiguousarray(face512_rgb[:, :, ::-1])
        restored01 = self._chaofen(face512_bgr)             # RGB [0,1]
        fg_mask = common_face_mask((roi.shape[1], roi.shape[0]))
        recovered_roi = recover_from_reverse_matrix(
            restored01, roi, M512, fg_mask)                 # uint8 RGB
        out = frame_bgr.copy()
        out[y1:y2, x1:x2] = recovered_roi[:, :, ::-1]
        return out

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1) \
            .unsqueeze(0).float().to(self.device)
