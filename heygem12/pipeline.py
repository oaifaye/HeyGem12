"""End-to-end lip-sync pipeline: video + audio -> lip-synced video.

Geometry follows the checkpoint's own metadata (see face/align256.py): faces are
aligned to a canonical 256x256 crop, zeroed by ``input_mask``, rendered at
256x256 and fused back with the feathered ``fuse_mask``.
"""

import os
import subprocess
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
import torch

from heygem12.audio.frontend import WeNetFeatureExtractor
from heygem12.face.align256 import (EYE_CENTER, TEMPLATE_K, Align256,
                                    default_fuse_strength, load_meta)
from heygem12.face.detection import YuNetFaceDetector
from heygem12.face.landmarks import PFLDLandmarker
from heygem12.models import load_dinet_hd


class LipSyncPipeline:
    def __init__(
        self,
        checkpoints_dir: str,
        device: str = "cpu",
        ref_frame: str = "middle",  # "middle" | "first" -- UNUSED: the original uses
        # the SAME frame's crop (unmasked) as the DINet reference, built per frame.
        restore: bool = False,      # chaofen_after: GFPGAN on the RENDER (original does NOT do this)
        # chaofen_before: the original GFPGAN-restores the SOURCE frame and crops
        # from the restored frame (round-7 server dumps: its DINet arg0/ref crops
        # carry the restoration's band lapvar ~89 vs ~35 raw; the restored frame
        # never reaches the output -- the composite pastes the raw render into
        # the RAW frame through a face-oval mask).
        restore_source: bool = True,
        fuse_mode: str = "oval",    # "oval" = original's face-oval hard paste | "fuse_mask" = old scheme
        kps_stabilize: bool = True, # temporal smoothing of the 5-point sets (kills 1-frame detect jitter)
        kps_vcalib: bool = True,   # per-video vertical semantics calibration: PFLD's nose/mouth sit
                                   # ~20% closer to the eye line than the SCRFD five points the original
                                   # aligns with (measured demo1: PFLD mouthY/eye 0.80 vs SCRFD 1.0),
                                   # stretching the served crop's lower face ~18px too tall in the
                                   # 256 canvas -> the network renders a fat jaw/"double chin".
                                   # Stretch points 2-4 about the eye line by the per-video median
                                   # SCRFD/PFLD mouth-height ratio so the crop geometry matches.
        render_ema: float = 0.0,    # DISABLED by default: temporal EMA on the DINet render ghosts lip
                                    # shapes wherever the repaint is partial (the decoder redraws the
                                    # whole masked region at phoneme boundaries, so blend<->raw mode
                                    # switching reads as mouth shimmer -- measured 2026-09-30). The
                                    # original does NO temporal render processing; keep 0.
        ):
        self.device = device
        dinet_path = os.path.join(checkpoints_dir, "dinet_v1_20240131.pth")
        print("[heygem12] loading DINet_HD ...")
        self.model = load_dinet_hd(dinet_path, device=device)
        meta = load_meta(dinet_path)
        self.aligner_K = TEMPLATE_K
        self.aligner_center = EYE_CENTER
        self.aligner = Align256(meta, K=self.aligner_K, center=self.aligner_center,
                                fuse_strength=default_fuse_strength(meta))
        print("[heygem12] dinet_v1 input %s | input_mask zero frac %.2f | fuse wh=%.2f"
              % (meta.get("model_input_size"),
                 float((self.aligner.input_mask[..., 0] == 0).mean()),
                 self.aligner.fuse_strength))
        print("[heygem12] loading WeNet frontend ...")
        self.audio_frontend = WeNetFeatureExtractor(
            os.path.join(checkpoints_dir, "wenetmodel.pt"), device=device)
        self.detector = YuNetFaceDetector(
            os.path.join(checkpoints_dir, "face_detection_yunet_2023mar.onnx"))
        self.landmarker = PFLDLandmarker(
            os.path.join(checkpoints_dir, "pfpld_robust_sim_bs1_8003.onnx"))
        self.kps_vcalib = kps_vcalib
        self._scrfd = None
        self._scrfd_path = os.path.join(checkpoints_dir, "scrfd_500m_bnkps_shape640x640.onnx")
        self.ref_frame = ref_frame
        self.restore = restore
        self.restore_source = restore_source
        self.fuse_mode = fuse_mode
        self.kps_stabilize = kps_stabilize
        self.render_ema = float(render_ema)
        self.restorer = None
        if restore or restore_source:
            try:
                import onnxruntime as ort

                gfp = os.path.join(checkpoints_dir, "GFPGANv1.4.onnx")
                if os.path.exists(gfp):
                    from heygem12.ortutil import make_session, ort_providers

                    self.restorer = make_session(gfp)
                    self._init_restore()
                    print(f"[heygem12] GFPGAN face restore: enabled ({ort_providers(self.restorer)})")
                else:
                    print(f"[heygem12] GFPGAN onnx not found at {gfp}, restore disabled")
            except Exception as e:  # pragma: no cover
                print(f"[heygem12] GFPGAN unavailable ({e}), restore disabled")

    # ---------------------------------------------------------------- helpers
    def _keypoints_for_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """5-point keypoints per frame, for the 256x256 alignment.

        YuNet detects the face and gives a rough 5-point set, which is used to
        warp the 112x112 crop that PFLD needs; the PFLD-98 output is then
        reduced to the arcface-order 5 points actually used for alignment.
        YuNet's own keypoints are NOT accurate enough: on a near-static talking
        head their eye-line tilt swings -0.4..-4.1 deg between frames (PFLD stays
        within 0.6 deg of level), which misaligns the crop by a few degrees and
        makes it wobble over time.
        """
        out: List[np.ndarray] = []
        ratios: List[float] = []
        last: Optional[np.ndarray] = None
        for i, frame in enumerate(frames):
            det = self.detector.detect(frame)
            if det is None or det["kps"] is None:
                if last is None:
                    raise RuntimeError(f"no face detected in frame {i}")
                out.append(last)
                continue
            rough = np.asarray(det["kps"], dtype=np.float32)
            pts98 = self.landmarker.predict_aligned(frame, rough)
            five = self.landmarker.five_points(pts98)
            # fall back to the detector if PFLD produced something implausible
            if not np.isfinite(five).all() or float(np.ptp(five[:, 0])) < 1.0:
                five = rough
            # Semantic eye-scale calibration: DINet (and the original pipeline's
            # DINet crop) was served DETECTOR five-point crops. PFLD pupil
            # distances drift per person relative to detector eye semantics
            # (measured: woman clip detector/PFLD eye ratio = 0.93 -> our PFLD
            # crops put the face ~13% too small -> the DINet mouth response was
            # damped to ~half amplitude; with the ratio applied it recovers to
            # the original's). Estimate the ratio per frame, lock the MEDIAN per
            # video (per-person constant, robust to per-frame detector jitter),
            # and pull the PFLD eye pair inward/outward about its midpoint --
            # eye-line tilt (PFLD, stable) is preserved.
            de = float(np.linalg.norm(rough[0] - rough[1]))
            pe = float(np.linalg.norm(five[0] - five[1]))
            if pe > 1.0:
                ratios.append(de / pe)
            last = five.astype(np.float32)
            out.append(last)
        if ratios:
            R = float(np.median(ratios))
            if abs(R - 1.0) > 0.02:
                seen = set()
                for five in out:
                    if id(five) in seen:
                        continue
                    seen.add(id(five))
                    em = (five[0] + five[1]) / 2.0
                    five[0] = em + (five[0] - em) * R
                    five[1] = em + (five[1] - em) * R
        if self.kps_stabilize:
            out = self._stabilize_keypoints(out)
        if self.kps_vcalib:
            out = self._vcalib_keypoints(frames, out)
        return out

    def _vcalib_keypoints(self, frames, out):
        """Stretch nose/mouth points about the eye line to match SCRFD semantics.

        PFLD's reduced five points place the nose tip / mouth corners
        vertically closer to the eye line than the SCRFD five points the
        original pipeline aligns with; the resulting similarity fit serves
        DINet a crop whose lower face is too tall (chin ~18 canvas px lower
        than the original's, measured on demo1 dumps), which the network
        renders as a heavy jaw / "double chin". The per-video median ratio
        SCRFD/PFLD of (mouth_y - eye_y)/eye_dist is applied to points 2..4.
        """
        from .face.detector import SCRFD
        if self._scrfd is None:
            if not os.path.exists(self._scrfd_path):
                return out
            self._scrfd = SCRFD(self._scrfd_path)
        def ratio(pt):
            eye = float(np.linalg.norm(pt[0] - pt[1]))
            eyey = (float(pt[0][1]) + float(pt[1][1])) / 2.0
            my = (float(pt[3][1]) + float(pt[4][1])) / 2.0
            return (my - eyey) / eye if eye > 1.0 else None
        ks = []
        step = max(1, len(frames) // 50)
        for i in range(0, len(frames), step):
            try:
                d = self._scrfd.detect(frames[i])
            except Exception:
                continue
            if d is None or d.get("kps") is None:
                continue
            rs = ratio(np.asarray(d["kps"], np.float32))
            ro = ratio(out[i])
            if rs and ro and 0.3 < ro and 0.8 < rs / ro < 1.35:
                ks.append(rs / ro)
        if not ks:
            return out
        k = float(np.median(ks))
        # Small corrections are already absorbed by the SIFT-fitted TEMPLATE_256
        # (fitted on the woman clip, whose SIFT transform carries a ~21px vertical
        # offset). Applying them again double-corrects. Measured with matched
        # audio (woman_cn f52): k=1.055 moves the under-chin dip by only ~1.6/255
        # (negligible), while demo1's k=1.185 fixed an 18px chin-row error.
        # Only apply clearly person-specific drift.
        if abs(k - 1.0) < 0.08:
            return out
        seen = set()
        for five in out:
            if id(five) in seen:
                continue
            seen.add(id(five))
            eyey = (float(five[0][1]) + float(five[1][1])) / 2.0
            for j in (2, 3, 4):
                five[j][1] = eyey + (float(five[j][1]) - eyey) * k
        print("[heygem12] vertical kps calibration: k=%.3f (n=%d frames)" % (k, len(ks)))
        return out

    @staticmethod
    def _stabilize_keypoints(kps_list: List[np.ndarray],
                             alpha_min: float = 0.12, m0: float = 2.0) -> List[np.ndarray]:
        """Temporal stabilization of the per-frame 5-point sets.

        YuNet/PFLD occasionally return a one-frame outlier: measured on the
        test set, isolated consecutive-frame jumps of 7-11 px occur where the
        neighboring frames barely move (median motion ~2.3 px). A single bad
        frame shifts the similarity warp, so the DINet crop and the pasted
        face-oval both jump for that one frame and read as a face/mouth
        flicker. Pipeline: (1) bridge isolated 2-frame excursions (detector
        glitches, not real motion) by interpolation; (2) median-3 kills
        single-frame spikes; (3) a HARD-switch EMA -- strong smoothing when
        nearly still, exact raw values when moving (see comment at the loop).
        """
        k = np.stack(kps_list)                       # (N,5,2)
        # Offline spike-pair rejection: median-3 cannot kill a bad detection
        # that lasts TWO consecutive frames. Isolated 2-frame excursions that
        # are calm immediately before and after are almost always detector
        # glitches, not real motion (real head motion stays elevated for
        # several frames) -- bridge them by interpolation.
        if len(k) >= 6:
            d = np.abs(np.diff(k, axis=0)).max(axis=(1, 2))   # d[j]: jump j -> j+1
            for j in range(2, len(d) - 3):
                # bad frames j, j+1: elevated jump in (d[j-1]), between (d[j])
                # and out (d[j+1]), calm on both sides
                if (d[j - 2] < 3.0 and d[j - 1] > 5.0 and d[j] > 5.0
                        and d[j + 1] > 5.0 and d[j + 2] < 3.0):
                    t1, t2 = 1.0 / 3.0, 2.0 / 3.0
                    k[j] = (1.0 - t1) * k[j - 1] + t1 * k[j + 2]
                    k[j + 1] = (1.0 - t2) * k[j - 1] + t2 * k[j + 2]
        med = k.copy()
        if len(k) >= 3:
            med[1:-1] = np.median(np.stack([k[:-2], k[1:-1], k[2:]]), axis=0)
        # Motion-adaptive EMA: near-still frames blend at the floor factor (damps
        # the residual 2-3 px detect jitter), but the blend factor rises with the
        # measured motion so sustained fast movement (head turns, the f16/f104
        # style gestures) is tracked within ~1.5 px instead of lagging 6-8 px
        # (the fixed-alpha EMA's failure mode, measured on this test set).
        out = np.empty_like(k)
        out[0] = med[0]
        # HARD switch, not a ramp: partial blending creates a motion-dependent
        # lag, so the effective face velocity becomes non-uniform and reads as
        # a lower-face stutter at motion onsets/stops (user-reported at 2.24 s
        # on the woman clip; her typical small motion is ~2.4 px/frame). With
        # a switch there is ZERO lag while moving (out == raw) and ZERO
        # pull-back when stopping (the state never deviates during raw
        # frames), so transitions are as smooth as the raw detections.
        for i in range(1, len(k)):
            m = float(np.abs(med[i] - med[i - 1]).max())   # current motion (px)
            a = 1.0 if m >= m0 else alpha_min
            out[i] = a * med[i] + (1.0 - a) * out[i - 1]
        # Scale anchoring: the detectors need a few frames to settle (measured:
        # PFLD eye distance drifts ~2% over the first ~8 frames of a clip while
        # the subject is nearly still), and an EMA only delays such a ramp, so
        # the pasted face visibly grows/shrinks at the start of the video.
        # Anchor each frame's similarity scale to a running median (+-20 fr):
        # real face scale in static-camera footage is constant (measured YuNet
        # bbox spread ~+-1% across whole clips), so this removes the settling
        # without affecting genuine motion (translation/rotation untouched).
        eye = np.linalg.norm(out[:, 0] - out[:, 1], axis=1)      # (N,)
        half = 20
        for i in range(len(out)):
            lo, hi = max(0, i - half), min(len(out), i + half + 1)
            target = float(np.median(eye[lo:hi]))
            cur = float(eye[i])
            if abs(cur - target) > 1e-3:
                c = (out[i][0] + out[i][1]) / 2.0                # eye midpoint
                out[i] = c + (out[i] - c) * (target / cur)
        return [row for row in out]

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
        frames, fps = self._read_frames(video_path, max_frames)
        n = len(frames)
        print(f"[heygem12] {n} frames @ {fps:.2f} fps | {frames[0].shape[1]}x{frames[0].shape[0]}")

        print("[heygem12] detecting faces ...")
        t = time.perf_counter()
        kps_list = self._keypoints_for_frames(frames)
        t_det = time.perf_counter() - t
        print(f"[heygem12] detection done: {t_det:.1f}s ({t_det / n * 1000:.0f} ms/frame)")

        print("[heygem12] extracting audio features (WeNet) ...")
        t = time.perf_counter()
        windows = self.audio_frontend.windows_for_video(audio_path, num_frames=n, fps=fps)
        n = windows.shape[0]   # audio-limited runs stop at the original's output length
        t_audio = time.perf_counter() - t
        print(f"[heygem12] audio features done: {t_audio:.1f}s")

        # DINetV1's reference is NOT a separate frame: the original feeds the SAME
        # frame's crop unmasked as ref (hook dump, arg1 == arg0 outside input_mask
        # bit-exact). The network transfers real mouth texture (teeth!) from it;
        # a fixed mid-video ref makes open-mouth frames degenerate to a lip-colour
        # blob. Built per frame inside the loop below.

        print(f"[heygem12] synthesizing ({self.aligner.input_size}x{self.aligner.input_size} "
              f"aligned crop, audio window T={windows.shape[1]}, "
              f"chaofen_before={bool(self.restorer and self.restore_source)}, "
              f"chaofen_after={bool(self.restorer and self.restore)}, "
              f"fuse={self.fuse_mode}) ...")
        t_gen0 = time.perf_counter()
        out_frames: List[np.ndarray] = []
        acc = {"align": 0.0, "chaofen": 0.0, "dinet": 0.0, "fuse": 0.0}
        ema_prev: Optional[np.ndarray] = None
        prev_raw: Optional[np.ndarray] = None
        for i in range(n):
            t = time.perf_counter()
            rgb, M = self.aligner.align(frames[i], kps_list[i])
            acc["align"] += time.perf_counter() - t
            if rgb is None:
                out_frames.append(frames[i])
                continue
            if self.restorer is not None and self.restore_source:
                t = time.perf_counter()
                rgb = self.restore_crop(frames[i], kps_list[i], M, rgb)
                acc["chaofen"] += time.perf_counter() - t
            src_t = self._to_tensor(self.aligner.mask_source(rgb))
            ref_t = self._to_tensor(rgb)      # same frame, unmasked (the original's scheme)
            aud_t = torch.from_numpy(windows[i]).permute(1, 0).unsqueeze(0).float().to(self.device)
            t = time.perf_counter()
            pred = self.model(src_t, ref_t, aud_t)          # (1,3,256,256) in [0,1]
            rendered = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            acc["dinet"] += time.perf_counter() - t
            # The decoder regenerates the whole masked region (cheeks/jaw included)
            # per frame, so consecutive renders wobble in face shape/brightness even
            # with near-identical inputs (BN running-stats inconsistency, see P3) --
            # perceived as the face going "fat-thin" between frames. Blend with the
            # temporally-smoothed previous render, ADAPTIVELY: when the mouth region
            # barely changed (near-still frame) blend strongly; when it changed a
            # lot (real speech motion) pass the render through raw, otherwise the
            # blended lip shapes ghost (measured: still-frame diffs ~0.006-0.06,
            # fast-speech diffs ~0.35 -> threshold band 0.08..0.2).
            raw_render = rendered
            if self.render_ema > 0.0 and ema_prev is not None:
                d = float((np.abs(raw_render - prev_raw) * self._mouth_weight()).sum()
                          / self._mouth_weight_sum)
                lo, hi = 0.08, 0.2
                s = 0.0 if d >= hi else (1.0 if d <= lo else (d - lo) / (hi - lo))
                # s=1 on a near-still frame -> blend at the floor factor; s=0 when
                # the mouth really moved -> pass the render through raw.
                a = 1.0 - (1.0 - self.render_ema) * s
                rendered = a * raw_render + (1.0 - a) * ema_prev
            prev_raw = raw_render
            ema_prev = rendered
            if self.restorer is not None and self.restore:
                rendered = self._restore_render(frames[i], kps_list[i], M, rendered)
            t = time.perf_counter()
            if self.fuse_mode == "oval":
                out_frame = self.aligner.paste_back_oval(frames[i], M, rendered)
            else:
                out_frame = self.aligner.paste_back(frames[i], M, rendered)
            acc["fuse"] += time.perf_counter() - t
            out_frames.append(out_frame)
            if (i + 1) % 10 == 0 or i == n - 1:
                el = time.perf_counter() - t_gen0
                print(f"  frame {i + 1}/{n} | {el:.1f}s elapsed, {el / (i + 1) * 1000:.0f} ms/frame")
        t_gen = time.perf_counter() - t_gen0
        print(f"[heygem12] synthesis done: {t_gen:.1f}s ({t_gen / n * 1000:.0f} ms/frame) | "
              f"align {acc['align']:.1f}s | chaofen {acc['chaofen']:.1f}s | "
              f"dinet {acc['dinet']:.1f}s | fuse {acc['fuse']:.1f}s")

        t = time.perf_counter()
        ret = self._write_video(out_frames, fps, audio_path, output_path)
        print(f"[heygem12] write video done: {time.perf_counter() - t:.1f}s | "
              f"total {time.perf_counter() - t0:.1f}s")
        return ret

    # ----------------------------------------------------------------- pieces
    def _mouth_weight(self) -> np.ndarray:
        """Feathered mouth-region weight (256x256x1) for the adaptive render EMA.

        Built from the aligner's fitted template mouth corners (same centre as
        the fuse-mode mouth ellipse, but tighter: rx45/ry35) so the motion
        detector only watches the lips.
        """
        w = getattr(self, "_mouth_w", None)
        if w is None:
            cx = float((self.aligner.template[3][0] + self.aligner.template[4][0]) / 2)
            cy = float((self.aligner.template[3][1] + self.aligner.template[4][1]) / 2) + 8.0
            ell = np.zeros((self.aligner.input_size,) * 2, np.float32)
            cv2.ellipse(ell, (int(round(cx)), int(round(cy))), (45, 35), 0, 0, 360, 1.0, -1)
            w = cv2.GaussianBlur(ell, (0, 0), 8.0)[:, :, None]
            self._mouth_w = w
            self._mouth_weight_sum = float(w.sum())
        return w

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1) \
            .unsqueeze(0).float().to(self.device)

    # GFPGAN runs in the frame-aligned arcface-512 canvas, then the restored face is
    # mapped back into DINet's 256x256 crop space. For chaofen_before that means
    # warping the SOURCE frame up; for chaofen_after (_restore_render) it means
    # warping the RENDER up -- never the source, or the synthesized lips are lost.
    #
    # Restoring the 256 render directly is NOT equivalent: the similarity from our
    # template (eye row 70) to arcface-512 (eye row 206.8) carries a +66.8px
    # vertical shift at scale 2.0, so crop rows >222 and cols <16 fall outside the
    # 512 canvas and come back BLACK - exactly along fuse_mask's bottom/left edge,
    # producing a black rectangular border in the output. Aligning from the frame
    # instead never clips, and the residual uncovered sliver (crop rows >222) is
    # handled by an explicit validity mask that falls back to the raw render.
    #
    # Placement itself is measured (silent audio, so the ideal output == source;
    # source mouth laplacian var ~48.5):
    #     no restore        MAE 0.0557  corr 0.8938  lapvar   7.6   (too soft)
    #     restore on frame  MAE 0.0531  corr 0.8909  lapvar 192.6   (4x over-sharp)
     #     restore on crop   MAE 0.0511  corr 0.9002  lapvar  51.6   (matches source)

    def _init_restore(self):
        from heygem12.face.align256 import ARCFACE112

        self._ref512 = (ARCFACE112 * 4.0).astype(np.float32)

    def _gfpgan_512(self, frame: np.ndarray, five_kps: np.ndarray):
        """GFPGAN-restore the frame in arcface-512 space.

        Returns (restored512 uint8 BGR, M512) or (None, None) if unalignable.
        """
        M512, _ = cv2.estimateAffinePartial2D(
            np.asarray(five_kps, dtype=np.float32), self._ref512, method=cv2.LMEDS)
        if M512 is None:
            return None, None
        face = cv2.warpAffine(frame, M512, (512, 512))
        blob = face[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        out = self.restorer.run(None, {"input": blob})[0][0]
        restored = (np.clip(out.transpose(1, 2, 0), 0, 1) * 255.0)[:, :, ::-1].astype(np.uint8)
        return restored, M512

    @staticmethod
    def _compose(M_dst_from_frame: np.ndarray, M_frame_from_src: np.ndarray) -> np.ndarray:
        """Compose two 2x3 affinities into src-space -> dst-space."""
        a = np.vstack([M_dst_from_frame, [0.0, 0.0, 1.0]])
        b = np.vstack([M_frame_from_src, [0.0, 0.0, 1.0]])
        return (a @ b)[:2]

    def restore_crop(self, frame: np.ndarray, five_kps: np.ndarray,
                     M: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """``chaofen_before``: sharpen the source crop with GFPGAN.

        The original pipeline runs a per-frame super-resolution step before
        cropping (``trans_dh_service`` exports ``chaofen_src`` / ``chaofen_before``
        / ``blur_threshold``, and the deployment ships no SR model other than
        GFPGAN). It matters a lot: in the unmasked band of the crop -- which
        DINet never regenerates, so it is pure source content -- the original's
        crops carry laplacian variance 157 while a crop of the raw frames carries
        only 13-15. A blurry source crop therefore yields a blurry render no
        matter what, which is the "mouth looks smeared" symptom.

        One GFPGAN call is resampled into DINet's 256x256 space via
        ``A = M_256 . M_512^-1``; pixels the 512 canvas does not cover fall back
        to the raw aligned crop so nothing goes black.
        """
        restored512, M512 = self._gfpgan_512(frame, five_kps)
        if restored512 is None:
            return rgb
        A = self._compose(M, cv2.invertAffineTransform(M512))
        rgb512 = restored512[:, :, ::-1].astype(np.float32) / 255.0
        sharp = cv2.warpAffine(rgb512, A, (256, 256))
        valid = cv2.warpAffine(np.ones((512, 512), np.float32), A, (256, 256))[:, :, None]
        # GFPGAN hallucinates a mosaic sparkle inside the iris and a small dark
        # blob at the inner eye corner for some faces (measured on jie1: both
        # eyes, every frame). The eyes are above input_mask's first zero row,
        # i.e. pass-through content, so the artifact reaches the final frame.
        # Keep the raw content inside pupil-centred ellipses; everything else
        # (cheeks/forehead/jaw, which drives the render's sharpness) stays
        # restored.
        pupils = cv2.transform(np.asarray(five_kps, np.float32)[:2, None, :],
                               M).reshape(-1, 2)
        protect = np.zeros((256, 256), np.float32)
        for (px, py) in pupils:
            cv2.ellipse(protect, (int(round(float(px))), int(round(float(py)))),
                        (38, 28), 0, 0, 360, 1.0, -1)
        # GFPGAN also draws a dashed seam line + blotchy texture across the
        # neck on some faces (measured on the man_* batch, crop rows ~190-230):
        # the neck sits far outside its face prior and gets an unstable
        # "stitched" repaint. The restored crop is DINet's REFERENCE (arg1,
        # unmasked), so the network copies the seam into the render -- seen as
        # a flickering dark patch at the neck (the original's canvas alignment
        # does not produce it). The neck needs no super-resolution: keep raw
        # content there (rows ~186+, inside the face oval's lower part).
        cv2.ellipse(protect, (128, 226), (85, 40), 0, 0, 360, 1.0, -1)
        protect = cv2.GaussianBlur(protect, (0, 0), 4.0)[:, :, None]
        eff = valid * (1.0 - protect)
        return np.clip(rgb * (1.0 - eff) + sharp * eff, 0.0, 1.0)

    def _restore_render(self, frame: np.ndarray, five_kps: np.ndarray,
                        M: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """``chaofen_after``: GFPGAN-sharpen the RENDERED 256x256 crop.

        The DINet render has the right structure (crop-space correlation 0.95
        with ground truth) but limited high-frequency detail, so the restore adds
        back skin/lip texture. GFPGAN must run on the RENDER -- warping the
        source frame into the 512 canvas instead would paste the source's closed
        mouth over the synthesized lips and silently kill all lip motion.
        Pipeline: render (256, RGB float) -> warp up to the arcface-512 canvas
        (A^-1) -> GFPGAN -> warp back down (A). Anything the 512 canvas does not
        cover keeps the raw render, so no black edge can leak into the fusion.
        """
        M512, _ = cv2.estimateAffinePartial2D(
            np.asarray(five_kps, dtype=np.float32), self._ref512, method=cv2.LMEDS)
        if M512 is None:
            return rendered
        A = self._compose(M, cv2.invertAffineTransform(M512))   # 512 -> 256
        up = cv2.warpAffine(rendered.astype(np.float32),
                            cv2.invertAffineTransform(A), (512, 512))
        blob = np.ascontiguousarray(up.transpose(2, 0, 1)[None])
        out = self.restorer.run(None, {"input": blob})[0][0]
        restored = np.clip(out.transpose(1, 2, 0), 0, 1).astype(np.float32)
        sharp = cv2.warpAffine(restored, A, (256, 256))
        valid = cv2.warpAffine(np.ones((512, 512), np.float32), A, (256, 256))[:, :, None]
        return np.clip(rendered * (1.0 - valid) + sharp * valid, 0.0, 1.0)

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
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                "-movflags", "+faststart",   # moov atom first: web/README players start immediately
                output_path,
            ])
        except Exception:
            import shutil
            shutil.move(tmp_video, output_path)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
        return output_path
