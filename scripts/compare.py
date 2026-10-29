# -*- coding: utf-8 -*-
"""Batch side-by-side comparison video generator (original HeyGem vs ours).

Usage:
  python compare.py                # auto: scan assets/out for *_orig.mp4 partners
  python compare.py out.mp4 A.mp4 B.mp4 [C.mp4 D.mp4 ...]   # explicit pairs

Auto mode pairing (in assets/out):
  <stem>_orig.mp4  +  <stem>.mp4        ->  cmp_<stem>.mp4
  <stem>_orig.mp4  +  <stem>_ours.mp4   ->  cmp_<stem>.mp4

The side-by-side video is labelled ORIG | OURS, uses min(frame counts),
and muxes audio from the "ours" file (falls back to the orig file).
Requires ffmpeg on PATH for the final mux.

With --show-mask, a green contour marks the face-oval region that our
pipeline replaces (computed with the project's own detectors; the contour
is drawn on BOTH halves -- the original replaces the same region, its own
alignment may differ by a few px).
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

OUT_DIR = "/opt/projects/heygem12/assets/out"
CKPT_DIR = "/opt/projects/heygem12/checkpoints"
LABELS = ("ORIG", "OURS")
LABEL_COLOR = (0, 0, 255)   # red in BGR
MASK_COLOR = (0, 255, 0)    # green contour


def _mask_maker(checkpoints: str):
    """Build (aligner, kps_fn) reproducing the pipeline's crop geometry."""
    sys.path.insert(0, "/opt/projects/heygem12")
    from heygem12.face.align256 import (Align256, TEMPLATE_K, EYE_CENTER,
                                        default_fuse_strength, load_meta)
    from heygem12.face.detection import YuNetFaceDetector
    from heygem12.face.landmarks import PFLDLandmarker
    from heygem12.pipeline import LipSyncPipeline

    meta = load_meta(os.path.join(checkpoints, "dinet_v1_20240131.pth"))
    aligner = Align256(meta, K=TEMPLATE_K, center=EYE_CENTER,
                       fuse_strength=default_fuse_strength(meta))
    detector = YuNetFaceDetector(os.path.join(checkpoints, "face_detection_yunet_2023mar.onnx"))
    landmarker = PFLDLandmarker(os.path.join(checkpoints, "pfpld_robust_sim_bs1_8003.onnx"))

    def kps_for(frames):
        """Mirror LipSyncPipeline._keypoints_for_frames (YuNet -> PFLD 5pt ->
        per-video median eye-scale R -> temporal stabilization)."""
        out, ratios, last = [], [], None
        for i, frame in enumerate(frames):
            det = detector.detect(frame)
            if det is None or det["kps"] is None:
                if last is None:
                    raise RuntimeError("no face in frame %d" % i)
                out.append(last)
                continue
            rough = np.asarray(det["kps"], dtype=np.float32)
            pts98 = landmarker.predict_aligned(frame, rough)
            five = landmarker.five_points(pts98)
            if not np.isfinite(five).all() or float(np.ptp(five[:, 0])) < 1.0:
                five = rough
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
        return LipSyncPipeline._stabilize_keypoints(out)

    return aligner, kps_for


def _fit(frame: np.ndarray, size: tuple) -> np.ndarray:
    """Resize into size (w, h) keeping aspect, center-pad with black."""
    w, h = size
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if nw == w and nh == h:
        return r
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = r
    return canvas


def make_compare(orig_path: str, ours_path: str, out_path: str,
                 layout: str = "h", show_mask: bool = False,
                 checkpoints: str = CKPT_DIR) -> None:
    co, cu = cv2.VideoCapture(orig_path), cv2.VideoCapture(ours_path)
    if not co.isOpened() or not cu.isOpened():
        raise RuntimeError("cannot open %s / %s" % (orig_path, ours_path))
    fo, fu = co.get(cv2.CAP_PROP_FPS), cu.get(cv2.CAP_PROP_FPS)
    fps = fo if fo > 1 else (fu if fu > 1 else 25.0)
    n = min(int(co.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cu.get(cv2.CAP_PROP_FRAME_COUNT)))

    aligner = kps_fn = None
    if show_mask:
        # pre-pass: keypoints per frame from the OURS video (identical source
        # frames to orig before the paste) -> crop transform per frame
        aligner, kps_fn = _mask_maker(checkpoints)
        cap = cv2.VideoCapture(ours_path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        all_kps = kps_fn(frames[:n] if n > 0 else frames)
        del frames
        co.release()
        co = cv2.VideoCapture(orig_path)

    ok_o, fr_o = co.read()
    ok_u, fr_u = cu.read()
    if not ok_o or not ok_u:
        raise RuntimeError("empty video %s / %s" % (orig_path, ours_path))
    # common cell size: half of the orig frame (h) or full (v)
    ch, cw = fr_o.shape[:2]
    cell = (cw // 2, ch) if layout == "h" else (cw, ch // 2)
    W, H = (cell[0] * 2, cell[1]) if layout == "h" else (cell[0], cell[1] * 2)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    vw = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    font_scale = max(0.8, cell[0] / 700.0)
    thickness = max(2, int(round(font_scale * 3)))
    done = 0
    while ok_o and ok_u and (n <= 0 or done < n):
        if show_mask:
            try:
                rgb, M = aligner.align(fr_u, all_kps[done])
            except Exception:
                rgb, M = None, None
            if rgb is not None and M is not None:
                alpha = aligner.paste_mask_alpha(fr_u.shape, M)
                binm = (alpha > 0.5).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                th = max(2, fr_u.shape[1] // 500)
                cv2.drawContours(fr_o, cnts, -1, MASK_COLOR, th)
                cv2.drawContours(fr_u, cnts, -1, MASK_COLOR, th)
        a = _fit(fr_o, cell)
        b = _fit(fr_u, cell)
        if layout == "h":
            row = np.concatenate([a, b], axis=1)
            pos = [(12, 40), (cell[0] + 12, 40)]
        else:
            row = np.concatenate([a, b], axis=0)
            pos = [(12, 40), (12, cell[1] + 40)]
        for txt, (x, y) in zip(LABELS, pos):
            cv2.putText(row, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, LABEL_COLOR, thickness, cv2.LINE_AA)
        vw.write(row)
        done += 1
        ok_o, fr_o = co.read()
        ok_u, fr_u = cu.read()
    vw.release()
    co.release()
    cu.release()

    # mux audio from ours, fall back to orig
    audio_src = ours_path
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=codec_type", "-of",
                            "csv=p=0", ours_path], capture_output=True, text=True)
    if "audio" not in probe.stdout:
        audio_src = orig_path
    # pick the best available h264 encoder (server ffmpeg lacks libx264)
    encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                          capture_output=True, text=True).stdout
    if "libx264" in encs:
        vcodec = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    elif "h264_nvenc" in encs:
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        vcodec = ["-c:v", "mpeg4", "-q:v", "3"]
    cmd = ["ffmpeg", "-y", "-i", tmp.name, "-i", audio_src,
           "-map", "0:v:0", "-map", "1:a:0?"] + vcodec + [
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(tmp.name)
    print("[cmp] %s | %s -> %s (%d frames @ %.2f fps)" %
          (os.path.basename(orig_path), os.path.basename(ours_path),
           out_path, done, fps))


def auto_pairs(out_dir: str):
    pairs = []
    for orig in sorted(glob.glob(os.path.join(out_dir, "*_orig.mp4"))):
        stem = os.path.basename(orig)[:-len("_orig.mp4")]
        ours = os.path.join(out_dir, stem + ".mp4")
        if not os.path.exists(ours):
            ours = os.path.join(out_dir, stem + "_ours.mp4")
        if os.path.exists(ours):
            pairs.append((orig, ours, os.path.join(out_dir, "cmp_%s.mp4" % stem)))
        else:
            print("[cmp] SKIP %s: no partner found" % orig)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="*", help="out.mp4 A.mp4 B.mp4 ...")
    ap.add_argument("--layout", choices=["h", "v"], default="h")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--show-mask", action="store_true",
                    help="draw the replaced face-oval region (green contour) on both halves")
    ap.add_argument("--checkpoints", default=CKPT_DIR)
    a = ap.parse_args()
    if a.args:
        if len(a.args) % 3 != 0:
            sys.exit("explicit mode needs triples: out.mp4 orig.mp4 ours.mp4")
        triples = [(a.args[i + 1], a.args[i + 2], a.args[i])
                   for i in range(0, len(a.args), 3)]
    else:
        triples = auto_pairs(a.out_dir)
    if not triples:
        sys.exit("no pairs found")
    for orig, ours, out in triples:
        try:
            make_compare(orig, ours, out, a.layout, a.show_mask, a.checkpoints)
        except Exception as e:
            print("[cmp] FAIL %s: %r" % (orig, e))


if __name__ == "__main__":
    main()
