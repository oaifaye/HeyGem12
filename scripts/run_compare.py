# -*- coding: utf-8 -*-
"""End-to-end comparison runner: for each (video, audio) pair, render with the
ORIGINAL HeyGem (HG/run.py, py3.8 env) and with THIS project (heygem12.cli),
then mux the two outputs into a side-by-side comparison video.

Usage (on the server, run with sudo because assets/out is root-owned):
  python scripts/run_compare.py assets/media/demo1_video.mp4 assets/media/woman_cn.wav \
                                assets/media/demo2_video.mp4 assets/media/man_cn.wav

Each pair = <video> <audio> (two positional args, repeatable). The output stem
is the video's basename; results land in assets/out/:
  <stem>_orig.mp4   original HeyGem output
  <stem>.mp4        our output
  cmp_<stem>.mp4    side-by-side comparison (ORIG | OURS)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import cv2

HERE = "/opt/projects/heygem12"
HG_DIR = os.path.join(HERE, "HG")
PY_ORIG = "/opt/anaconda3/envs/heygem38/bin/python"
PY_OURS = "/opt/anaconda3/envs/heygem312/bin/python"
OUT_DIR = os.path.join(HERE, "assets", "out")

sys.path.insert(0, os.path.join(HERE, "scripts"))
from compare import make_compare  # noqa: E402


def _clean_env():
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)   # onnxruntime/py torch conflict workaround
    env.pop("PYTHONPATH", None)        # never leak the DINet hook
    return env


def run_original(video: str, audio: str, dest: str) -> None:
    # The original service hardcodes work_id "1004" (HG/run.py) and SKIPS the
    # whole task when artifacts of a previous run still exist (it reuses
    # 1004_format.* / 1004-r.mp4) -- clean them or every pair would get pair 1's
    # output. Empirically confirmed: stale 1004-r.mp4 mtime across runs.
    import glob as _glob
    for pat in ("1004-*.mp4", "1004-*.wav", "1004_format.*"):
        for f in _glob.glob(os.path.join(HG_DIR, pat)):
            os.remove(f)
    for f in _glob.glob(os.path.join(HG_DIR, "result", "1004*")):
        os.remove(f)
    before = set()
    if os.path.isdir(HG_DIR):
        before = {f for f in os.listdir(HG_DIR) if f.endswith("-r.mp4")}
    cmd = [PY_ORIG, "run.py", "--audio_path", audio, "--video_path", video]
    env = _clean_env()
    # the original's format step shells out to "ffmpeg"; /usr/local/bin/ffmpeg
    # (custom 5.1.2, no libx264) breaks it with "Option not found", so put the
    # full-featured /usr/bin/ffmpeg first on PATH
    env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
    print("[orig] %s + %s ..." % (video, audio), flush=True)
    p = subprocess.run(cmd, cwd=HG_DIR, env=env,
                       capture_output=True, text=True)
    m = re.search(r"Video result saved in (\S+)", p.stdout + p.stderr)
    result = None
    if m and os.path.exists(m.group(1)):
        result = m.group(1)
    else:
        new = [f for f in os.listdir(HG_DIR)
               if f.endswith("-r.mp4") and f not in before]
        if new:
            result = os.path.join(HG_DIR, max(new, key=lambda f: os.path.getmtime(
                os.path.join(HG_DIR, f))))
    if result is None:
        sys.exit("[orig] FAILED for %s\n%s" % (video, p.stderr[-3000:]))
    shutil.copyfile(result, dest)
    print("[orig] -> %s" % dest, flush=True)


def run_ours(video: str, audio: str, dest: str, device: str) -> None:
    cmd = [PY_OURS, "-X", "utf8", "-u", "-m", "heygem12.cli",
           "--video", video, "--audio", audio, "--output", dest,
           "--device", device]
    print("[ours] %s + %s ..." % (video, audio), flush=True)
    p = subprocess.run(cmd, cwd=HERE, env=_clean_env())
    if p.returncode != 0 or not os.path.exists(dest):
        sys.exit("[ours] FAILED for %s (rc=%d)" % (video, p.returncode))
    print("[ours] -> %s" % dest, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="<video> <audio> [<video> <audio> ...]")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--layout", choices=["h", "v"], default="h")
    ap.add_argument("--skip-orig", action="store_true", help="reuse existing <stem>_orig.mp4")
    ap.add_argument("--skip-ours", action="store_true", help="reuse existing <stem>.mp4")
    ap.add_argument("--show-mask", action="store_true",
                    help="draw the replaced face-oval region on the comparison video")
    a = ap.parse_args()
    if len(a.pairs) % 2 != 0:
        sys.exit("need an even number of args: <video> <audio> pairs")
    os.makedirs(a.out_dir, exist_ok=True)
    prev_first = None
    for i in range(0, len(a.pairs), 2):
        video = os.path.abspath(a.pairs[i])
        audio = os.path.abspath(a.pairs[i + 1])
        if not os.path.exists(video) or not os.path.exists(audio):
            sys.exit("missing file: %s / %s" % (video, audio))
        stem = os.path.splitext(os.path.basename(video))[0]
        orig_out = os.path.join(a.out_dir, stem + "_orig.mp4")
        ours_out = os.path.join(a.out_dir, stem + ".mp4")
        cmp_out = os.path.join(a.out_dir, "cmp_%s.mp4" % stem)
        if not (a.skip_orig and os.path.exists(orig_out)):
            run_original(video, audio, orig_out)
        # guard against the service's output cache: different source videos must
        # give different first frames
        cap = cv2.VideoCapture(orig_out)
        ok0, f0 = cap.read()
        cap.release()
        if not ok0:
            sys.exit("[orig] empty output %s" % orig_out)
        h0 = hash(f0.tobytes())
        if prev_first is not None and h0 == prev_first:
            sys.exit("[orig] output %s identical to the previous pair's -- "
                     "the original service likely served a cached file" % orig_out)
        prev_first = h0
        if not (a.skip_ours and os.path.exists(ours_out)):
            run_ours(video, audio, ours_out, a.device)
        make_compare(orig_out, ours_out, cmp_out, a.layout, a.show_mask)
    print("ALL_COMPARE_DONE")


if __name__ == "__main__":
    main()

'''
sudo bash -c 'cd /opt/projects/heygem12 && env -u LD_LIBRARY_PATH /opt/anaconda3/envs/heygem312/bin/python -X utf8 -u scripts/run_compare.py \
  assets/media/demo1_video.mp4 assets/media/woman_cn.wav \
  assets/media/demo2_video.mp4 assets/media/man_cn.wav \
  assets/media/demo3_video.mp4 assets/media/man_cn.wav'
  
  
sudo bash -c 'cd /opt/projects/heygem12 && env -u LD_LIBRARY_PATH /opt/anaconda3/envs/heygem312/bin/python -X utf8 -u scripts/run_compare.py --show-mask \
  assets/media/demo1_video.mp4 assets/media/woman_cn.wav \
  assets/media/demo2_video.mp4 assets/media/man_cn.wav \
  assets/media/demo3_video.mp4 assets/media/man_cn.wav'

'''

