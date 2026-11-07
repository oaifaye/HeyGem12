"""ffmpeg/ffprobe helpers, URL download, and input preparation for the pipeline."""
import hashlib
import ipaddress
import json
import math
import os
import shutil
import socket
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from .config import settings


class MediaError(Exception):
    """Input problem attributable to the client (bad file, bad URL, too long...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _run(cmd, timeout=1800) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({r.returncode}): {r.stderr.strip()[-800:]}")


def probe(path: str) -> Dict:
    r = subprocess.run([settings.ffprobe, "-v", "error", "-show_format", "-show_streams",
                        "-of", "json", path], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise MediaError("invalid_media", f"cannot read media file: {r.stderr.strip()[-300:]}")
    info = json.loads(r.stdout or "{}")
    out = {"duration": float(info.get("format", {}).get("duration") or 0), "video": None, "audio": None}
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and out["video"] is None and \
                s.get("disposition", {}).get("attached_pic", 0) == 0:
            num, _, den = (s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            out["video"] = {"width": s.get("width"), "height": s.get("height"), "fps": fps,
                            "duration": float(s.get("duration") or out["duration"] or 0)}
        elif s.get("codec_type") == "audio" and out["audio"] is None:
            out["audio"] = {"duration": float(s.get("duration") or out["duration"] or 0),
                            "sample_rate": int(s.get("sample_rate") or 0)}
    return out


# ------------------------------------------------------------------ downloads
def _check_public_host(url: str) -> None:
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise MediaError("invalid_parameter", f"unsupported URL: {url!r} (http/https only)")
    if settings.allow_private_urls:
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise MediaError("invalid_url", f"cannot resolve host {p.hostname!r}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise MediaError("invalid_url", f"URL host {p.hostname!r} resolves to a non-public address "
                             f"({ip}); set HEYGEM12_API_ALLOW_PRIVATE_URLS=1 to allow it")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_host(newurl)  # re-validate every redirect hop
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dest: Path) -> Path:
    _check_public_host(url)
    limit = settings.max_upload_mb * 1024 * 1024
    opener = urllib.request.build_opener(_NoRedirect())
    req = urllib.request.Request(url, headers={"User-Agent": "heygem12-api/1.0"})
    try:
        with opener.open(req, timeout=60) as resp, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise MediaError("file_too_large", f"remote file exceeds {settings.max_upload_mb} MB")
                f.write(chunk)
    except MediaError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise MediaError("download_failed", f"cannot download {url}: {e}")
    return dest


# -------------------------------------------------------------- preparation
def prepare_audio(src: str, dst: Path) -> float:
    """Any audio (or video with audio) -> 16 kHz mono PCM wav. Returns duration (s)."""
    info = probe(src)
    if not info["audio"]:
        raise MediaError("invalid_audio", "audio input has no audio stream")
    _run([settings.ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vn",
          "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    return probe(str(dst))["duration"]


def _encode_args(fps: float):
    return ["-an", "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", f"{fps:.6g}"]


def _reversed_copy(src: str, fps: float, dur: float) -> Path:
    """Reversed version of the source, cached by content hash. Reversal is done in
    5 s chunks so ffmpeg's reverse filter never has to buffer a whole long clip."""
    h = hashlib.sha1()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out = settings.cache_dir / f"rev_{h.hexdigest()[:20]}_{fps:.3f}.mp4"
    if out.exists():
        return out
    tmp = settings.work_dir / f"rev_{os.getpid()}_{h.hexdigest()[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        seg = 5.0
        n = max(1, math.ceil(dur / seg))
        parts = []
        for i in range(n):
            p = tmp / f"r{i:04d}.mp4"
            _run([settings.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{i * seg}", "-t", f"{seg}",
                  "-i", src, "-vf", "reverse", *_encode_args(fps), str(p)])
            if p.exists() and p.stat().st_size > 0:
                parts.append(p)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in reversed(parts)), encoding="utf-8")
        part_out = tmp / "rev.mp4"
        _run([settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
              *_encode_args(fps), str(part_out)])
        shutil.move(str(part_out), out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def prepare_video(src: str, dst: Path, target_duration: Optional[float], workdir: Path) -> Dict:
    """Re-encode the source video (h264 crf15, cfr, no audio) for the pipeline.

    If target_duration exceeds the source length, the video is extended by
    ping-pong looping (forward, reversed, forward, ...) so there is no visible
    jump where the loop restarts; if it is shorter, the video is trimmed.
    Returns probe info of the prepared video.
    """
    info = probe(src)
    v = info["video"]
    if not v:
        raise MediaError("invalid_video", "video input has no video stream")
    fps = v["fps"] if 1 <= v["fps"] <= 120 else 25.0
    dur = v["duration"] or info["duration"]
    if target_duration and target_duration > dur + 0.05:
        fwd = workdir / "fwd.mp4"
        _run([settings.ffmpeg, "-y", "-loglevel", "error", "-i", src, *_encode_args(fps), str(fwd)])
        rev = _reversed_copy(src, fps, dur)
        n = math.ceil(target_duration / dur)
        lst = workdir / "loop.txt"
        lst.write_text("".join(f"file '{(fwd if i % 2 == 0 else rev).as_posix()}'\n" for i in range(n)),
                       encoding="utf-8")
        _run([settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
              "-t", f"{target_duration:.3f}", *_encode_args(fps), str(dst)])
    else:
        # Trim to the output length: the pipeline detects faces on every input frame
        # and holds them all in RAM, so frames past the audio end are pure waste.
        trim = ["-t", f"{target_duration + 0.1:.3f}"] if target_duration and target_duration < dur else []
        _run([settings.ffmpeg, "-y", "-loglevel", "error", "-i", src, *trim, *_encode_args(fps), str(dst)])
    out = probe(str(dst))
    if not out["video"]:
        raise RuntimeError("prepared video has no video stream")
    return out
