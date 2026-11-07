"""GPU worker: a separate process that keeps the HeyGem12 models loaded.

Runs in its own process so the supervisor can kill and restart it if a job
hangs on the GPU. Protocol over a multiprocessing Pipe:
  parent -> {"video", "audio", "output", "precision"}
  child  -> {"ready": True, "warmup_s": ...} once, then per job
            {"ok": True, "wall_s": ...} or {"ok": False, "error": "..."}
"""
import os
import sys
import time
import traceback


def worker_main(conn, checkpoints: str, device: str, log_path: str, warmup: bool) -> None:
    # Pipeline prints progress; send it to a log file instead of the server console.
    log = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = log
    # Inputs are already normalized by the service; skip the pipeline's own
    # re-encode (and the temp dirs it never cleans up).
    os.environ["HEYGEM12_SKIP_FORMAT"] = "1"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from heygem12.pipeline import LipSyncPipeline

        t = time.perf_counter()
        pipes = {False: LipSyncPipeline(checkpoints, device=device)}
        warm_s = 0.0
        if warmup:
            # First GPU work in a process pays a ~16 s MIOpen init; pay it now.
            wv = os.path.join(repo, "assets", "man1_video.mp4")
            wa = os.path.join(repo, "assets", "man_en.wav")
            if os.path.exists(wv) and os.path.exists(wa):
                import tempfile
                out = os.path.join(tempfile.gettempdir(), f"heygem12_warmup_{os.getpid()}.mp4")
                os.environ.pop("HEYGEM12_SKIP_FORMAT")
                pipes[False].run(wv, wa, out, max_frames=8)
                os.environ["HEYGEM12_SKIP_FORMAT"] = "1"
                try:
                    os.remove(out)
                except OSError:
                    pass
        warm_s = time.perf_counter() - t
        print(f"[worker] ready in {warm_s:.1f}s (pid {os.getpid()})", flush=True)
        conn.send({"ready": True, "warmup_s": warm_s})
    except Exception:
        conn.send({"ready": False, "error": traceback.format_exc()[-2000:]})
        return

    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        t = time.perf_counter()
        try:
            precision = bool(job.get("precision"))
            if precision not in pipes:
                # 'precision' mode = GFPGAN restoration of the source face before DINet
                pipes[precision] = LipSyncPipeline(checkpoints, device=device, restore_source=True)
            print(f"[worker] job {job.get('id')} precision={precision}", flush=True)
            pipes[precision].run(job["video"], job["audio"], job["output"])
            conn.send({"ok": True, "wall_s": time.perf_counter() - t})
        except Exception as e:
            traceback.print_exc()
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"[:1000],
                       "wall_s": time.perf_counter() - t})
