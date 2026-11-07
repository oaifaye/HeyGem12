"""Command line interface: python -m heygem12.cli --video v.mp4 --audio a.wav --output out.mp4"""

import argparse
import os
import sys

_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def main():
    parser = argparse.ArgumentParser(
        prog="heygem12",
        description="Lip-sync a video to a new audio (Python 3.12 reimplementation of HeyGem core).")
    parser.add_argument("--video", required=True, help="path to source video (mp4 etc.)")
    parser.add_argument("--audio", required=True, help="path to driving audio (wav)")
    parser.add_argument("--output", required=True, help="path to output video (mp4)")
    parser.add_argument("--checkpoints", default=_DEFAULT_CKPT, help="checkpoints directory")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--audio-device", default=None,
                        help="device of the WeNet speech features (default: same as --device)")
    parser.add_argument("--ort-device", default="auto", choices=["auto", "cpu"],
                        help="ONNX face models: auto prefers the GPU, cpu keeps them off it")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames (debug/smoke test)")
    parser.add_argument("--restore", action="store_true",
                        help="enable chaofen_after (GFPGAN on the DINet render; the original "
                             "does NOT do this -- its restoration only feeds the input crop)")
    parser.add_argument("--pre-restore", action="store_true",
                        help="enable chaofen_before (GFPGAN-restore the frame before cropping; "
                             "the original gates this on a blur/quality check and mostly leaves "
                             "it OFF for sharp sources -- default off, see HANDOFF)")
    args = parser.parse_args()
    os.environ["HEYGEM12_ORT"] = args.ort_device

    # Match the original's numeric environment: no TF32 matmul scatter.
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    from heygem12.pipeline import LipSyncPipeline

    pipeline = LipSyncPipeline(args.checkpoints, device=args.device,
                               audio_device=args.audio_device,
                               restore=args.restore,
                               restore_source=args.pre_restore)
    out = pipeline.run(args.video, args.audio, args.output, max_frames=args.max_frames)
    print(f"[heygem12] done: {out}")


if __name__ == "__main__":
    sys.exit(main())
