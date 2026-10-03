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
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames (debug/smoke test)")
    parser.add_argument("--restore", action="store_true",
                        help="enable chaofen_after (GFPGAN on the DINet render; the original "
                             "does NOT do this -- its restoration only feeds the input crop)")
    parser.add_argument("--no-pre-restore", action="store_true",
                        help="disable chaofen_before (GFPGAN-sharpening the source crop "
                             "before feeding DINet)")
    parser.add_argument("--fuse-mode", default="oval", choices=("oval", "fuse_mask"),
                        help='oval = original-style face-oval hard paste (default); '
                             'fuse_mask = old feathered fuse_mask+mouth-ellipse blend')
    parser.add_argument("--no-kps-stabilize", action="store_true",
                        help="disable temporal stabilization of the per-frame 5-point "
                             "keypoints (default on: kills 1-frame detect jitter that "
                             "shows up as face/mouth flicker)")
    parser.add_argument("--render-ema", type=float, default=0.0,
                        help="EXPERIMENTAL temporal EMA on the DINet render (0..1; default 0 = "
                             "off). Temporal blending ghosts lip shapes during speech because "
                             "the decoder repaints the whole masked region at phoneme "
                             "boundaries -- left at 0 (the original does no temporal render "
                             "processing either)")
    args = parser.parse_args()

    from heygem12.pipeline import LipSyncPipeline

    pipeline = LipSyncPipeline(args.checkpoints, device=args.device,
                               restore=args.restore,
                               restore_source=not args.no_pre_restore,
                               fuse_mode=args.fuse_mode,
                               kps_stabilize=not args.no_kps_stabilize,
                               render_ema=args.render_ema)
    out = pipeline.run(args.video, args.audio, args.output, max_frames=args.max_frames)
    print(f"[heygem12] done: {out}")


if __name__ == "__main__":
    sys.exit(main())
