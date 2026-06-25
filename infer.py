#!/usr/bin/env python3
"""CLI: annotate vocal bursts in audio file(s).

  python infer.py clip.wav                       # prints JSON to stdout
  python infer.py a.wav b.mp3 --out results.json --threshold 0.88 --device cuda:0
"""
import argparse, json, glob, sys
from vocalburst_pipeline import VocalBurstPipeline, BEST_THRESHOLD

ap = argparse.ArgumentParser()
ap.add_argument("audio", nargs="+", help="audio file(s) or globs")
ap.add_argument("--out", default=None, help="write JSON here (else stdout)")
ap.add_argument("--threshold", type=float, default=BEST_THRESHOLD)
ap.add_argument("--merge-gap", type=float, default=0.3)
ap.add_argument("--min-dur", type=float, default=0.5)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--weights-dir", default=None, help="local weights dir (else auto-download from HF)")
ap.add_argument("--no-caption", action="store_true")
args = ap.parse_args()

paths = [p for pat in args.audio for p in (glob.glob(pat) or [pat])]
vb = VocalBurstPipeline(device=args.device, threshold=args.threshold,
                        merge_gap=args.merge_gap, min_dur=args.min_dur, weights_dir=args.weights_dir)
res = {p: vb.annotate(p, caption=not args.no_caption) for p in paths}
out = json.dumps(res if len(res) > 1 else next(iter(res.values())), indent=2, ensure_ascii=False)
(open(args.out, "w").write(out) if args.out else sys.stdout.write(out + "\n"))
