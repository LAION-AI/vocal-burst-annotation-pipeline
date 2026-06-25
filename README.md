# Vocal-Burst Annotation Pipeline

Detect **and** caption **vocal bursts** in any audio — the non-speech human vocalizations that ordinary
ASR ignores: laughs, giggles, chuckles, sighs, gasps, moans, groans, screams, cries/sobs, coughs,
sneezes, sniffs, throat-clears, lip smacks/pops, panting, yawns, hums, grunts, …

It is a two-model **ensemble**:

1. **Detector** — [`laion/vocalburst-locator`](https://huggingface.co/laion/vocalburst-locator): finds
   *where* the bursts are (start/end timestamps).
2. **Captioner** — [`laion/vocalburst-captioning-whisper`](https://huggingface.co/laion/vocalburst-captioning-whisper):
   describes *what* each detected burst is.

Together they turn raw audio into a list of timestamped, captioned vocal-burst events. This is the
vocal-burst pre-pass of the LAION
[Universal Audio Annotation Pipeline](https://github.com/LAION-AI/univeral-audio-annotation-pipeline),
packaged here as a small standalone, inference-only library.

### 🎧 Live demo (audio + predictions, renders in your browser)
**https://laion-ai.github.io/vocal-burst-annotation-pipeline/** — interactive report with audio samples,
the model's predictions, and Gemini-3.1-Pro quality scores for the best detection thresholds; plus a
side-by-side comparison of the old sound-effect captioner vs the fine-tuned vocal-burst captioner.

---

## High-level overview

```
            ┌─────────────────────────── INPUT: audio (any length) ───────────────────────────┐
            ▼                                                                                  │
  ┌─────────────────────┐   per-frame burst probability (50 fps), threshold 0.88,             │
  │ VOCAL-BURST LOCATOR  │   merge_gap 0.3 s, min_dur 0.5 s  →  [{start, end, confidence}]     │
  │ vocalburst-locator   │                                                                     │
  └──────────┬──────────┘                                                                     │
             │ cut each detected segment                                                       │
             ▼                                                                                  │
  ┌─────────────────────┐                                                                      │
  │ VOCAL-BURST CAPTIONER│   describe each segment  →  caption string                          │
  │ vocalburst-captioning│                                                                      │
  │ -whisper             │                                                                      │
  └──────────┬──────────┘                                                                     │
             ▼                                                                                  │
   OUTPUT: [{start, end, confidence, duration, caption}, …]  ◄──────────────────────────────────┘
```

- **Detection threshold = 0.88** is the recommended default — the best operating point from a
  Gemini-3.1-Pro-judged sweep (see [Benchmark](#benchmark--how-we-chose-the-threshold)).
- Both models are Whisper-small-based (~1 GB VRAM each, ≤30 s windows; longer audio is windowed
  automatically).
- Inference-only. Weights auto-download from HuggingFace, or use the bundled offline copies.

## Install

```bash
git clone https://github.com/LAION-AI/vocal-burst-annotation-pipeline
cd vocal-burst-annotation-pipeline
pip install -r requirements.txt          # torch, transformers, huggingface_hub, librosa, soundfile, numpy
```

## Quick start

### Python
```python
from vocalburst_pipeline import VocalBurstPipeline

vb = VocalBurstPipeline(device="cuda:0")        # loads locator + captioner (auto-downloads weights)
events = vb.annotate("clip.wav")                # detect @0.88 → caption
for e in events:
    print(f"{e['start']:.2f}-{e['end']:.2f}s  (conf {e['confidence']:.2f})  {e['caption']}")
```

```text
3.21-4.85s  (conf 0.94)  A bright, genuine burst of laughter, light and amused.
12.50-13.10s (conf 0.90) A short, sharp gasp of surprise.
```

### Command line
```bash
python infer.py clip.wav                                  # JSON to stdout
python infer.py "audio/*.wav" --out results.json          # batch, write JSON
python infer.py clip.wav --threshold 0.85 --device cuda:0 # tune the threshold
```

### Just the detector, or just the captioner
```python
from vocalburst_pipeline import VocalBurstLocator, VocalBurstCaptioner
import librosa
wav = librosa.load("clip.wav", sr=16000, mono=True)[0]

loc = VocalBurstLocator(device="cuda:0")
events = loc.detect(wav, threshold=0.88, merge_gap=0.3, min_dur=0.5)   # [{start,end,confidence,duration}]

cap = VocalBurstCaptioner(device="cuda:0")
segs = [wav[int(e["start"]*16000):int(e["end"]*16000)] for e in events]
captions = cap.caption(segs)
```

## Output format

`annotate()` (and the CLI) return a list of events sorted by start time:

```json
[
  {"start": 3.21, "end": 4.85, "confidence": 0.94, "duration": 1.64,
   "caption": "A bright, genuine burst of laughter, light and amused."},
  {"start": 12.50, "end": 13.10, "confidence": 0.90, "duration": 0.60,
   "caption": "A short, sharp gasp of surprise."}
]
```

## How it works (details)

### 1. Detector — `laion/vocalburst-locator`
A **Whisper-small encoder** with a per-frame binary **segmentation head** (`proj → temporal Conv1d →
linear`) that emits a vocal-burst probability for **every 20 ms frame (50 fps)** of a 30 s window. Audio
longer than 30 s is split into ≤30 s windows, run as one batch, and the per-window probabilities are
concatenated into a single continuous timeline (so a burst that straddles a window boundary stays one
event). The timeline is **median-smoothed** (erases 1–2-frame spikes), **thresholded**, then grouped into
events with **small-gap merging** and a **minimum-duration filter**.

Post-processing parameters (defaults are the validated operating point):

| param | default | meaning |
|-------|---------|---------|
| `threshold` | **0.88** | per-frame confidence cutoff (higher = fewer false positives / more misses) |
| `merge_gap` | 0.3 s | merge two events separated by less than this |
| `min_dur` | 0.5 s | drop events shorter than this |
| `smooth_frames` | 5 | median-filter window over the frame timeline |

### 2. Captioner — `laion/vocalburst-captioning-whisper`
A **Whisper-small fine-tune** of `laion/sound-effect-captioning-whisper`, trained on
[`laion/improved_synthetic_vocal_burts`](https://huggingface.co/datasets/laion/improved_synthetic_vocal_burts)
(target = the `Flash 2.5 Annotation.caption` field; best run **LR 1e-5, 3 epochs**, cosine schedule with
10 % warmup, bf16). It raises the audio↔caption agreement (voiceclap-small-v2 cosine) from **0.190**
(untuned base) to **0.251**. Each detected segment is cut and captioned in a batched call.

## Weights

By default the code **auto-downloads** the weights from HuggingFace
(`laion/vocalburst-locator` + `laion/vocalburst-captioning-whisper`) and caches them — nothing else needed.

For **offline use**, the weights are also bundled here, split into <95 MB parts (to fit GitHub's
100 MB/file limit). Reassemble them once:

```bash
bash weights/assemble_weights.sh          # cats the parts back together + verifies sha256
export VB_WEIGHTS_DIR="$(pwd)/weights"    # make the library use the local copies
```

`weights/locator/model.pt` (~972 MB) and `weights/captioner/model.safetensors` (~967 MB) are produced;
the small captioner config/tokenizer files are committed directly.

## Benchmark — how we chose the threshold

We swept the detector's confidence threshold from **0.85 → 0.92 (1 % steps)** over **150 audio clips**
(clean-speech false-positive checks + clips with inserted bursts + isolated bursts), with
`merge_gap = 0.3 s`, `min_dur = 0.5 s`. For **every (clip × threshold)** the detected segments were
captioned, and the **audio + (start, end, caption)** list was sent to **Gemini 3.1 Pro**, which rated
three axes 0–5 (5 = perfect): **caption quality**, **timestamp accuracy**, and **completeness** (cover
ALL real bursts, penalizing both misses and false positives) — **1,200 independent LLM judgments**.
*overall* = mean of the three.

| rank | threshold | overall | completeness | caption quality | timestamp accuracy |
|------|-----------|---------|--------------|-----------------|--------------------|
| 🥇 | **0.88** | **3.475** | 3.11 | 3.24 | 4.07 |
| 🥈 | 0.89 | 3.469 | 3.15 | 3.18 | 4.08 |
| 🥉 | 0.85 | 3.466 | 3.11 | 3.22 | 4.07 |
| 4 | 0.90 | 3.445 | 3.10 | 3.24 | 4.00 |
| 5 | 0.86 | 3.411 | 3.05 | 3.14 | 4.04 |
| 6 | 0.87 | 3.390 | 3.07 | 3.10 | 4.00 |
| 7 | 0.91 | 3.364 | 3.05 | 3.14 | 3.91 |
| 8 | 0.92 | 3.363 | 3.02 | 3.15 | 3.92 |

**Findings:** scores cluster tightly across the band; **threshold 0.88 is the best overall** (the default
here). Timestamp accuracy is consistently strong (~4.0); completeness is the weakest axis and degrades at
0.91–0.92 as real bursts get missed. The full interactive report (audio players + predictions + per-clip
Gemini scores for the top-3 thresholds) is the [live demo](https://laion-ai.github.io/vocal-burst-annotation-pipeline/demo.html).

## Models & links

- Detector: [`laion/vocalburst-locator`](https://huggingface.co/laion/vocalburst-locator)
- Captioner: [`laion/vocalburst-captioning-whisper`](https://huggingface.co/laion/vocalburst-captioning-whisper)
  (fine-tune of [`laion/sound-effect-captioning-whisper`](https://huggingface.co/laion/sound-effect-captioning-whisper))
- Captioner training data: [`laion/improved_synthetic_vocal_burts`](https://huggingface.co/datasets/laion/improved_synthetic_vocal_burts)
- Full audio-annotation pipeline: [LAION-AI/univeral-audio-annotation-pipeline](https://github.com/LAION-AI/univeral-audio-annotation-pipeline)

## Limitations

- **30-second window**: processed in ≤30 s windows (handled automatically; boundaries accurate to ±20 ms).
- **Synthetic training data**: trained on synthetic mixtures; real-world performance may vary.
- **Dense speech backgrounds** can raise false positives — the 0.88 threshold is tuned to mitigate this.

## License

Apache 2.0 — see [LICENSE](LICENSE). Model weights follow their respective HuggingFace model-card licenses.
