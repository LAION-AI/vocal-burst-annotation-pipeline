"""End-to-end vocal-burst annotation: detect -> cut -> caption.

    from vocalburst_pipeline import VocalBurstPipeline
    vb = VocalBurstPipeline(device="cuda:0")        # loads locator + captioner
    events = vb.annotate("clip.wav")                # [{start, end, confidence, caption}]
"""
from typing import List, Dict, Optional

import numpy as np

from .locator import VocalBurstLocator, SAMPLE_RATE, BEST_THRESHOLD
from .captioner import VocalBurstCaptioner


def load_audio_16k(path: str) -> np.ndarray:
    """Load any audio file as mono 16 kHz float32."""
    import librosa
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]


class VocalBurstPipeline:
    """The locator + captioner ensemble.

    Args:
        device: torch device.
        threshold: detection confidence (default 0.88 — best on the Gemini-judged sweep).
        merge_gap / min_dur: locator post-processing (defaults 0.3 s / 0.5 s).
        weights_dir: optional local weights dir (else auto-download from HF).
    """

    def __init__(self, device: str = "cuda:0", threshold: float = BEST_THRESHOLD,
                 merge_gap: float = 0.3, min_dur: float = 0.5, weights_dir: Optional[str] = None):
        self.threshold, self.merge_gap, self.min_dur = threshold, merge_gap, min_dur
        loc_w = f"{weights_dir}/locator/model.pt" if weights_dir else None
        cap_w = f"{weights_dir}/captioner" if weights_dir else None
        self.locator = VocalBurstLocator(device=device, weights_path=loc_w)
        self.captioner = VocalBurstCaptioner(device=device, weights_dir=cap_w)

    def annotate(self, audio, caption: bool = True) -> List[Dict]:
        """Detect vocal bursts and (optionally) caption them.

        Args:
            audio: file path, or a mono 16 kHz float32 numpy array.
        Returns: ``[{start, end, confidence, duration, caption}]`` sorted by start time.
        """
        wav = load_audio_16k(audio) if isinstance(audio, str) else np.asarray(audio, dtype=np.float32)
        events = self.locator.detect(wav, threshold=self.threshold,
                                     merge_gap=self.merge_gap, min_dur=self.min_dur)
        if caption and events:
            segs = [wav[int(e["start"] * SAMPLE_RATE):int(e["end"] * SAMPLE_RATE)] for e in events]
            for e, c in zip(events, self.captioner.caption(segs)):
                e["caption"] = c
        return sorted(events, key=lambda e: e["start"])

    def cleanup(self):
        self.locator.cleanup(); self.captioner.cleanup()
