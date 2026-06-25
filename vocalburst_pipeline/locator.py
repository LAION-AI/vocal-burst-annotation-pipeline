"""Vocal-burst LOCATOR — laion/vocalburst-locator.

A Whisper-small encoder + a per-frame binary segmentation head that emits a vocal-burst
probability for every 20 ms frame (50 fps) of a 30 s window. Audio longer than 30 s is split
into <=30 s windows, run as a batch, and the per-window probabilities are concatenated into one
continuous timeline (so a burst straddling a window boundary stays one event). The timeline is
median-smoothed, thresholded, and grouped into events with small-gap merging + a minimum-duration
filter. Returns ``[{start, end, confidence, duration}]``.

Weights load from a local path if given, else from the HF hub (``laion/vocalburst-locator``).
"""
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
WINDOW_SECONDS = 30.0
FPS = 50                       # 1500 frames / 30 s
LOCATOR_REPO = "laion/vocalburst-locator"
BEST_THRESHOLD = 0.88          # best on the Gemini-3.1-Pro-judged 0.85-0.92 sweep (see README)


class WhisperSegmenter(nn.Module):
    """Whisper-small encoder + per-frame binary segmentation head."""

    def __init__(self, whisper_id: str = "openai/whisper-small"):
        super().__init__()
        from transformers import WhisperModel
        self.whisper = WhisperModel.from_pretrained(whisper_id)
        d_model = self.whisper.config.d_model       # 768
        hidden = max(256, d_model // 2)             # 384
        self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.1))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(), nn.Dropout(0.1))
        self.out = nn.Linear(hidden, 1)

    def forward(self, input_features: torch.FloatTensor) -> torch.FloatTensor:
        enc = self.whisper.encoder(input_features=input_features).last_hidden_state
        h = self.proj(enc)
        h = self.temporal(h.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out(h).squeeze(-1)              # [B, 1500] logits


def _median_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter over the frame timeline — removes 1-2 frame spikes/dropouts."""
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))], dtype=np.float32)


def _extract_events(probs: np.ndarray, threshold: float, merge_gap: float,
                    min_dur: float) -> List[tuple]:
    """Group a frame-probability timeline into (start_s, end_s, confidence) events."""
    binary = (probs > threshold).astype(np.float32)
    raw, in_ev, start = [], False, 0
    for f in range(len(binary)):
        if binary[f] > 0.5 and not in_ev:
            start, in_ev = f, True
        elif binary[f] <= 0.5 and in_ev:
            raw.append((start, f)); in_ev = False
    if in_ev:
        raw.append((start, len(binary)))
    if not raw:
        return []
    sec = 1.0 / FPS
    events = [((s * sec, e * sec), (s, e)) for s, e in raw]
    merged = [events[0]]
    for (s, e), (fs, fe) in events[1:]:
        (ps, pe), (pfs, pfe) = merged[-1]
        if s - pe <= merge_gap:                      # bridge small gaps (outliers)
            merged[-1] = ((ps, e), (pfs, fe))
        else:
            merged.append(((s, e), (fs, fe)))
    out = []
    for (s, e), (fs, fe) in merged:
        if e - s >= min_dur:
            out.append((round(s, 3), round(e, 3), round(float(probs[fs:fe].mean()), 3)))
    return out


class VocalBurstLocator:
    """Detect vocal-burst candidates across an arbitrarily long clip. GPU VRAM: ~1 GB.

    Args:
        device: torch device string.
        weights_path: optional local path to ``model.pt``. If None, downloads from
            ``laion/vocalburst-locator`` (or set ``$VB_WEIGHTS_DIR/locator/model.pt``).
    """

    def __init__(self, device: str = "cuda:0", weights_path: Optional[str] = None):
        from transformers import WhisperFeatureExtractor
        self.device = torch.device(device)
        if weights_path is None:
            env = os.environ.get("VB_WEIGHTS_DIR")
            local = os.path.join(env, "locator", "model.pt") if env else None
            if local and os.path.exists(local):
                weights_path = local
            else:
                from huggingface_hub import hf_hub_download
                weights_path = hf_hub_download(repo_id=LOCATOR_REPO, filename="model.pt")
        print(f"Loading vocal-burst locator from {weights_path} on {device}...")
        self.model = WhisperSegmenter("openai/whisper-small")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        print("Vocal-burst locator loaded.")

    @torch.no_grad()
    def detect(self, wav: np.ndarray, threshold: float = BEST_THRESHOLD,
               merge_gap: float = 0.3, min_dur: float = 0.5,
               smooth_frames: int = 5) -> List[Dict]:
        """Return candidate events ``[{start, end, confidence, duration}]``.

        Args:
            wav: mono 16 kHz float32 audio of the FULL clip (any length).
            threshold: confidence cutoff (default 0.88 — best on the Gemini-judged sweep).
            merge_gap: merge events closer than this (s). min_dur: drop events shorter than this (s).
        """
        win = int(WINDOW_SECONDS * SAMPLE_RATE)
        total = len(wav)
        n_win = max(1, int(np.ceil(total / win)))
        windows, valid_frames = [], []
        for i in range(n_win):
            seg = wav[i * win:(i + 1) * win]
            valid_frames.append(int(round(min(WINDOW_SECONDS, (total - i * win) / SAMPLE_RATE) * FPS)))
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            windows.append(seg)
        feats = self.fe(windows, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(feats)
        else:
            logits = self.model(feats)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        timeline = np.concatenate([probs[i, :valid_frames[i]] for i in range(n_win)])
        timeline = _median_smooth(timeline, k=smooth_frames)
        events = _extract_events(timeline, threshold, merge_gap, min_dur)
        return [{"start": s, "end": e, "confidence": c, "duration": round(e - s, 3)}
                for s, e, c in events]

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
