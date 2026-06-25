"""Vocal-burst CAPTIONER — laion/vocalburst-captioning-whisper.

A Whisper-small fine-tune of ``laion/sound-effect-captioning-whisper`` trained on
``laion/improved_synthetic_vocal_burts`` to describe vocal bursts (laughs, sighs, gasps, moans,
screams, coughs, sneezes, throat-clears, lip smacks, panting, crying …). Given short 16 kHz mono
segments it returns a free-text caption per segment. The ensemble partner of the locator.

Weights load from a local directory if given, else from the HF hub
(``laion/vocalburst-captioning-whisper``).
"""
import os
from typing import List, Optional

import numpy as np
import torch

SAMPLE_RATE = 16000
CAPTIONER_REPO = "laion/vocalburst-captioning-whisper"


class VocalBurstCaptioner:
    """Caption short 16 kHz mono vocal-burst segments. GPU VRAM: ~1 GB.

    Args:
        device: torch device string.
        weights_dir: optional local dir with the model + tokenizer files. If None, downloads
            from ``laion/vocalburst-captioning-whisper`` (or ``$VB_WEIGHTS_DIR/captioner``).
    """

    def __init__(self, device: str = "cuda:0", weights_dir: Optional[str] = None):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device
        if weights_dir is None:
            env = os.environ.get("VB_WEIGHTS_DIR")
            local = os.path.join(env, "captioner") if env else None
            weights_dir = local if (local and os.path.exists(os.path.join(local, "model.safetensors"))) else CAPTIONER_REPO
        print(f"Loading vocal-burst captioner from {weights_dir} on {device}...")
        # Processor from base whisper-small (standard vocab) for robustness.
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(weights_dir).to(device).eval()
        self.model.generation_config.forced_decoder_ids = None
        print("Vocal-burst captioner loaded.")

    @torch.no_grad()
    def caption(self, segments: List[np.ndarray], batch_size: int = 8,
                max_new_tokens: int = 200) -> List[str]:
        """Caption a list of mono 16 kHz float32 segments (batched)."""
        out = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            feats = self.processor(batch, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt").input_features.to(self.device)
            ids = self.model.generate(feats, max_new_tokens=max_new_tokens)
            out.extend(t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True))
        return out

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
