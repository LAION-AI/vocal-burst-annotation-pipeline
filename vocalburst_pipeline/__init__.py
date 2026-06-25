"""Vocal-Burst Annotation Pipeline — detector + captioner ensemble."""
from .pipeline import VocalBurstPipeline, load_audio_16k
from .locator import VocalBurstLocator, BEST_THRESHOLD, SAMPLE_RATE
from .captioner import VocalBurstCaptioner
__all__ = ["VocalBurstPipeline", "VocalBurstLocator", "VocalBurstCaptioner",
           "load_audio_16k", "BEST_THRESHOLD", "SAMPLE_RATE"]
__version__ = "1.0.0"
