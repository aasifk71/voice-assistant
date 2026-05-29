"""
Speech-to-Text module using faster-whisper.
Handles VAD (voice activity detection) + transcription.
"""

import numpy as np
from faster_whisper import WhisperModel

# Model options: "tiny", "base", "small", "medium", "large-v3"
# "small" is the sweet spot: ~244MB, ~200ms on GPU
MODEL_SIZE = "small"

_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[STT] Loading Whisper {MODEL_SIZE} on CUDA...")
        _model = WhisperModel(
            MODEL_SIZE,
            device="cuda",
            compute_type="float16",  # faster on GPU, negligible quality loss
        )
        print("[STT] Model ready.")
    return _model


def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw PCM audio bytes to text.

    Args:
        audio_bytes: Raw 16-bit PCM audio bytes at sample_rate
        sample_rate: Audio sample rate (default 16000 Hz)

    Returns:
        Transcribed text string, or "" if nothing detected
    """
    model = get_model()

    # Convert bytes → float32 numpy array normalised to [-1, 1]
    audio_np = (
        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    )

    segments, info = model.transcribe(
        audio_np,
        beam_size=5,
        language="en",           # remove for auto-detect (adds ~50ms)
        vad_filter=True,         # skip silent segments automatically
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"[STT] Transcribed ({info.duration:.1f}s audio): '{text}'")
    return text


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sounddevice as sd

    DURATION = 5  # seconds
    SR = 16000
    print(f"Recording {DURATION}s of audio from mic…")
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    pcm_bytes = recording.tobytes()
    result = transcribe(pcm_bytes, SR)
    print(f"Result: {result}")
