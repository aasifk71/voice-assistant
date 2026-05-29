"""
Text-to-Speech module using Piper TTS.
Piper is fast, runs fully offline, and produces natural-sounding speech.

Install:
    pip install piper-tts

Download a voice model:
    python -m piper --download-dir ./models en_US-lessac-medium

Then set PIPER_MODEL_PATH below (or via env var).
"""

import io
import os
import subprocess
import wave

# Path to the downloaded .onnx voice model
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL",
    "./models/en_US-lessac-medium.onnx",  # change to your downloaded model
)

# Output sample rate (Piper outputs 22050 Hz by default)
SAMPLE_RATE = 22050


def synthesize(text: str) -> bytes:
    """
    Convert text to WAV audio bytes using Piper TTS.

    Returns:
        Raw WAV file bytes (can be sent directly over WebSocket or written to file)
    """
    if not text.strip():
        return b""

    print(f"[TTS] Synthesizing: '{text[:60]}{'...' if len(text)>60 else ''}'")

    # Piper is called as a subprocess (simplest integration)
    result = subprocess.run(
        [
            "piper",
            "--model", PIPER_MODEL_PATH,
            "--output-raw",          # raw PCM output, we wrap in WAV below
        ],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=15,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {result.stderr.decode()}")

    # Wrap raw PCM in a WAV container so browsers can play it natively
    pcm_bytes = result.stdout
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=SAMPLE_RATE, channels=1, sampwidth=2)

    print(f"[TTS] Generated {len(wav_bytes)} bytes of audio")
    return wav_bytes


def synthesize_streaming(text: str):
    """
    Generator that yields WAV audio chunks as Piper produces them.
    Useful for lower perceived latency: start playing before all audio is ready.

    Yields bytes chunks.
    """
    # For streaming, we pipe output sentence-by-sentence
    # Split on sentence boundaries to start TTS sooner
    sentences = _split_sentences(text)

    for sentence in sentences:
        if sentence.strip():
            yield synthesize(sentence)


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int, sampwidth: int) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter to enable streaming TTS."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! I am your voice assistant. How can I help you today?"
    wav = synthesize(text)

    out_path = "/tmp/test_output.wav"
    with open(out_path, "wb") as f:
        f.write(wav)
    print(f"Audio written to {out_path}")

    # Auto-play if aplay is available
    try:
        subprocess.run(["aplay", out_path], check=True)
    except FileNotFoundError:
        print("Install 'aplay' to auto-play, or open the file manually.")
