"""
FastAPI WebSocket server — the glue layer.

Flow:
  Browser → sends raw PCM audio chunks over WebSocket
  Server  → STT (Whisper) → LLM (Mistral) → TTS (Piper) → sends WAV back
  Browser → plays received WAV chunks

Start:
    uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload
"""

import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.stt import transcribe
from server.llm import chat, reset_history
from server.tts import synthesize

app = FastAPI(title="Voice Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the browser client from /client
app.mount("/static", StaticFiles(directory="client"), name="static")


@app.get("/")
async def index():
    return FileResponse("client/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")
    reset_history()  # fresh conversation per connection

    try:
        while True:
            # ── Receive audio bytes from client ──────────────────────────────
            audio_bytes = await ws.receive_bytes()

            if not audio_bytes:
                continue

            t0 = time.perf_counter()

            # ── STT ──────────────────────────────────────────────────────────
            transcript = transcribe(audio_bytes, sample_rate=16000)

            if not transcript:
                # Nothing heard — send empty signal so UI can reset
                await ws.send_json({"type": "transcript", "text": ""})
                continue

            # Stream transcript to UI immediately (shows while LLM thinks)
            await ws.send_json({"type": "transcript", "text": transcript})

            t1 = time.perf_counter()

            # ── LLM ──────────────────────────────────────────────────────────
            llm_reply = chat(transcript)
            await ws.send_json({"type": "reply", "text": llm_reply})

            t2 = time.perf_counter()

            # ── TTS ──────────────────────────────────────────────────────────
            wav_bytes = synthesize(llm_reply)
            await ws.send_bytes(wav_bytes)   # raw WAV — client plays it

            t3 = time.perf_counter()

            # ── Latency report ───────────────────────────────────────────────
            print(
                f"[LATENCY] STT={t1-t0:.2f}s  LLM={t2-t1:.2f}s  "
                f"TTS={t3-t2:.2f}s  total={t3-t0:.2f}s"
            )
            await ws.send_json({
                "type": "latency",
                "stt":   round(t1 - t0, 3),
                "llm":   round(t2 - t1, 3),
                "tts":   round(t3 - t2, 3),
                "total": round(t3 - t0, 3),
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
