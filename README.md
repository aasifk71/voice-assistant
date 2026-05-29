# Real-Time Voice Assistant

A low-latency voice assistant built entirely using open-source models running locally on GPU infrastructure.

The system captures live microphone audio from the browser, transcribes speech using Whisper, generates responses with Mistral 7B, and converts responses back into speech using Piper TTS — all in under one second of average latency. The goal was to create a fully local alternative to cloud-based assistants while maintaining natural conversational responsiveness.

---

# What It Does

This project enables real-time voice conversations with an AI assistant directly from the browser.

Users can hold a key or click a button to speak naturally, receive near-instant AI responses, and hear synthesized speech streamed back in real time. Unlike most commercial assistants, the entire pipeline runs on open-source models without depending on proprietary APIs.

---

# Why I Built This

Most voice assistants either rely heavily on cloud APIs or suffer from noticeable latency that breaks conversational flow. I wanted to explore whether modern open-source models were mature enough to deliver a genuinely responsive voice experience on commodity GPU hardware.

This project also gave me the opportunity to work across multiple real-time systems domains simultaneously: low-latency audio streaming, WebSockets, GPU inference optimization, speech processing, and orchestration between independent AI models. I specifically chose this problem because it combines systems engineering with applied AI in a way that feels close to real production infrastructure.

---

# How to Run It

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-assistant.git
cd voice-assistant
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install piper-tts
```

---

## 3. Configure Environment Variables

```bash
cp .env.example .env
```

Update `.env` with your preferred model paths and inference backend.

---

## 4. Start the LLM Backend

### Option A — Ollama

```bash
ollama pull mistral
ollama serve
```

### Option B — vLLM (Recommended)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dtype float16
```

---

## 5. Download a Piper Voice

```bash
mkdir -p models

python -m piper \
  --download-dir ./models \
  en_US-lessac-medium
```

---

## 6. Start the FastAPI Server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080
```

Open:

```text
http://localhost:8080
```

Hold `Space` or click the orb to talk.

---

# Architecture Decisions

## WebSockets Instead of REST

I used WebSockets for bidirectional audio streaming because voice interaction requires continuous low-latency communication. REST would introduce unnecessary request overhead and make real-time streaming significantly harder.

---

## Faster-Whisper Over OpenAI Whisper API

I chose Faster-Whisper because it runs locally, has lower latency, and avoids recurring API costs. Since the goal was a fully open-source local assistant, depending on a hosted STT API would defeat the purpose.

---

## vLLM for LLM Inference

I used vLLM instead of raw Hugging Face transformers because it provides significantly faster token generation and optimized GPU memory handling. This reduced overall response latency while making larger models feasible on limited VRAM.

---

## Piper Instead of Cloud TTS

Most cloud TTS services sound great but introduce network latency and API dependency. Piper offered an excellent tradeoff between speed, offline capability, and acceptable voice quality.

---

## Push-to-Talk Instead of Always Listening

I intentionally started with push-to-talk interaction because continuous wake-word systems add substantial complexity around VAD, false activations, and streaming pipelines. This kept the initial system simpler and more reliable.

---

# What I Used AI For

I used AI tools primarily for boilerplate acceleration and debugging assistance.

AI helped generate:

* Initial FastAPI WebSocket scaffolding
* Basic browser audio capture code
* Some repetitive setup/configuration commands
* README structure drafts

The core orchestration logic, latency optimization decisions, streaming pipeline design, and system integration were written and refined manually.

I also overrode several AI-generated suggestions:

* I removed polling-based communication in favor of persistent WebSockets because polling introduced avoidable latency.
* I avoided oversized Whisper models despite AI recommendations for accuracy, prioritizing responsiveness instead.
* I simplified the frontend interaction model after AI-generated UI suggestions added unnecessary complexity.

Most of the engineering effort ultimately went into connecting independent components into a stable low-latency pipeline rather than generating isolated code snippets.

---

# What I Would Change With 4 More Weeks

If I were shipping this to real users, I would focus heavily on streaming and reliability improvements.

The biggest upgrade would be true streaming speech synthesis — generating audio sentence-by-sentence while the LLM is still responding. This would significantly reduce perceived latency and make conversations feel much more natural.

Other major improvements would include:

* Wake-word detection for hands-free usage
* Voice activity detection (VAD)
* Interruptible speech generation
* Persistent conversation memory
* Multi-user session handling
* Dockerized deployment
* Better frontend UX and accessibility
* Multi-language support
* Observability dashboards for latency monitoring

I would also benchmark smaller instruction-tuned models to optimize the latency/quality tradeoff further for consumer GPUs.

---
