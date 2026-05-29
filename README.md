# 🎙️ Aria — Real-Time Voice Assistant (Customer Support)

Aria is a low-latency conversational voice assistant built entirely using open-source models running on JarvisLabs GPU infrastructure.

Designed for **ShopEasy customer support**, Aria can handle spoken customer queries in real time — listening, reasoning, and responding naturally through speech.

**Hold Space → Speak → Release → Hear Aria respond.**

---



# 🗣️ Sample Conversation

| Turn | Speaker | Text                                                                                                                                     |
| ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | User    | "Hi, I want to return a jacket I bought last week."                                                                                      |
| 1    | Aria    | "No problem! ShopEasy accepts returns within 30 days. Could you share your order number so I can get that started for you?"              |
| 2    | User    | "My order number is 98765."                                                                                                              |
| 2    | Aria    | "Got it! I've initiated the return for order 98765, and a prepaid shipping label will be sent to your registered email within the hour." |
| 3    | User    | "How long does the refund take?"                                                                                                         |
| 3    | Aria    | "Once we receive the item, refunds are typically processed within 5 to 7 business days back to your original payment method."            |

---

# 🚀 What It Does

Aria enables users to have natural spoken conversations with an AI-powered customer support assistant directly from the browser.

The system:

1. Captures live microphone audio
2. Converts speech to text using Whisper
3. Generates contextual responses using Mistral 7B
4. Converts responses back into speech using Piper TTS
5. Streams audio back to the browser in real time

The goal was to build a fully open-source voice assistant with low enough latency that the interaction feels conversational rather than sequential.

---

# ❓ Why I Built This

Most customer support systems still rely on chatbots or high-latency voice systems backed by proprietary APIs. I wanted to explore whether modern open-source models were now capable of delivering real-time conversational experiences end-to-end on commodity GPU hardware.

This project also allowed me to work across several challenging engineering domains simultaneously:

* Real-time audio streaming
* WebSockets
* GPU inference optimization
* Speech processing
* Async orchestration between AI models
* Latency-sensitive system design

Rather than simply chaining models together, the focus was on making the interaction feel responsive and natural for spoken conversation.

---

# 🧠 Pipeline Architecture

```text
Browser mic → PCM audio → WebSocket
                               ↓
                  [JarvisLabs GPU Instance]
                               ↓
               faster-whisper (Whisper Small)   ← STT
                               ↓
               Mistral 7B via Ollama            ← LLM
                               ↓
               Piper TTS (en_US-lessac-medium)  ← TTS
                               ↓
              WAV audio → WebSocket → Browser speaker
```

---

# ⚡ Latency Benchmarks

Measured on a JarvisLabs RTX 4090 instance.

| Stage                      | Avg Time   |
| -------------------------- | ---------- |
| STT — faster-whisper small | ~0.20s     |
| LLM — Mistral 7B (Ollama)  | ~0.55s     |
| TTS — Piper                | ~0.15s     |
| **End-to-End**             | **~0.90s** |

---

# ⚙️ What I Did to Reduce Latency

* Used **faster-whisper (CTranslate2)** instead of the original Whisper implementation (~5× faster inference)
* Enabled `float16` GPU inference
* Used VAD filtering to skip silence immediately
* Kept responses intentionally concise (1–2 spoken-friendly sentences)
* Used Piper TTS because it synthesizes audio in ~150ms
* Used persistent WebSocket connections instead of REST requests
* Streamed audio directly rather than saving intermediate files

---

# 🛍️ Grounded Use Case — ShopEasy Customer Support

Aria is specifically grounded as a customer support assistant for **ShopEasy**.

She can help users with:

* 📦 Order tracking
* 🔄 Returns and refunds
* 🚚 Shipping information
* 🔑 Account access issues
* 💳 Payments and invoices
* 📧 Escalation to human support

The assistant is intentionally constrained to customer-support-related topics to keep responses reliable and focused.

---

# 🛠️ Tech Stack

| Component  | Technology           |
| ---------- | -------------------- |
| ASR        | faster-whisper       |
| LLM        | Mistral 7B           |
| TTS        | Piper                |
| Backend    | FastAPI + WebSockets |
| Frontend   | HTML + JavaScript    |
| Deployment | JarvisLabs GPU       |

---

# ▶️ Setup & Run

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-assistant.git
cd voice-assistant
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Download Piper Voice

```bash
mkdir -p models

python -m piper \
  --download-dir ./models \
  en_US-lessac-medium
```

---

## 4. Install & Start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &
ollama pull mistral
```

---

## 5. Run the FastAPI Server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080
```

---

## 6. Open in Browser

Visit your JarvisLabs public URL on port `8080`.

### Controls

* Hold `Space` to talk
* Or click the orb button
* Release to send audio

---

# 📁 Project Structure

```text
voice-assistant/
├── server/
│   ├── main.py      # FastAPI + WebSocket server
│   ├── stt.py       # faster-whisper speech recognition
│   ├── llm.py       # Mistral 7B via Ollama
│   └── tts.py       # Piper text-to-speech
│
├── client/
│   └── index.html   # Browser UI
│
├── models/          # Downloaded Piper voices
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🤖 What I Used AI For

I used AI coding assistants primarily for:

* Initial FastAPI boilerplate generation
* Basic WebSocket scaffolding
* Browser microphone capture setup
* README formatting and iteration
* Debugging repetitive integration issues

The core system orchestration, latency optimization decisions, streaming pipeline design, async coordination, and architecture decisions were implemented manually.

I also overrode several AI-generated suggestions:

* Replaced polling-based communication with persistent WebSockets
* Avoided larger Whisper models because latency mattered more than marginal accuracy gains
* Simplified frontend interactions to prioritize responsiveness over UI complexity

Most of the engineering effort ultimately went into making independently functioning models behave like a cohesive real-time conversational system.

---

# 🔮 What I’d Improve With 4 More Weeks

If shipping this to real users, I would focus heavily on streaming and production reliability improvements.

### Highest-priority upgrades:

* True streaming TTS (generate speech sentence-by-sentence while the LLM is still decoding)
* Wake-word detection
* Voice activity detection (VAD)
* Interruptible speech playback
* Persistent conversation memory
* Real order-management API integration
* Multi-user session handling
* Dockerized deployment
* Observability dashboards for latency/GPU monitoring
* Better frontend UX and accessibility
* Mobile support
* Multi-language support

I would also benchmark smaller reasoning models to further optimize the latency-quality tradeoff for consumer GPUs.

---

# 🚢 Deployment

The complete system was deployed on JarvisLabs GPU infrastructure.

JarvisLabs was used for:

* Whisper inference
* Mistral 7B inference
* Piper TTS synthesis
* Hosting the FastAPI application

---

# 🙏 Acknowledgements

* faster-whisper
* Ollama
* Piper TTS
* FastAPI
* JarvisLabs

---
