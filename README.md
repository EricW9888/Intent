# Intent – Local, Context-Aware Speech Transcription

Intent is a local-first speech-to-text tool with live context compression, optional speaker diarization, topic drift detection, and a PyQt GUI. It defaults to the open-source Whisper models via `faster-whisper` and can optionally use Gemini or local Ollama for topic detection. No cloud calls are required unless you opt into Gemini or Deepgram.

## Features
- Live mic transcription with adaptive VAD and overlap handling to avoid boundary word loss.
- Rolling context: keeps recent verbatim text; compresses older text into summary, key terms, and open threads within a small budget.
- Optional hallucination filtering and artifact cleanup to reduce Whisper repetition.
- Optional speaker diarization with persistent profiles (SpeechBrain ECAPA).
- Topic tracking and off-topic detection using your chosen AI provider (Gemini or local Ollama).
- Optional Deepgram fallback for transcription and optional Deepgram TTS warnings.
- PyQt GUI: session/folder tree, transcript view, concept map tab, settings for AI provider and Deepgram.
- Outputs plain text transcripts and JSON metadata per session.

## Requirements
- Python 3.10+ (tested on 3.11).
- FFmpeg available on PATH (needed by sounddevice/torchaudio).
- Whisper models download on first use and are cached in `models/` (gitignored).
- Optional: CUDA-enabled PyTorch if you want GPU (`--backend-device cuda`).
- Optional: Ollama running on `localhost:11434` for local topic detection.
- Optional: Gemini API key if you choose Gemini for topic detection.
- Optional: Deepgram API key for cloud transcription or TTS warnings.

## Install
```bash
python -m venv .venv
. .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start (CLI)
```bash
# List microphone devices
python main.py --list-devices

# Start with defaults (small.en, CPU)
python main.py --device 3

# Faster startup / smaller model
python main.py --model tiny --backend-device cpu

# Use GPU if available
python main.py --backend-device cuda --compute-type float16

# Add domain glossary to bias names/terms
python main.py --notes "Acme, Orion API, Sara Patel"
```
Outputs land in `transcripts/live_YYYYMMDD_HHMMSS.txt` plus matching `.json`.

Useful flags:
- `--no-diarize` to disable speaker tracking; `--speakers-file speakers.json` to persist profiles.
- `--no-hallucination-filter` to keep raw output.
- `--llm-provider gemini --gemini-api-key YOUR_KEY` to use Gemini for topic detection.
- `--llm-provider ollama --ollama-model llama3.2:1b` to use a local Ollama model.
- `--deepgram-enabled --deepgram-api-key YOUR_KEY` to use Deepgram instead of local Whisper.

## Quick start (GUI)
```bash
python gui.py
```
Use Settings to choose AI provider (Gemini or local Ollama) and toggle Deepgram. The GUI lets you start/stop sessions, view transcripts, generate concept maps, and browse sessions/folders.

## How it works
- **Engine (main.py)**: captures audio blocks, segments speech with adaptive RMS thresholds, transcribes via faster-whisper (or Deepgram), deduplicates overlaps, filters hallucinations, and appends to a rolling transcript. Older context is compressed into a structured memory.
- **Diarization**: optional SpeechBrain ECAPA embeddings with cosine matching and incremental profile updates.
- **Topic detection**: optional; uses sentence-transformers embeddings plus your chosen AI provider (Gemini or Ollama) to infer/refresh topics and warn on drift.
- **Concept maps**: transcripts can be turned into concept graphs in the GUI.

## Repo hygiene
`.gitignore` excludes venvs, caches, transcripts, databases, and downloaded models. Source and config files are tracked; heavy artifacts are not.

## Troubleshooting
- Model download slow: try `--model tiny` first; ensure Hugging Face cache access.
- Gemini selected but no key: set `--gemini-api-key` or switch provider to Ollama; otherwise topic detection falls back to heuristics.
- GPU issues: switch to CPU (`--backend-device cpu`) or install CUDA-enabled PyTorch.
- Mic permission on macOS: grant the Terminal/IDE microphone access in System Settings → Privacy & Security.
