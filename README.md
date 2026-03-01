# Intent – Local, Context‑Aware Speech Transcription

Intent is a fully local live speech‑to‑text tool with live context compression, optional speaker diarization, and smart topic drift detection. It uses `faster-whisper` for transcription, writes live transcripts + metadata, and ships both a CLI and a PyQt GUI.

## Features
- Live microphone transcription with adaptive voice activity detection and overlap handling to avoid word loss.
- Rolling context: keeps recent verbatim text, compresses older context into a structured memory (summary, names/terms, open threads).
- Optional hallucination filtering and artifact cleanup to reduce Whisper loops.
- Optional speaker diarization with persistent profiles (SpeechBrain) and per‑utterance speaker tags.
- Topic tracking and off‑topic warnings (local Ollama + sentence-transformers).
- Optional Deepgram cloud fallback (Nova-3) for tough audio; optional Deepgram TTS warnings.
- PyQt GUI with session list, transcript view, and concept map widgets; dark/light styles.
- Outputs plain text transcript and JSON metadata per session.

## Requirements
- Python 3.10+ (tested on 3.11).
- FFmpeg installed on PATH (for torchaudio/sounddevice stability).
- For GPU: CUDA-supported PyTorch if you want `--backend-device cuda`; otherwise CPU works.
- Optional extras:
  - Speaker diarization: `torch`, `torchaudio`, `speechbrain` (already in `requirements.txt`, but CUDA wheels may be needed on Linux/Windows).
  - Topic tracking: `sentence-transformers` (in requirements).
  - Ollama (for local topic inference) running on localhost:11434.
  - Deepgram API key (for cloud transcription or TTS warnings).

## Install
```bash
python -m venv .venv
. .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start (CLI)
```bash
# List mics
python main.py --list-devices

# Start with defaults (medium.en, CPU, local only)
python main.py --device 3

# Faster startup / smaller model
python main.py --model tiny --backend-device cpu

# Use GPU (if available)
python main.py --backend-device cuda --compute-type float16

# Add glossary / names to bias accuracy
python main.py --notes "Acme, Q4 roadmap, Orion API, Sara Patel"
```
Outputs go to `transcripts/live_YYYYMMDD_HHMMSS.txt` plus a matching `.json` with chunk metadata, compression events, and settings.

Common flags:
- `--no-diarize` to disable speaker tracking, `--speakers-file speakers.json` to persist profiles.
- `--no-hallucination-filter` to keep all raw output.
- `--ollama-model llama3.2:1b` to choose a local LLM for topic tracking.
- `--deepgram-enabled --deepgram-api-key YOUR_KEY` to use Deepgram instead of local Whisper.

## Quick start (GUI)
```bash
python gui.py
```
The GUI wraps the same engine: start/stop, view transcripts, browse sessions, and visualize concepts. Settings dialog lets you toggle Ollama topic detection and Deepgram.

## How it works
- **Engine** (`main.py`): captures audio blocks (`sounddevice`), segments speech with adaptive RMS thresholds, pushes utterances to a worker thread that transcribes via `faster-whisper` (or Deepgram). Deduplicates chunk overlaps, filters hallucinations, and appends to a rolling transcript.
- **Context compression**: keeps recent verbatim text; older text is condensed into a four‑section memory (summary, names/terms, open threads, style notes) within a character budget.
- **Diarization**: optional SpeechBrain ECAPA embeddings with cosine matching and incremental profile updates.
- **Topic tracking**: optional; sentence-transformers embeddings + Ollama summarization to detect drift and warn.

## Model caching
Models download to `models/` (gitignored). First run may take a few minutes; subsequent runs start in seconds. Change model with `--model tiny|small|medium|large-v3|<local-path>`.

## Repo hygiene
`.gitignore` excludes virtualenvs, caches, transcripts, DBs, and downloaded models. Only source/config files are tracked.

## Collaborators (GitHub)
To add a collaborator on the `intent` repo:
1. GitHub UI: Repo → Settings → Collaborators → Add people → choose role (Write/Maintain).
2. CLI (after `gh auth login`): `gh repo add-collaborator <username> --permission write`.

## Troubleshooting
- Authentication failed on push: create a PAT with `repo` scope or use SSH, then `git push -u origin main`.
- Model download slow: try `--model tiny` first; ensure Hugging Face cache is reachable.
- Missing CUDA: switch to `--backend-device cpu` or install CUDA-enabled PyTorch.
- No mic access on macOS: grant Terminal/IDE microphone permission in System Settings → Privacy & Security.
