from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

DEFAULT_MODEL = "small.en"
DEFAULT_BACKEND_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "default"
DEFAULT_CPU_THREADS = 0
DEFAULT_BEAM_SIZE = 1
DEFAULT_BEST_OF = 3
DEFAULT_PATIENCE = 1.0
DEFAULT_SPEAKER_THRESHOLD = 0.45
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_BLOCK_MS = 30
DEFAULT_CALIBRATION_SECONDS = 1.5
DEFAULT_SILENCE_SECONDS = 0.6
DEFAULT_PRE_ROLL_SECONDS = 0.5
DEFAULT_MAX_UTTERANCE_SECONDS = 24.0
DEFAULT_FORCE_SPLIT_OVERLAP_SECONDS = 0.75
DEFAULT_MIN_UTTERANCE_SECONDS = 0.25
DEFAULT_ENERGY_FLOOR = 0.003
DEFAULT_SPEECH_RATIO = 1.8
DEFAULT_EXACT_CONTEXT_CHARS = 6_000
DEFAULT_RECENT_CONTEXT_CHARS = 2_500
DEFAULT_MEMORY_BUDGET_CHARS = 1_600
DEFAULT_PROMPT_TAIL_CHARS = 700
DEFAULT_PROMPT_BUDGET_CHARS = 1_700
DEFAULT_HALLUCINATION_FILTER = True

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CAMEL_OR_CODE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+|[a-z]+_[a-z0-9_]+|[a-z]+-[a-z0-9-]+)\b")
PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:\d+)?\b")
NUMBER_PHRASE_RE = re.compile(r"\b(?:\d+[A-Za-z%/-]*|[A-Za-z]+\d+[A-Za-z0-9-]*)\b")
URL_RE = re.compile(r"\b(?:https?://\S+|www\.\S+|\S+@\S+)\b")
SECTION_HEADER_RE = re.compile(r"(?im)^(SUMMARY|NAMES_AND_TERMS|OPEN_THREADS|STYLE_NOTES):\s*")

ARTIFACT_PREFIX_RE = re.compile(
    r"^(?:"
    r"[<>←→¯∫]+[-\s]*"
    r"|[A-Z]{2,}[A-Za-z0-9]*[-]\s*"
    r"|[A-Z][A-Za-z]+-(?=\s*[A-Z])"
    r")+\s*"
)

STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "but", "by", "can", "could", "did", "do", "does", "doing", "for",
    "from", "get", "got", "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "like", "me", "more", "my", "now", "of",
    "on", "or", "our", "out", "re", "really", "right", "said", "she", "so", "some", "that", "the",
    "their", "them", "there", "they", "this", "to", "too", "uh", "um", "up", "was", "we", "well",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "yeah", "you",
    "your",
}


@dataclass
class AppConfig:
    model: str
    backend_device: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    best_of: int
    patience: float
    output_path: Path
    metadata_path: Path
    language: str | None
    notes: str
    device: int | None
    sample_rate: int
    block_ms: int
    calibration_seconds: float
    silence_seconds: float
    pre_roll_seconds: float
    max_utterance_seconds: float
    force_split_overlap_seconds: float
    min_utterance_seconds: float
    energy_floor: float
    speech_ratio: float
    exact_context_chars: int
    recent_context_chars: int
    memory_budget_chars: int
    prompt_tail_chars: int
    prompt_budget_chars: int
    hallucination_filter: bool
    diarize: bool
    speaker_threshold: float
    speakers_file: Path | None
    ollama_model: str
    warmup_only: bool
    verbose_model: bool


@dataclass
class CompressionEvent:
    utterance_index: int
    archived_chars: int
    kept_recent_chars: int
    compressed_memory_chars: int

    def to_dict(self) -> dict[str, int]:
        return {
            "utterance_index": self.utterance_index,
            "archived_chars": self.archived_chars,
            "kept_recent_chars": self.kept_recent_chars,
            "compressed_memory_chars": self.compressed_memory_chars,
        }


@dataclass
class TranscriptContext:
    compressed_memory: str = ""
    recent_segments: list[str] = field(default_factory=list)
    compression_events: list[CompressionEvent] = field(default_factory=list)

    def recent_char_count(self) -> int:
        return sum(len(segment) for segment in self.recent_segments)

    def recent_tail(self, budget: int) -> str:
        pieces: list[str] = []
        char_count = 0
        for segment in reversed(self.recent_segments):
            pieces.append(segment)
            char_count += len(segment)
            if char_count >= budget:
                break
        return "\n\n".join(reversed(pieces))[-budget:]


@dataclass(frozen=True)
class TokenSpan:
    token: str
    end: int


@dataclass
class Utterance:
    index: int
    audio: np.ndarray
    start_seconds: float
    end_seconds: float
    forced_split: bool


@dataclass
class UtteranceRecord:
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    raw_text: str
    forced_split: bool
    speaker: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "text": self.text,
            "raw_text": self.raw_text,
            "forced_split": self.forced_split,
        }
        if self.speaker:
            result["speaker"] = self.speaker
        return result


@dataclass
class SessionState:
    started_at: datetime
    transcript: str = ""
    context: TranscriptContext = field(default_factory=TranscriptContext)
    utterances: list[UtteranceRecord] = field(default_factory=list)
    last_speaker: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class SpeakerProfile:
    label: str
    embedding: np.ndarray
    sample_count: int


class SpeakerTracker:
    def __init__(self, threshold: float, profiles_path: Path | None):
        self.threshold = threshold
        self.profiles_path = profiles_path
        self.speakers: list[SpeakerProfile] = []
        self._classifier: Any = None
        if profiles_path and profiles_path.exists():
            self._load_profiles()

    def _ensure_model(self) -> None:
        if self._classifier is not None:
            return
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]
        from speechbrain.inference.speaker import EncoderClassifier

        model_dir = Path(__file__).resolve().parent / "models" / "speaker_encoder"
        self._classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
        )
        print("Speaker encoder ready.", file=sys.stderr)

    def identify(self, audio: np.ndarray, sample_rate: int) -> str:
        self._ensure_model()
        import torch

        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            embedding = self._classifier.encode_batch(signal).squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        best_sim = -1.0
        best_idx = -1
        for i, sp in enumerate(self.speakers):
            sim = float(np.dot(embedding, sp.embedding))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold and best_idx >= 0:
            sp = self.speakers[best_idx]
            alpha = min(0.3, 1.0 / (sp.sample_count + 1))
            sp.embedding = (1 - alpha) * sp.embedding + alpha * embedding
            sp.embedding /= np.linalg.norm(sp.embedding) + 1e-8
            sp.sample_count += 1
            return sp.label

        label = f"Speaker {len(self.speakers) + 1}"
        self.speakers.append(SpeakerProfile(label=label, embedding=embedding, sample_count=1))
        return label

    def _load_profiles(self) -> None:
        if not self.profiles_path:
            return
        try:
            data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
            for entry in data:
                self.speakers.append(SpeakerProfile(
                    label=entry["label"],
                    embedding=np.array(entry["embedding"], dtype=np.float32),
                    sample_count=entry.get("sample_count", 1),
                ))
            print(f"Loaded {len(self.speakers)} speaker profile(s) from {self.profiles_path}", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: could not load speaker profiles: {exc}", file=sys.stderr)

    def save_profiles(self) -> None:
        if not self.profiles_path:
            return
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"label": sp.label, "embedding": sp.embedding.tolist(), "sample_count": sp.sample_count}
            for sp in self.speakers
        ]
        self.profiles_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class AudioSegmenter:
    def __init__(self, config: AppConfig):
        self.config = config
        self.block_size = int(config.sample_rate * config.block_ms / 1000)
        self.pre_roll_blocks = max(1, int(round(config.pre_roll_seconds * 1000 / config.block_ms)))
        self.max_blocks = max(1, int(round(config.max_utterance_seconds * 1000 / config.block_ms)))
        self.overlap_blocks = max(1, int(round(config.force_split_overlap_seconds * 1000 / config.block_ms)))
        self.silence_blocks = max(1, int(round(config.silence_seconds * 1000 / config.block_ms)))
        self.start_blocks = 2
        self.min_blocks = max(1, int(round(config.min_utterance_seconds * 1000 / config.block_ms)))

        self.pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_blocks)
        self.active_blocks: list[np.ndarray] = []
        self.active = False
        self.speech_run = 0
        self.silence_run = 0
        self.total_samples = 0
        self.utterance_start_sample = 0
        self.noise_floor = config.energy_floor
        self.calibration_rms: list[float] = []
        self.calibration_samples_target = int(config.calibration_seconds * config.sample_rate)
        self.utterance_index = 0

    def _rms(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

    def _speech_threshold(self) -> float:
        return max(self.config.energy_floor, self.noise_floor * self.config.speech_ratio)

    def _update_noise_floor(self, rms: float) -> None:
        if self.total_samples <= self.calibration_samples_target:
            self.calibration_rms.append(rms)
            if self.total_samples >= self.calibration_samples_target and self.calibration_rms:
                self.noise_floor = max(self.config.energy_floor, float(np.percentile(self.calibration_rms, 65)))
            return
        if not self.active:
            self.noise_floor = max(self.config.energy_floor, (self.noise_floor * 0.98) + (rms * 0.02))

    def _emit(self, blocks: list[np.ndarray], end_sample: int, forced_split: bool) -> Utterance | None:
        if len(blocks) < self.min_blocks:
            return None
        audio = np.concatenate(blocks).astype(np.float32, copy=False)
        self.utterance_index += 1
        return Utterance(
            index=self.utterance_index,
            audio=audio,
            start_seconds=self.utterance_start_sample / self.config.sample_rate,
            end_seconds=end_sample / self.config.sample_rate,
            forced_split=forced_split,
        )

    def process_block(self, block: np.ndarray) -> list[Utterance]:
        outputs: list[Utterance] = []
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        rms = self._rms(block)
        threshold = self._speech_threshold()
        is_speech = rms >= threshold

        self.total_samples += len(block)
        self._update_noise_floor(rms)

        if not self.active:
            self.pre_roll.append(block.copy())
            if is_speech:
                self.speech_run += 1
            else:
                self.speech_run = 0
            if self.speech_run >= self.start_blocks:
                self.active = True
                self.silence_run = 0
                self.active_blocks = list(self.pre_roll)
                start_offset = len(self.active_blocks) * self.block_size
                self.utterance_start_sample = max(0, self.total_samples - start_offset)
                self.pre_roll.clear()
            return outputs

        self.active_blocks.append(block.copy())
        if is_speech:
            self.silence_run = 0
        else:
            self.silence_run += 1

        if len(self.active_blocks) >= self.max_blocks:
            overlap = min(self.overlap_blocks, max(1, len(self.active_blocks) // 4))
            emit_blocks = self.active_blocks[:-overlap] if len(self.active_blocks) > overlap else self.active_blocks[:]
            keep_blocks = self.active_blocks[-overlap:]
            end_sample = self.total_samples - (len(keep_blocks) * self.block_size)
            utterance = self._emit(emit_blocks, end_sample, forced_split=True)
            if utterance is not None:
                outputs.append(utterance)
            self.active_blocks = [block.copy() for block in keep_blocks]
            self.utterance_start_sample = max(0, end_sample - len(self.active_blocks) * self.block_size)
            self.silence_run = 0
            return outputs

        if self.silence_run >= self.silence_blocks:
            utterance = self._emit(self.active_blocks[:], self.total_samples, forced_split=False)
            if utterance is not None:
                outputs.append(utterance)
            self.active = False
            self.active_blocks = []
            self.pre_roll.clear()
            self.speech_run = 0
            self.silence_run = 0

        return outputs

    def flush(self) -> Utterance | None:
        if not self.active_blocks:
            return None
        utterance = self._emit(self.active_blocks[:], self.total_samples, forced_split=False)
        self.active = False
        self.active_blocks = []
        self.pre_roll.clear()
        self.speech_run = 0
        self.silence_run = 0
        return utterance


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Fully local cross-platform live speech-to-text using Whisper on your microphone."
    )
    parser.add_argument("--list-devices", action="store_true", help="List microphone devices and exit.")
    parser.add_argument("--device", help="Input device index or a substring of the input device name.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name or local path. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--backend-device",
        default=DEFAULT_BACKEND_DEVICE,
        choices=["auto", "cpu", "cuda"],
        help=f"Inference device for the local model. Default: {DEFAULT_BACKEND_DEVICE}",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"Model compute type for faster-whisper, for example default, float32, float16, int8. Default: {DEFAULT_COMPUTE_TYPE}",
    )
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="CPU threads for faster-whisper. 0 lets the runtime choose.")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="Beam size. Higher is slower but can improve accuracy.")
    parser.add_argument("--best-of", type=int, default=DEFAULT_BEST_OF, help="Sampling fallback candidates when beam search falls back to temperature > 0.")
    parser.add_argument("--patience", type=float, default=DEFAULT_PATIENCE, help="Beam search patience.")
    parser.add_argument("--language", help="Optional language code such as 'en'.")
    parser.add_argument("--notes", default="", help="Optional glossary of names, jargon, or product terms.")
    parser.add_argument("--notes-file", type=Path, help="Text file with names or jargon to bias transcription.")
    parser.add_argument("--output", type=Path, help="Transcript output path.")
    parser.add_argument("--metadata", type=Path, help="Metadata JSON output path.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Microphone capture sample rate.")
    parser.add_argument("--block-ms", type=int, default=DEFAULT_BLOCK_MS, help="Audio block size in milliseconds.")
    parser.add_argument("--calibration-seconds", type=float, default=DEFAULT_CALIBRATION_SECONDS, help="Ambient noise calibration window.")
    parser.add_argument("--silence-seconds", type=float, default=DEFAULT_SILENCE_SECONDS, help="Silence needed to end an utterance.")
    parser.add_argument("--pre-roll-seconds", type=float, default=DEFAULT_PRE_ROLL_SECONDS, help="Audio kept before speech starts.")
    parser.add_argument("--max-utterance-seconds", type=float, default=DEFAULT_MAX_UTTERANCE_SECONDS, help="Force-split long speech after this many seconds.")
    parser.add_argument("--force-split-overlap-seconds", type=float, default=DEFAULT_FORCE_SPLIT_OVERLAP_SECONDS, help="Overlap retained across forced splits.")
    parser.add_argument("--min-utterance-seconds", type=float, default=DEFAULT_MIN_UTTERANCE_SECONDS, help="Minimum speech length before transcription.")
    parser.add_argument("--energy-floor", type=float, default=DEFAULT_ENERGY_FLOOR, help="Minimum RMS threshold for speech detection.")
    parser.add_argument("--speech-ratio", type=float, default=DEFAULT_SPEECH_RATIO, help="How far above the noise floor speech must be.")
    parser.add_argument("--exact-context-chars", type=int, default=DEFAULT_EXACT_CONTEXT_CHARS, help="Exact transcript chars kept before compression.")
    parser.add_argument("--recent-context-chars", type=int, default=DEFAULT_RECENT_CONTEXT_CHARS, help="Exact transcript chars kept after compression.")
    parser.add_argument("--memory-budget-chars", type=int, default=DEFAULT_MEMORY_BUDGET_CHARS, help="Compressed context memory budget.")
    parser.add_argument("--prompt-tail-chars", type=int, default=DEFAULT_PROMPT_TAIL_CHARS, help="Recent exact transcript chars passed to new utterances.")
    parser.add_argument("--prompt-budget-chars", type=int, default=DEFAULT_PROMPT_BUDGET_CHARS, help="Overall prompt budget passed into Whisper.")
    parser.add_argument("--no-hallucination-filter", action="store_true", help="Disable hallucination detection and filtering.")
    parser.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization.")
    parser.add_argument("--speaker-threshold", type=float, default=DEFAULT_SPEAKER_THRESHOLD, help="Cosine similarity threshold for matching speakers. Lower = more aggressive merging.")
    parser.add_argument("--speakers-file", type=Path, help="JSON file to load/save speaker profiles for recognition across sessions.")
    parser.add_argument("--ollama-model", default="llama3.2:1b", help="Ollama model for topic inference. Default: llama3.2:1b")
    parser.add_argument("--warmup-only", action="store_true", help="Download and load the model, then exit.")
    parser.add_argument("--verbose-model", action="store_true", help="Show faster-whisper progress output.")
    args = parser.parse_args()

    notes = args.notes.strip()
    if args.notes_file:
        notes_file = args.notes_file.expanduser().resolve()
        notes = "\n".join(part for part in [notes, notes_file.read_text(encoding="utf-8").strip()] if part).strip()

    output_path, metadata_path = default_output_paths(args.output, args.metadata)
    device = resolve_input_device(args.device) if args.device else None

    config = AppConfig(
        model=args.model,
        backend_device=args.backend_device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=args.patience,
        output_path=output_path,
        metadata_path=metadata_path,
        language=args.language,
        notes=notes,
        device=device,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        calibration_seconds=args.calibration_seconds,
        silence_seconds=args.silence_seconds,
        pre_roll_seconds=args.pre_roll_seconds,
        max_utterance_seconds=args.max_utterance_seconds,
        force_split_overlap_seconds=args.force_split_overlap_seconds,
        min_utterance_seconds=args.min_utterance_seconds,
        energy_floor=args.energy_floor,
        speech_ratio=args.speech_ratio,
        exact_context_chars=args.exact_context_chars,
        recent_context_chars=args.recent_context_chars,
        memory_budget_chars=args.memory_budget_chars,
        prompt_tail_chars=args.prompt_tail_chars,
        prompt_budget_chars=args.prompt_budget_chars,
        hallucination_filter=not args.no_hallucination_filter,
        diarize=not args.no_diarize,
        speaker_threshold=args.speaker_threshold,
        speakers_file=args.speakers_file.expanduser().resolve() if args.speakers_file else None,
        ollama_model=args.ollama_model,
        warmup_only=args.warmup_only,
        verbose_model=args.verbose_model,
    )

    if args.list_devices:
        list_input_devices()
        raise SystemExit(0)

    validate_config(config)
    return config


def default_output_paths(output: Path | None, metadata: Path | None) -> tuple[Path, Path]:
    if output:
        output_path = output.expanduser().resolve()
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_path = (Path.cwd() / "transcripts" / f"live_{stamp}.txt").resolve()

    if metadata:
        metadata_path = metadata.expanduser().resolve()
    else:
        metadata_path = output_path.with_suffix(".json")

    return output_path, metadata_path


def resolve_input_device(device_spec: str) -> int:
    devices = sd.query_devices()
    if device_spec.isdigit():
        index = int(device_spec)
        try:
            device = devices[index]
        except Exception as exc:
            raise SystemExit(f"Input device {index} not found.") from exc
        if int(device["max_input_channels"]) <= 0:
            raise SystemExit(f"Device {index} is not an input device.")
        return index

    lowered = device_spec.lower()
    matches = [
        index
        for index, device in enumerate(devices)
        if lowered in str(device["name"]).lower() and int(device["max_input_channels"]) > 0
    ]
    if not matches:
        raise SystemExit(f"No input device matched {device_spec!r}.")
    if len(matches) > 1:
        joined = ", ".join(f"{index}:{devices[index]['name']}" for index in matches)
        raise SystemExit(f"Multiple input devices matched {device_spec!r}: {joined}")
    return matches[0]


def list_input_devices() -> None:
    default_input, _ = sd.default.device
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) <= 0:
            continue
        marker = "*" if index == default_input else " "
        print(f"{marker} {index}: {device['name']} ({int(device['max_input_channels'])} in)")


def validate_config(config: AppConfig) -> None:
    if config.cpu_threads < 0:
        raise SystemExit("--cpu-threads must be >= 0.")
    if config.beam_size <= 0 or config.best_of <= 0:
        raise SystemExit("--beam-size and --best-of must be positive.")
    if config.patience <= 0:
        raise SystemExit("--patience must be positive.")
    if config.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive.")
    if config.block_ms <= 0:
        raise SystemExit("--block-ms must be positive.")
    if config.silence_seconds <= 0 or config.max_utterance_seconds <= 0:
        raise SystemExit("Silence and utterance durations must be positive.")
    if config.force_split_overlap_seconds <= 0:
        raise SystemExit("--force-split-overlap-seconds must be positive.")
    if config.force_split_overlap_seconds >= config.max_utterance_seconds:
        raise SystemExit("--force-split-overlap-seconds must be smaller than --max-utterance-seconds.")
    if config.min_utterance_seconds <= 0:
        raise SystemExit("--min-utterance-seconds must be positive.")
    if config.recent_context_chars > config.exact_context_chars:
        raise SystemExit("--recent-context-chars must be <= --exact-context-chars.")
    if config.prompt_tail_chars > config.exact_context_chars:
        raise SystemExit("--prompt-tail-chars must be <= --exact-context-chars.")
    if config.memory_budget_chars < 300:
        raise SystemExit("--memory-budget-chars is too small to be useful.")
    if config.prompt_budget_chars < 300:
        raise SystemExit("--prompt-budget-chars is too small to be useful.")


def load_local_model(config: AppConfig) -> WhisperModel:
    download_root = Path(__file__).resolve().parent / "models"
    download_root.mkdir(exist_ok=True)
    print(f"Loading local model: {config.model} (cache: {download_root})", file=sys.stderr)
    model = WhisperModel(
        config.model,
        device=config.backend_device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
        num_workers=1,
        download_root=str(download_root),
    )
    print("Model ready.", file=sys.stderr)
    return model


def build_prompt(context: TranscriptContext, config: AppConfig) -> str | None:
    parts: list[str] = []
    if config.notes:
        parts.append("Important names and jargon:\n" + config.notes[:500].strip())
    if context.compressed_memory:
        parts.append("Compressed memory:\n" + context.compressed_memory[: config.memory_budget_chars].strip())
    recent_tail = context.recent_tail(config.prompt_tail_chars)
    if recent_tail:
        parts.append("Recent transcript tail:\n" + recent_tail.strip())
    prompt = "\n\n".join(part for part in parts if part).strip()
    if not prompt:
        return None
    return prompt[-config.prompt_budget_chars :]


def tokenize_with_spans(text: str) -> list[TokenSpan]:
    return [TokenSpan(token=match.group(0).lower(), end=match.end()) for match in WORD_RE.finditer(text)]


def trim_overlapping_prefix(previous_text: str, current_text: str, min_words: int = 8, max_words: int = 80) -> str:
    previous_tokens = tokenize_with_spans(previous_text)
    current_tokens = tokenize_with_spans(current_text)
    max_match = min(max_words, len(previous_tokens), len(current_tokens))
    for word_count in range(max_match, min_words - 1, -1):
        if [token.token for token in previous_tokens[-word_count:]] == [token.token for token in current_tokens[:word_count]]:
            return current_text[current_tokens[word_count - 1].end :].lstrip()
    return current_text


def merge_transcript(existing: str, new_text: str) -> str:
    new_text = new_text.strip()
    if not new_text:
        return existing
    if not existing:
        return new_text
    if existing.endswith(("-", "--", "\u2014")):
        return existing + new_text
    if new_text[0] in ".,!?;:)":
        return existing + new_text
    return existing + " " + new_text


def normalize_memory_source(text: str) -> str:
    return SECTION_HEADER_RE.sub("", text).replace("- ", "")


def split_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    return [piece for piece in pieces if len(piece) >= 8]


def extract_terms(text: str, limit: int = 32) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip(" ,.;:!?()[]{}\"'")
        key = cleaned.lower()
        if not cleaned or len(cleaned) < 2:
            return
        if key in seen:
            return
        if key in {"summary", "names_and_terms", "open_threads", "style_notes"}:
            return
        seen.add(key)
        ordered.append(cleaned)

    for pattern in (URL_RE, ACRONYM_RE, CAMEL_OR_CODE_RE, PROPER_NOUN_RE, NUMBER_PHRASE_RE):
        for match in pattern.finditer(text):
            add(match.group(0))

    tokens = [match.group(0) for match in WORD_RE.finditer(text)]
    lower_counts = Counter(token.lower() for token in tokens if len(token) >= 4 and token.lower() not in STOPWORDS)
    first_seen: dict[str, str] = {}
    for token in tokens:
        key = token.lower()
        first_seen.setdefault(key, token)
    for token, count in lower_counts.most_common(limit * 2):
        if count < 2:
            continue
        add(first_seen[token])
        if len(ordered) >= limit:
            break

    return ordered[:limit]


def sentence_score(sentence: str, key_terms: list[str], position: float = 0.5) -> float:
    score = position * 2.0
    if any(char.isdigit() for char in sentence):
        score += 3.0
    if "?" in sentence:
        score += 2.0
    if ACRONYM_RE.search(sentence):
        score += 1.6
    if PROPER_NOUN_RE.search(sentence):
        score += 1.6
    if URL_RE.search(sentence):
        score += 2.0
    sentence_len = len(sentence)
    if 40 <= sentence_len <= 220:
        score += 1.0
    lower = sentence.lower()
    score += min(sum(1 for term in key_terms[:16] if term.lower() in lower) * 0.35, 3.0)
    return score + min(len(WORD_RE.findall(sentence)) / 18.0, 2.0)


def render_memory(summary_items: list[str], terms: list[str], open_threads: list[str]) -> str:
    summary_text = "\n".join(f"- {item}" for item in summary_items) if summary_items else "- No persistent summary yet."
    terms_text = ", ".join(terms) if terms else "none"
    threads_text = "\n".join(f"- {item}" for item in open_threads) if open_threads else "- No unresolved threads."
    style_notes = "- Preserve names, numbers, acronyms, and unfinished thoughts exactly."
    return (
        "SUMMARY:\n"
        f"{summary_text}\n\n"
        "NAMES_AND_TERMS:\n"
        f"{terms_text}\n\n"
        "OPEN_THREADS:\n"
        f"{threads_text}\n\n"
        "STYLE_NOTES:\n"
        f"{style_notes}"
    ).strip()


def compress_text_memory(existing_memory: str, archived_text: str, budget: int) -> str:
    source = "\n".join(part for part in [normalize_memory_source(existing_memory), archived_text] if part.strip()).strip()
    if not source:
        return ""

    key_terms = extract_terms(source, limit=40)
    sentences = split_sentences(source)
    scored: list[tuple[float, int, str]] = []
    seen_sentences: set[str] = set()
    for index, sentence in enumerate(sentences):
        normalized = re.sub(r"\s+", " ", sentence.lower())
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        position = index / max(1, len(sentences) - 1)
        scored.append((sentence_score(sentence, key_terms, position), index, sentence))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:8]
    summary_items = [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]

    archived_sentences = split_sentences(archived_text)
    open_candidates = [sentence for sentence in archived_sentences if "?" in sentence or len(sentence) > 24]
    open_threads = open_candidates[-4:] if open_candidates else archived_sentences[-3:]

    while True:
        memory = render_memory(summary_items, key_terms, open_threads)
        if len(memory) <= budget:
            return memory
        if len(summary_items) > 4:
            summary_items.pop(0)
            continue
        if len(key_terms) > 14:
            key_terms.pop()
            continue
        if len(open_threads) > 1:
            open_threads.pop(0)
            continue
        return memory[:budget].rsplit("\n", 1)[0].strip() or memory[:budget].strip()


def maybe_roll_context(context: TranscriptContext, config: AppConfig, utterance_index: int) -> None:
    if context.recent_char_count() <= config.exact_context_chars:
        return

    kept_segments: list[str] = []
    kept_chars = 0
    for segment in reversed(context.recent_segments):
        kept_segments.append(segment)
        kept_chars += len(segment)
        if kept_chars >= config.recent_context_chars:
            break
    kept_segments.reverse()

    archived_count = len(context.recent_segments) - len(kept_segments)
    archived_segments = context.recent_segments[:archived_count]
    archived_text = "\n\n".join(archived_segments).strip()
    if not archived_text:
        return

    context.compressed_memory = compress_text_memory(context.compressed_memory, archived_text, config.memory_budget_chars)
    context.recent_segments = kept_segments
    context.compression_events.append(
        CompressionEvent(
            utterance_index=utterance_index,
            archived_chars=len(archived_text),
            kept_recent_chars=sum(len(segment) for segment in kept_segments),
            compressed_memory_chars=len(context.compressed_memory),
        )
    )


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def write_outputs(session: SessionState, config: AppConfig) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with session.lock:
        transcript = session.transcript
        utterances = [record.to_dict() for record in session.utterances]
        compression_events = [event.to_dict() for event in session.context.compression_events]
        compressed_memory = session.context.compressed_memory

    config.output_path.write_text((transcript + "\n") if transcript else "", encoding="utf-8")
    metadata = {
        "started_at_utc": session.started_at.astimezone(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "faster-whisper",
        "model": config.model,
        "backend_device": config.backend_device,
        "compute_type": config.compute_type,
        "cpu_threads": config.cpu_threads,
        "beam_size": config.beam_size,
        "best_of": config.best_of,
        "patience": config.patience,
        "language": config.language,
        "device": config.device,
        "sample_rate": config.sample_rate,
        "block_ms": config.block_ms,
        "notes_supplied": bool(config.notes),
        "utterances": utterances,
        "compression_events": compression_events,
        "final_compressed_memory": compressed_memory,
        "output_file": str(config.output_path),
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def clean_whisper_artifacts(text: str) -> str:
    """Strip common Whisper hallucination artifacts from transcribed text."""
    text = ARTIFACT_PREFIX_RE.sub("", text)
    text = re.sub(r"^[^A-Za-z\"'(]+", "", text)
    return text.strip()


def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset and peak-normalize for consistent Whisper input."""
    audio = audio.astype(np.float32, copy=True)
    audio -= np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio *= 0.9 / peak
    return audio


def is_hallucination(text: str, recent_transcript: str) -> bool:
    """Detect repetition loops and exact duplicates that slip past VAD."""
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    lowered = stripped.lower().rstrip(".!?, ")
    words = lowered.split()
    if len(words) >= 6:
        for n in range(2, min(5, len(words) // 2 + 1)):
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if ngrams:
                most_common_count = Counter(ngrams).most_common(1)[0][1]
                if most_common_count >= 3 and most_common_count * n >= len(words) * 0.5:
                    return True
    recent_lower = recent_transcript.lower().strip()
    if len(lowered) > 10 and recent_lower.endswith(lowered):
        return True
    return False


def hotwords_from_notes(notes: str) -> str | None:
    if not notes.strip():
        return None
    cleaned = ", ".join(piece.strip() for piece in notes.replace("\n", ",").split(",") if piece.strip())
    return cleaned[:500] or None


def speak_warning(message: str) -> None:
    """Play a spoken warning through macOS text-to-speech (non-blocking)."""
    try:
        subprocess.Popen(
            ["say", "-v", "Samantha", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(f"TTS unavailable: {message}", file=sys.stderr)


def ensure_ollama(preferred_model: str = "auto") -> dict:
    """Check if Ollama is available, try to start it if not, and detect the best model.

    Returns a dict with:
        'status': 'ready' | 'started' | 'not_installed' | 'unavailable'
        'model':  name of the detected model (e.g. 'deepseek-r1:8b') or None
    """
    import shutil
    import subprocess
    import time as _time

    result = {"status": "unavailable", "model": None}

    def _ping() -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def _detect_model() -> str | None:
        """Query Ollama for available models and pick the best one."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if not models:
                    return None
                # If user specified a model and it's available, use it
                if preferred_model != "auto" and preferred_model in models:
                    return preferred_model
                # Otherwise pick the first available model
                # Prefer smaller models for speed: sort by parameter size if possible
                return models[0]
        except Exception:
            return None

    # 1. Already running?
    if _ping():
        result["status"] = "ready"
        result["model"] = _detect_model()
        return result

    # 2. Is ollama installed?
    if shutil.which("ollama") is None:
        result["status"] = "not_installed"
        return result

    # 3. Try to start it
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(10):
            _time.sleep(0.5)
            if _ping():
                print("Ollama auto-started successfully.", file=sys.stderr)
                result["status"] = "started"
                result["model"] = _detect_model()
                return result
    except Exception as exc:
        print(f"Failed to start Ollama: {exc}", file=sys.stderr)

    return result


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from deepseek-r1 chain-of-thought output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_ollama(prompt: str, model: str = "deepseek-r1:8b") -> str | None:
    """Query local Ollama instance. Returns the response text, or None on any failure."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
            raw = data.get("response", "").strip()
            cleaned = _strip_think_tags(raw)
            return cleaned or None
    except Exception as exc:
        print(f"Ollama unavailable ({exc.__class__.__name__}), falling back.", file=sys.stderr)
        return None


class TopicTracker:
    """Track topic drift with dynamic, multi-topic evolution.

    Instead of a single fixed topic, the tracker maintains a list of known topics
    discovered throughout the conversation. When sustained drift is detected, it
    re-queries Ollama to determine if the conversation moved to a new legitimate
    topic or is genuinely off-track.

    Topics are weighted by recency so old agenda items don't dominate the
    similarity comparison.
    """

    BOOTSTRAP_UTTERANCES = 8
    WARN_COOLDOWN_CHECKS = 6
    REFRESH_INTERVAL = 20  # re-summarize active topic every N utterances
    TOPIC_DECAY = 0.85     # weight multiplier for each older topic

    # Short filler phrases to skip during fallback topic extraction
    _FILLER_PATTERNS = re.compile(
        r"^(is it going|are we live|hello|hey|hi|yo|um+|uh+|okay|ok|alright|"
        r"can you hear|testing|check|one two|let'?s go|let'?s see|yeah|yep|nope|sure|"
        r"what'?s up|how are you|good morning|good afternoon)[.!?\s]*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        drift_threshold: float = 0.32,
        grace_count: int = 3,
        ollama_model: str = "llama3.2:1b",
        on_drift: Any = None,
        on_similarity: Any = None,
        on_topic_inferred: Any = None,
    ):
        self.drift_threshold = drift_threshold
        self.grace_count = grace_count
        self.ollama_model = ollama_model
        self.on_drift = on_drift
        self.on_similarity = on_similarity
        self.on_topic_inferred = on_topic_inferred
        self._model: Any = None

        # Multi-topic state: list of (topic_text, embedding, weight)
        self._known_topics: list[tuple[str, Any, float]] = []
        self._baseline_texts: list[str] = []
        self._bootstrapped = False
        self.topic: str = ""  # current active topic text

        self._drift_streak = 0
        self._checks_since_warn = 0
        self._utterance_count = 0
        self._last_refresh_count = 0
        self._reclassifying = False  # prevent concurrent reclassification
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_dir = Path(__file__).resolve().parent / "models" / "topic_encoder"
        self._model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(model_dir))
        print("Topic encoder ready.", file=sys.stderr)

    def _add_topic(self, topic_text: str) -> None:
        """Add a new topic to the known list, decaying all previous topic weights."""
        self._ensure_model()
        embedding = self._model.encode(topic_text, normalize_embeddings=True)
        # Decay existing topic weights
        self._known_topics = [
            (text, emb, weight * self.TOPIC_DECAY)
            for text, emb, weight in self._known_topics
        ]
        self._known_topics.append((topic_text, embedding, 1.0))
        self.topic = topic_text
        print(f"Topic added: {topic_text} (total: {len(self._known_topics)})", file=sys.stderr)

    def _best_similarity(self, embedding: Any) -> float:
        """Compute weighted best similarity against all known topics."""
        if not self._known_topics:
            return 1.0
        best = 0.0
        for _text, topic_emb, weight in self._known_topics:
            raw_sim = float(np.dot(topic_emb, embedding))
            weighted = raw_sim * weight
            best = max(best, weighted)
        return best

    def _extract_fallback_topic(self, combined: str) -> str:
        """Extract the most substantive sentence from the bootstrap text, skipping greetings/filler."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", combined) if s.strip()]
        # Filter out filler and very short sentences
        substantive = [
            s for s in sentences
            if len(s.split()) >= 5 and not self._FILLER_PATTERNS.match(s)
        ]
        if substantive:
            # Pick the longest sentence as the most informative
            return max(substantive, key=len)[:120]
        # If nothing passes, just take the longest raw sentence
        if sentences:
            return max(sentences, key=len)[:120]
        return combined[:120]

    def _bootstrap_topic(self) -> None:
        combined = " ".join(self._baseline_texts)
        if len(combined.split()) < 8:
            return

        prompt = (
            "Below is the beginning of a live conversation. "
            "Summarize the main topic in a short phrase (3-8 words). "
            "Only return the topic phrase, nothing else.\n\n"
            f"Conversation:\n{combined}\n\nTopic:"
        )
        llm_topic = ask_ollama(prompt, model=self.ollama_model)
        if llm_topic and 3 < len(llm_topic) < 100:
            topic = llm_topic.strip().rstrip(".")
            source = "ollama"
            print(f"Topic inferred (Ollama): {topic}", file=sys.stderr)
        else:
            raw_fallback = self._extract_fallback_topic(combined)
            # Try Ollama to condense the fallback sentence into a short phrase
            condense_prompt = (
                "Shorten this sentence into a brief topic title (3-8 words). "
                "Only return the short title, nothing else.\n\n"
                f"Sentence: {raw_fallback}\n\nShort title:"
            )
            condensed = ask_ollama(condense_prompt, model=self.ollama_model)
            if condensed and 3 < len(condensed) < 80:
                topic = condensed.strip().rstrip(".")
                source = "ollama"
                print(f"Topic condensed (Ollama): {topic}", file=sys.stderr)
            else:
                topic = raw_fallback
                source = "heuristic"
                print(f"Topic inferred (fallback): {topic}", file=sys.stderr)

        self._add_topic(topic)
        self._bootstrapped = True
        if self.on_topic_inferred:
            self.on_topic_inferred(topic, source)

    def _reclassify_drift(self, recent_window: str) -> None:
        """Ask Ollama whether the conversation shifted to a new topic or is off-track."""
        if self._reclassifying:
            return
        self._reclassifying = True
        try:
            known_list = ", ".join(f"'{t}'" for t, _, _ in self._known_topics)
            prompt = (
                f"The conversation has covered these topics so far: {known_list}.\n"
                f"The recent discussion is:\n\"{recent_window[:600]}\"\n\n"
                "Has the conversation moved to a new legitimate topic or agenda item, "
                "or is it genuinely off-track and unrelated?\n"
                "Reply with EXACTLY one of:\n"
                "NEW_TOPIC: <one sentence describing the new topic>\n"
                "OFF_TRACK\n"
            )
            response = ask_ollama(prompt, model=self.ollama_model)
            if response and response.strip().upper().startswith("NEW_TOPIC"):
                # Extract the new topic text
                new_topic = response.split(":", 1)[-1].strip().rstrip(".")
                if len(new_topic) > 5:
                    self._add_topic(new_topic)
                    with self._lock:
                        self._drift_streak = 0
                    if self.on_topic_inferred:
                        self.on_topic_inferred(new_topic, "ollama")
                    print(f"Topic evolved (Ollama): {new_topic}", file=sys.stderr)
                    return
            # If OFF_TRACK or Ollama failed, fire drift
            with self._lock:
                self._checks_since_warn = 0
            if self.on_drift:
                self.on_drift(0.0)
            speak_warning("Hey, it looks like the conversation has gone off topic.")
        except Exception as exc:
            print(f"Reclassification error: {exc}", file=sys.stderr)
        finally:
            self._reclassifying = False

    def _periodic_refresh(self, recent_window: str) -> None:
        """Re-summarize the active topic from recent conversation to keep baseline fresh."""
        prompt = (
            "Below is a segment of an ongoing conversation. "
            "What is the current main topic being discussed? "
            "Reply with one concise sentence only.\n\n"
            f"Conversation:\n{recent_window[:600]}\n\nCurrent topic:"
        )
        response = ask_ollama(prompt, model=self.ollama_model)
        if response and 5 < len(response) < 200:
            refreshed = response.strip().rstrip(".")
            # Check if it's actually different from the current topic
            self._ensure_model()
            new_emb = self._model.encode(refreshed, normalize_embeddings=True)
            best_sim = self._best_similarity(new_emb)
            if best_sim < 0.7:
                # Substantially different — it's a new topic
                self._add_topic(refreshed)
                if self.on_topic_inferred:
                    self.on_topic_inferred(refreshed, "ollama")
                print(f"Topic refreshed (new): {refreshed}", file=sys.stderr)
            else:
                # Just update the latest topic embedding to stay current
                if self._known_topics:
                    self._known_topics[-1] = (refreshed, new_emb, self._known_topics[-1][2])
                    self.topic = refreshed
                print(f"Topic refreshed (updated): {refreshed}", file=sys.stderr)

    def check(self, utterance_text: str, full_transcript: str) -> float:
        """Call after each utterance. Returns similarity (0-1) or 1.0 if still bootstrapping."""
        if not utterance_text.strip():
            return 1.0

        self._utterance_count += 1

        if not self._bootstrapped:
            self._baseline_texts.append(utterance_text)
            if len(self._baseline_texts) >= self.BOOTSTRAP_UTTERANCES:
                self._bootstrap_topic()
            return 1.0

        self._ensure_model()
        window = full_transcript[-800:]
        filler_re = re.compile(
            r"\b(uh+|um+|like|you know|i mean|okay|so|right|yeah|hmm+|huh|oh|ah)\b",
            re.IGNORECASE,
        )
        cleaned = filler_re.sub("", window)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned.split()) < 4:
            return 1.0

        embedding = self._model.encode(cleaned, normalize_embeddings=True)
        similarity = self._best_similarity(embedding)

        if self.on_similarity:
            self.on_similarity(similarity)

        # Periodic refresh
        if (self._utterance_count - self._last_refresh_count) >= self.REFRESH_INTERVAL:
            self._last_refresh_count = self._utterance_count
            threading.Thread(target=self._periodic_refresh, args=(cleaned,), daemon=True).start()

        with self._lock:
            self._checks_since_warn += 1
            if similarity < self.drift_threshold:
                self._drift_streak += 1
                if (
                    self._drift_streak >= self.grace_count
                    and self._checks_since_warn >= self.WARN_COOLDOWN_CHECKS
                ):
                    # Instead of immediately classifying drift, ask Ollama to reclassify
                    threading.Thread(
                        target=self._reclassify_drift, args=(cleaned,), daemon=True
                    ).start()
            else:
                self._drift_streak = max(0, self._drift_streak - 1)

        return similarity


class TranscriberEngine:
    """Reusable engine wrapping model, audio capture, segmentation, and transcription.

    Attributes:
        config: Application configuration.
        session: Mutable session state (transcript, utterances, context).
        on_utterance: Optional callback invoked with each new UtteranceRecord.
        on_compression: Optional callback invoked when context compression fires.
        on_status: Optional callback invoked on status changes ("listening", "transcribing", "stopped").
    """

    def __init__(
        self,
        config: AppConfig,
        on_utterance: Any = None,
        on_compression: Any = None,
        on_status: Any = None,
        on_drift: Any = None,
        on_similarity: Any = None,
    ):
        self.config = config
        self.on_utterance = on_utterance
        self.on_compression = on_compression
        self.on_status = on_status
        self.on_drift = on_drift
        self.on_similarity = on_similarity

        self.model: WhisperModel | None = None
        self.speaker_tracker: SpeakerTracker | None = None
        self.topic_tracker: TopicTracker | None = None
        self.session = SessionState(started_at=datetime.now().astimezone())
        self._utterance_queue: queue.Queue[Utterance | None] = queue.Queue()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._segmenter = AudioSegmenter(config)
        self._worker: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        self._running = False
        self._audio_thread: threading.Thread | None = None

    def load_model(self) -> None:
        self.model = load_local_model(self.config)

    def enable_topic_tracking(self, on_topic_inferred: Any = None) -> None:
        """Enable auto-inferred topic tracking."""
        try:
            self.topic_tracker = TopicTracker(
                ollama_model=self.config.ollama_model,
                on_drift=self.on_drift,
                on_similarity=self.on_similarity,
                on_topic_inferred=on_topic_inferred,
            )
            self.topic_tracker._ensure_model()
            print("Topic tracking enabled (will infer from conversation).", file=sys.stderr)
        except ImportError:
            print(
                "Warning: sentence-transformers not installed, topic tracking disabled.\n"
                "  Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
            self.topic_tracker = None

    def _init_speaker_tracker(self) -> None:
        if not self.config.diarize:
            return
        try:
            profiles_path = self.config.speakers_file or (
                Path(__file__).resolve().parent / "speakers.json"
            )
            self.speaker_tracker = SpeakerTracker(
                threshold=self.config.speaker_threshold, profiles_path=profiles_path,
            )
            self.speaker_tracker._ensure_model()
        except ImportError:
            print(
                "Warning: speechbrain not installed, diarization disabled.\n"
                "  Install with: pip install torch torchaudio speechbrain",
                file=sys.stderr,
            )
            self.speaker_tracker = None

    def _transcription_worker_loop(self) -> None:
        while True:
            utterance = self._utterance_queue.get()
            try:
                if utterance is None:
                    return
                if self.on_status:
                    self.on_status("transcribing")
                duration = utterance.end_seconds - utterance.start_seconds
                print(f"Transcribing utterance {utterance.index} ({duration:.1f}s)...", file=sys.stderr)
                self._transcribe_utterance(utterance)
                if self.on_status:
                    self.on_status("listening")
            except Exception as exc:
                print(f"Transcription error on utterance {getattr(utterance, 'index', '?')}: {exc}", file=sys.stderr)
            finally:
                self._utterance_queue.task_done()

    def _transcribe_utterance(self, utterance: Utterance) -> None:
        assert self.model is not None
        with self.session.lock:
            prompt = build_prompt(self.session.context, self.config)
            transcript_tail = self.session.transcript[-8_000:]

        segments, _info = self.model.transcribe(
            preprocess_audio(utterance.audio),
            language=self.config.language,
            task="transcribe",
            log_progress=self.config.verbose_model,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            patience=self.config.patience,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.45,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            hotwords=hotwords_from_notes(self.config.notes),
            without_timestamps=True,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.35,
                min_silence_duration_ms=250,
                min_speech_duration_ms=100,
                speech_pad_ms=200,
            ),
            repetition_penalty=1.2,
            no_repeat_ngram_size=0,
        )

        raw_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        text = clean_whisper_artifacts(trim_overlapping_prefix(transcript_tail, raw_text))
        if not text:
            return

        if self.config.hallucination_filter and is_hallucination(text, transcript_tail):
            print(f"[{format_seconds(utterance.start_seconds)}] (filtered: {text[:80]})", file=sys.stderr)
            return

        speaker = ""
        if self.speaker_tracker is not None:
            speaker = self.speaker_tracker.identify(utterance.audio, self.config.sample_rate)
            self.speaker_tracker.save_profiles()

        compression_fired = False
        with self.session.lock:
            prev_compression_count = len(self.session.context.compression_events)
            speaker_changed = speaker and speaker != self.session.last_speaker
            if speaker_changed:
                self.session.transcript = self.session.transcript.rstrip() + f"\n\n[{speaker}]\n" + text
                self.session.last_speaker = speaker
            else:
                self.session.transcript = merge_transcript(self.session.transcript, text)
            self.session.context.recent_segments.append(text)
            maybe_roll_context(self.session.context, self.config, utterance.index)
            record = UtteranceRecord(
                index=utterance.index,
                start_seconds=utterance.start_seconds,
                end_seconds=utterance.end_seconds,
                text=text,
                raw_text=raw_text,
                forced_split=utterance.forced_split,
                speaker=speaker,
            )
            self.session.utterances.append(record)
            compression_fired = len(self.session.context.compression_events) > prev_compression_count

        write_outputs(self.session, self.config)
        tag = f" [{speaker}]" if speaker else ""
        print(f"[{format_seconds(utterance.start_seconds)}]{tag} {text}")

        if self.on_utterance:
            self.on_utterance(record)
        if compression_fired and self.on_compression:
            with self.session.lock:
                event = self.session.context.compression_events[-1]
            self.on_compression(event)

        if self.topic_tracker is not None:
            with self.session.lock:
                full = self.session.transcript
            self.topic_tracker.check(text, full)

    def _audio_callback(self, indata: np.ndarray, frames: int, callback_time: Any, status: sd.CallbackFlags) -> None:
        del frames, callback_time
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        try:
            self._audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            print("Audio queue overflow: transcription is falling behind.", file=sys.stderr)

    def _audio_processing_loop(self) -> None:
        try:
            while self._running:
                try:
                    block = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    for utterance in self._segmenter.process_block(block):
                        self._utterance_queue.put(utterance)
                finally:
                    self._audio_queue.task_done()
        except Exception as exc:
            print(f"Audio processing error: {exc}", file=sys.stderr)

    def start(self) -> None:
        """Start live transcription with local microphone (non-blocking)."""
        if self._running:
            return
        if self.model is None:
            self.load_model()
        self._init_speaker_tracker()
        self.session = SessionState(started_at=datetime.now().astimezone())
        self._segmenter = AudioSegmenter(self.config)
        self._running = True
        self._headless = False

        self._worker = threading.Thread(target=self._transcription_worker_loop, daemon=True)
        self._worker.start()

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.config.sample_rate * self.config.block_ms / 1000),
            device=self.config.device,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self._audio_thread.start()

        write_outputs(self.session, self.config)
        diarize_status = f" | Diarization: {'on' if self.speaker_tracker else 'off'}"
        print(f"Listening.{diarize_status}", file=sys.stderr)

        if self.on_status:
            self.on_status("listening")

    def stop(self) -> None:
        """Stop live transcription, flush remaining audio."""
        if not self._running:
            return
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

            if self._audio_thread is not None:
                self._audio_thread.join(timeout=3)
                self._audio_thread = None

        flushed = self._segmenter.flush()
        if flushed is not None:
            self._utterance_queue.put(flushed)
        self._utterance_queue.put(None)
        self._utterance_queue.join()

        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

        write_outputs(self.session, self.config)
        print(f"Final transcript saved to {self.config.output_path}", file=sys.stderr)

        if self.on_status:
            self.on_status("stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_transcript(self) -> str:
        with self.session.lock:
            return self.session.transcript

    def get_utterances(self) -> list[UtteranceRecord]:
        with self.session.lock:
            return list(self.session.utterances)

    def get_compressed_memory(self) -> str:
        with self.session.lock:
            return self.session.context.compressed_memory


def run_live_transcriber(config: AppConfig) -> None:
    engine = TranscriberEngine(config)
    engine.load_model()
    if config.warmup_only:
        return
    engine.start()
    try:
        while engine.is_running:
            import time
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        engine.stop()


def main() -> int:
    config = parse_args()
    run_live_transcriber(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
