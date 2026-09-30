"""Turn a Markdown document into a continuous WAV narration with Qwen Audio.

The program intentionally uses only Python's standard library. Credentials are
read at runtime from DASHSCOPE_API_KEY and SFM_WORKSPACE_ID and are never stored
in manifests or output files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


VERSION = "0.1.0"
MODEL = "qwen-audio-3.1-tts-flash"
DEFAULT_VOICE = "longanlingxin_v3.1"
DEFAULT_INSTRUCTION = (
    "Sprich in natürlichem Hochdeutsch, ruhig, klar und ausdrucksstark wie eine "
    "professionelle Hörbuchsprecherin. Achte besonders auf Eigennamen, "
    "Abkürzungen und sinnvolle Satzpausen."
)
API_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


class QwenTtsError(RuntimeError):
    """A user-facing synthesis error."""


@dataclass(frozen=True)
class SynthesisConfig:
    model: str = MODEL
    voice: str = DEFAULT_VOICE
    language: str = "de"
    instruction: str = DEFAULT_INSTRUCTION
    sample_rate: int = 48000
    volume: int = 50
    rate: float = 1.0
    pitch: float = 1.0
    seed: int = 42
    pause_ms: int = 350
    chunk_chars: int = 1800
    normalize: bool = True


def _ensure_sentence_end(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?…:;":
        return text + "."
    return text


def markdown_to_speech(markdown: str, *, read_code: bool = False) -> str:
    """Convert common Markdown constructs into narration-friendly plain text."""
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    def fenced(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if not read_code or not body:
            return "\n\n"
        flattened = re.sub(r"\s+", " ", body)
        return f"\n\nCodebeispiel. {flattened}\n\n"

    text = re.sub(r"```[^\n]*\n(.*?)```", fenced, text, flags=re.DOTALL)
    text = re.sub(r"~~~[^\n]*\n(.*?)~~~", fenced, text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", lambda m: m.group(1), text)
    text = re.sub(r"<https?://[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*\[\^[^]]+\]:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\^[^]]+\]", "", text)

    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            output.append("")
            continue
        if re.fullmatch(r"(?:[-*_]\s*){3,}", line):
            output.append("")
            continue
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", line):
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if heading:
            output.extend(["", _ensure_sentence_end(heading.group(1)), ""])
            continue
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"^[-+*]\s+(?:\[[ xX]\]\s+)?", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            line = ". ".join(cells)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"(?<!\w)[*_~]{1,3}|[*_~]{1,3}(?!\w)", "", line)
        output.append(line)

    cleaned = html.unescape("\n".join(output))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in cleaned.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p).strip()


def load_pronunciations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenTtsError(f"Aussprachewörterbuch kann nicht gelesen werden: {path}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise QwenTtsError("Das Aussprachewörterbuch muss ein JSON-Objekt aus Textpaaren sein.")
    return {key: value for key, value in data.items() if key and value}


def apply_pronunciations(text: str, pronunciations: dict[str, str]) -> str:
    for written in sorted(pronunciations, key=len, reverse=True):
        spoken = pronunciations[written]
        if written[0].isalnum() and written[-1].isalnum():
            pattern = rf"(?<!\w){re.escape(written)}(?!\w)"
        else:
            pattern = re.escape(written)
        text = re.sub(pattern, lambda _match, replacement=spoken: replacement, text)
    return text


_ABBREVIATIONS = (
    "z. B.", "d. h.", "u. a.", "u. U.", "bzw.", "ca.", "Dr.", "Prof.",
    "Nr.", "Abb.", "usw.", "etc.", "vgl.", "ggf.", "inkl.", "exkl.",
)


def _sentences(paragraph: str) -> list[str]:
    protected = paragraph
    marker = "\u2024"
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", marker))
    pieces = re.split(r"(?<=[.!?…])\s+(?=[\"„“'»«(]*[A-ZÄÖÜ0-9])", protected)
    return [piece.replace(marker, ".").strip() for piece in pieces if piece.strip()]


def _hard_wrap(text: str, limit: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > limit:
            parts.append(" ".join(current))
            current, length = [], 0
        if len(word) > limit:
            if current:
                parts.append(" ".join(current))
                current, length = [], 0
            parts.extend(word[index:index + limit] for index in range(0, len(word), limit))
            continue
        current.append(word)
        length += extra
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(text: str, limit: int = 1800) -> list[str]:
    if limit < 200:
        raise QwenTtsError("--chunk-chars muss mindestens 200 sein.")
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append(paragraph)
            continue
        for sentence in _sentences(paragraph):
            units.extend([sentence] if len(sentence) <= limit else _hard_wrap(sentence, limit))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if current and len(candidate) > limit:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _read_json_response(response: Any) -> dict[str, Any]:
    raw = response.read().decode("utf-8", errors="replace")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QwenTtsError("Model Studio hat keine gültige JSON-Antwort geliefert.") from exc
    if not isinstance(result, dict):
        raise QwenTtsError("Model Studio hat ein unerwartetes Antwortformat geliefert.")
    return result


class QwenAudioClient:
    def __init__(self, api_key: str, workspace_id: str, config: SynthesisConfig) -> None:
        if not api_key:
            raise QwenTtsError("DASHSCOPE_API_KEY ist in diesem Prozess nicht gesetzt.")
        if not workspace_id:
            raise QwenTtsError("SFM_WORKSPACE_ID ist in diesem Prozess nicht gesetzt.")
        if not WORKSPACE_PATTERN.fullmatch(workspace_id):
            raise QwenTtsError("SFM_WORKSPACE_ID enthält unerwartete Zeichen.")
        self._api_key = api_key
        self._endpoint = (
            f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com{API_PATH}"
        )
        self.config = config

    def synthesize(self, text: str, destination: Path) -> dict[str, Any]:
        input_data: dict[str, Any] = {
            "text": text,
            "voice": self.config.voice,
            "format": "wav",
            "sample_rate": self.config.sample_rate,
            "volume": self.config.volume,
            "rate": self.config.rate,
            "pitch": self.config.pitch,
            "seed": self.config.seed,
            "language_hints": [self.config.language],
        }
        if self.config.instruction:
            input_data["instruction"] = self.config.instruction
        payload = json.dumps(
            {"model": self.config.model, "input": input_data}, ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        result: dict[str, Any] | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = _read_json_response(response)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1200]
                if exc.code == 429 or 500 <= exc.code < 600:
                    if attempt < 4:
                        time.sleep(2 ** (attempt + 1))
                        continue
                try:
                    detail = json.loads(body)
                    message = detail.get("message") or detail.get("code") or body
                except json.JSONDecodeError:
                    message = body or exc.reason
                raise QwenTtsError(f"Model Studio HTTP {exc.code}: {message}") from exc
            except urllib.error.URLError as exc:
                if attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise QwenTtsError(f"Model Studio ist nicht erreichbar: {exc.reason}") from exc

        if result is None:
            raise QwenTtsError("Model Studio antwortet auch nach mehreren Versuchen nicht.")
        output = result.get("output") or {}
        audio = output.get("audio") or {}
        url = audio.get("url")
        if not isinstance(url, str) or not url:
            raise QwenTtsError(f"Die API-Antwort enthält keine Audio-URL: {result}")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise QwenTtsError("Die API-Antwort enthält eine ungültige Audio-URL.")

        temp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                temp_path.write_bytes(response.read())
            validate_wav(temp_path)
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
        return {
            "request_id": result.get("request_id"),
            "usage": result.get("usage"),
            "audio_id": audio.get("id"),
        }


def validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() <= 0:
                raise QwenTtsError(f"Leere WAV-Datei: {path}")
    except (wave.Error, EOFError, OSError) as exc:
        raise QwenTtsError(f"Ungültige WAV-Datei: {path}") from exc


def concatenate_wavs(paths: Iterable[Path], destination: Path, pause_ms: int) -> None:
    paths = list(paths)
    if not paths:
        raise QwenTtsError("Keine Audiosegmente zum Zusammenfügen vorhanden.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    params: tuple[int, int, int, str] | None = None
    with wave.open(str(destination), "wb") as output:
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as source:
                current = (
                    source.getnchannels(), source.getsampwidth(), source.getframerate(),
                    source.getcomptype(),
                )
                if params is None:
                    params = current
                    output.setnchannels(current[0])
                    output.setsampwidth(current[1])
                    output.setframerate(current[2])
                    output.setcomptype(current[3], source.getcompname())
                elif current != params:
                    raise QwenTtsError(f"Audioformat von {path.name} passt nicht zu den übrigen Segmenten.")
                output.writeframes(source.readframes(source.getnframes()))
                if index < len(paths) - 1 and pause_ms > 0:
                    assert params is not None
                    silence_frames = round(params[2] * pause_ms / 1000)
                    output.writeframes(b"\x00" * silence_frames * params[0] * params[1])


def normalize_wav(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5", "-c:a", "pcm_s16le", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise QwenTtsError(f"Lautheitsnormalisierung fehlgeschlagen: {detail}")
    return True


def _atomic_json(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _job_id(source: bytes, config: SynthesisConfig, pronunciations: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(source)
    digest.update(json.dumps(asdict(config), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(pronunciations, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(VERSION.encode("ascii"))
    return digest.hexdigest()[:16]


def build_audio(
    source_path: Path,
    output_path: Path,
    config: SynthesisConfig,
    pronunciations_path: Path | None,
    *,
    read_code: bool,
    dry_run: bool,
) -> Path:
    try:
        source_bytes = source_path.read_bytes()
        markdown = source_bytes.decode("utf-8-sig")
    except OSError as exc:
        raise QwenTtsError(f"Eingabedatei kann nicht gelesen werden: {source_path}") from exc
    except UnicodeDecodeError as exc:
        raise QwenTtsError("Die Markdown-Datei muss UTF-8-kodiert sein.") from exc

    pronunciations = load_pronunciations(pronunciations_path)
    prepared = apply_pronunciations(markdown_to_speech(markdown, read_code=read_code), pronunciations)
    if not prepared:
        raise QwenTtsError("Nach der Markdown-Bereinigung ist kein sprechbarer Text übrig.")
    chunks = chunk_text(prepared, config.chunk_chars)
    job_id = _job_id(source_bytes, config, pronunciations)

    prepared_path = output_path.with_suffix(".prepared.txt")
    chunks_path = output_path.with_suffix(".chunks.json")
    prepared_path.write_text(prepared + "\n", encoding="utf-8")
    _atomic_json(chunks_path, {"job_id": job_id, "chunks": chunks})
    print(f"Vorbereitet: {len(prepared):,} Zeichen in {len(chunks)} Segment(en).")
    print(f"Kontrolltext: {prepared_path}")
    if dry_run:
        print("Testmodus: keine API-Anfrage gesendet.")
        return prepared_path

    client = QwenAudioClient(
        os.environ.get("DASHSCOPE_API_KEY", "").strip(),
        os.environ.get("SFM_WORKSPACE_ID", "").strip(),
        config,
    )
    work_dir = output_path.parent / ".qwen-md-tts" / f"{source_path.stem}-{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "version": VERSION,
        "job_id": job_id,
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "config": asdict(config),
        "segments": [],
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("job_id") == job_id:
                manifest = existing
        except (OSError, json.JSONDecodeError):
            pass

    existing_segments = {
        item.get("index"): item
        for item in manifest.get("segments", [])
        if isinstance(item, dict)
    }
    segment_paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        segment_path = work_dir / f"{index:04d}.wav"
        segment_paths.append(segment_path)
        if segment_path.exists():
            try:
                validate_wav(segment_path)
                print(f"[{index}/{len(chunks)}] Bereits vorhanden, übersprungen.")
                continue
            except QwenTtsError:
                segment_path.unlink(missing_ok=True)
        print(f"[{index}/{len(chunks)}] Erzeuge {len(chunk)} Zeichen …")
        metadata = client.synthesize(chunk, segment_path)
        existing_segments[index] = {
            "index": index,
            "characters": len(chunk),
            "file": segment_path.name,
            **metadata,
        }
        manifest["segments"] = [existing_segments[key] for key in sorted(existing_segments)]
        _atomic_json(manifest_path, manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen-md-tts-") as temp_dir:
        joined = Path(temp_dir) / "joined.wav"
        concatenate_wavs(segment_paths, joined, config.pause_ms)
        if config.normalize:
            normalized = Path(temp_dir) / "normalized.wav"
            if normalize_wav(joined, normalized):
                shutil.copyfile(normalized, output_path)
            else:
                print("Hinweis: ffmpeg fehlt; WAV wird ohne Lautheitsnormalisierung gespeichert.")
                shutil.copyfile(joined, output_path)
        else:
            shutil.copyfile(joined, output_path)
    validate_wav(output_path)
    print(f"Fertig: {output_path}")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Markdown mit qwen-audio-3.1-tts-flash in eine WAV-Datei umwandeln."
    )
    parser.add_argument("input", type=Path, help="UTF-8-kodierte Markdown-Datei")
    parser.add_argument("-o", "--output", type=Path, help="Ziel-WAV (Standard: neben der Eingabe)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Qwen-Audio-3.1-Stimmen-ID")
    parser.add_argument("--language", default="de", help="Sprachhinweis, für Deutsch: de")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Sprechstil als Freitext")
    parser.add_argument("--rate", type=float, default=1.0, help="Sprechtempo 0.5–2.0")
    parser.add_argument("--pitch", type=float, default=1.0, help="Tonhöhe 0.5–2.0")
    parser.add_argument("--volume", type=int, default=50, help="Lautstärke 0–100")
    parser.add_argument("--sample-rate", type=int, default=48000, choices=[8000, 12000, 16000, 22050, 24000, 44100, 48000])
    parser.add_argument("--seed", type=int, default=42, help="Reproduzierbarer Seed 0–65535")
    parser.add_argument("--chunk-chars", type=int, default=1800, help="Maximale Zeichen pro API-Aufruf")
    parser.add_argument("--pause-ms", type=int, default=350, help="Pause zwischen Segmenten")
    parser.add_argument("--dictionary", type=Path, help="JSON-Aussprachewörterbuch")
    parser.add_argument("--read-code", action="store_true", help="Codeblöcke ebenfalls vorlesen")
    parser.add_argument("--dry-run", action="store_true", help="Nur Text vorbereiten, keine API-Kosten")
    parser.add_argument("--no-normalize", action="store_true", help="Lautheitsnormalisierung überspringen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".wav")
    if output.suffix.lower() != ".wav":
        print("Fehler: Die Ausgabe muss derzeit eine .wav-Datei sein.", file=sys.stderr)
        return 2
    if not 0.5 <= args.rate <= 2.0 or not 0.5 <= args.pitch <= 2.0:
        print("Fehler: --rate und --pitch müssen zwischen 0.5 und 2.0 liegen.", file=sys.stderr)
        return 2
    if not 0 <= args.volume <= 100 or not 0 <= args.seed <= 65535:
        print("Fehler: --volume oder --seed liegt außerhalb des gültigen Bereichs.", file=sys.stderr)
        return 2
    if args.pause_ms < 0:
        print("Fehler: --pause-ms darf nicht negativ sein.", file=sys.stderr)
        return 2
    config = SynthesisConfig(
        voice=args.voice,
        language=args.language,
        instruction=args.instruction,
        sample_rate=args.sample_rate,
        volume=args.volume,
        rate=args.rate,
        pitch=args.pitch,
        seed=args.seed,
        pause_ms=args.pause_ms,
        chunk_chars=args.chunk_chars,
        normalize=not args.no_normalize,
    )
    try:
        build_audio(
            args.input,
            output,
            config,
            args.dictionary,
            read_code=args.read_code,
            dry_run=args.dry_run,
        )
    except QwenTtsError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

