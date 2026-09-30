import tempfile
import unittest
import wave
from pathlib import Path

from qwen_md_tts import (
    apply_pronunciations,
    chunk_text,
    concatenate_wavs,
    markdown_to_speech,
)


class MarkdownTests(unittest.TestCase):
    def test_markdown_is_made_speakable(self) -> None:
        source = """---
title: Intern
---
# Einführung

Das ist **wichtig** und [die Dokumentation](https://example.com/docs).

- Erster Punkt
- Zweiter Punkt

```python
print("nicht lesen")
```
"""
        result = markdown_to_speech(source)
        self.assertIn("Einführung.", result)
        self.assertIn("Das ist wichtig und die Dokumentation.", result)
        self.assertIn("Erster Punkt", result)
        self.assertNotIn("https://", result)
        self.assertNotIn("print", result)
        self.assertNotIn("**", result)

    def test_pronunciation_dictionary_prefers_long_terms(self) -> None:
        result = apply_pronunciations(
            "PostgreSQL und SQL nutzen eine API.",
            {"SQL": "S Q L", "PostgreSQL": "Postgres Q L", "API": "A P I"},
        )
        self.assertEqual(result, "Postgres Q L und S Q L nutzen eine A P I.")

    def test_chunks_stay_within_limit(self) -> None:
        text = "\n\n".join(
            [
                "Dies ist der erste längere Absatz. Er enthält mehrere vollständige Sätze und soll an einer natürlichen Satzgrenze getrennt werden.",
                "Dies ist der zweite längere Absatz. Auch dieser enthält genügend Wörter, damit die Segmentierung realistisch geprüft werden kann.",
                "Ein abschließender Absatz beendet den kleinen Test und stellt sicher, dass kein erzeugtes Segment die vorgegebene Länge überschreitet.",
            ]
        )
        chunks = chunk_text(text, 200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))


class WaveTests(unittest.TestCase):
    @staticmethod
    def _write_silence(path: Path, frames: int) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\x00\x00" * frames)

    def test_wav_concatenation_adds_pause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, target = root / "a.wav", root / "b.wav", root / "out.wav"
            self._write_silence(first, 800)
            self._write_silence(second, 800)
            concatenate_wavs([first, second], target, pause_ms=100)
            with wave.open(str(target), "rb") as result:
                self.assertEqual(result.getnframes(), 2400)


if __name__ == "__main__":
    unittest.main()
