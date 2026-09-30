# qwen-md-tts

Dieses kleine Python-Werkzeug wandelt eine UTF-8-kodierte Markdown-Datei mit
`qwen-audio-3.1-tts-flash` in eine zusammenhängende WAV-Datei um. Es ist auf
deutsche Sachtexte, Dokumentationen und Hörbuch-Manuskripte ausgerichtet.

## Funktionen

- entfernt Markdown-Syntax, Links, Metadaten und standardmäßig Codeblöcke
- zerlegt lange Dokumente an Absatz- und Satzgrenzen
- verwendet `language_hints=["de"]` und eine frei formulierbare Stilanweisung
- unterstützt ein lokales Aussprachewörterbuch
- lädt jede erzeugte Audiodatei sofort herunter
- setzt einen abgebrochenen Lauf anhand vorhandener Segmente fort
- fügt WAV-Segmente mit kurzen Pausen zusammen
- normalisiert die Lautheit mit ffmpeg, sofern verfügbar
- bietet einen kostenfreien `--dry-run` zur Kontrolle des gesprochenen Textes

## Voraussetzungen

- Python 3.11 oder neuer
- optional ffmpeg für Lautheitsnormalisierung
- ein API-Key und eine Workspace-ID aus Alibaba Cloud Model Studio, Region
  China/Beijing

Das Programm liest ausschließlich diese Umgebungsvariablen:

```text
DASHSCOPE_API_KEY
SFM_WORKSPACE_ID
```

Die Werte werden nicht in Dateien oder Protokolle geschrieben. Wenn die
Variablen unter Windows dauerhaft gesetzt wurden, nachdem Codex oder das
Terminal bereits gestartet war, muss die jeweilige Anwendung neu gestartet
werden, damit der neue Prozess sie übernimmt.

## Erster Test ohne API-Aufruf

```powershell
python .\qwen_md_tts.py .\mein-text.md --dry-run
```

Dabei entstehen `mein-text.prepared.txt` und `mein-text.chunks.json`. Vor dem
ersten kostenpflichtigen Lauf sollte der vorbereitete Text kontrolliert werden.

## Audio erzeugen

```powershell
python .\qwen_md_tts.py .\mein-text.md -o .\mein-text.wav
```

Andere offiziell deutschfähige Systemstimmen ausprobieren:

```powershell
python .\qwen_md_tts.py .\mein-text.md --voice xunanchuan_v3.1
python .\qwen_md_tts.py .\mein-text.md --voice longanhuan_v3.1
python .\qwen_md_tts.py .\mein-text.md --voice longanfengyue_v3.1
```

## Aussprachewörterbuch

Eine JSON-Datei ordnet die Schreibweise der gewünschten Sprechweise zu:

```json
{
  "API": "A P I",
  "Qwen": "Tschwen"
}
```

Aufruf:

```powershell
python .\qwen_md_tts.py .\mein-text.md --dictionary .\pronunciations.json
```

Die längsten Begriffe werden zuerst ersetzt, damit sich etwa `SQL` und
`PostgreSQL` nicht gegenseitig stören.

## Wichtige Optionen

```text
--voice VOICE            Stimmen-ID
--instruction TEXT       gewünschter Sprechstil
--rate 0.5..2.0          Sprechtempo
--pitch 0.5..2.0         Tonhöhe
--chunk-chars 1800       lokale Segmentgröße
--pause-ms 350           Pause zwischen Segmenten
--read-code              Codeblöcke mit vorlesen
--no-normalize           ffmpeg-Normalisierung deaktivieren
--dry-run                keine API-Anfrage senden
```

Zwischendateien und der Fortsetzungsstatus liegen unter `.qwen-md-tts/` neben
der Ausgabedatei. Verändert sich der Quelltext oder eine relevante Einstellung,
wird automatisch ein neuer Arbeitsordner verwendet.

## Tests

```powershell
python -m unittest discover -s tests -v
```

