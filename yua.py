import re
import os
import sys
import json
import uuid
import datetime
from pathlib import Path
from ollama import chat, web_fetch, web_search, list as ollama_list

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox, QSlider,
    QScrollArea, QFrame, QSizePolicy, QFileDialog, QComboBox, QListWidget, QListWidgetItem, QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import (
    QTextCursor, QTextBlockFormat, QTextCharFormat,
    QColor, QFont, QKeyEvent, QGuiApplication,
    QPixmap, QPainter, QBrush, QIcon,
)

# ── voice imports (lazy — only loaded when used) ──────────────────────────────
def _load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")

def _find_model_file(name: str) -> str:
    """Search for model files in common locations."""
    candidates = [
        Path(__file__).parent / name,              # same dir as script
        Path.home() / ".yua" / "models" / name,   # installed location
        Path.home() / ".yua" / name,               # legacy location
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"{name} not found. Run install.sh or download manually to ~/.yua/models/"
    )

def _load_kokoro():
    from kokoro_onnx import Kokoro
    return Kokoro(
        _find_model_file("kokoro-v1.0.onnx"),
        _find_model_file("voices-v1.0.bin")
    )

# ── persistent storage ───────────────────────────────────────────────────────
DATA_DIR  = Path.home() / ".yua"
CONV_DIR  = DATA_DIR / "conversations"
MEM_FILE  = DATA_DIR / "memory.json"
DATA_DIR.mkdir(exist_ok=True)
CONV_DIR.mkdir(exist_ok=True)

def _load_memory() -> str:
    """Return memory string to inject into system prompt."""
    try:
        data = json.loads(MEM_FILE.read_text())
        facts = data.get("facts", [])
        if not facts:
            return ""
        return "\n\n[Memory from previous conversations:]\n" + "\n".join(f"- {f}" for f in facts)
    except Exception:
        return ""

def _save_memory(facts: list[str]):
    MEM_FILE.write_text(json.dumps({"facts": facts}, ensure_ascii=False, indent=2))

def _list_conversations() -> list[dict]:
    """Return list of saved conversations sorted newest first."""
    convs = []
    for f in CONV_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            convs.append({"id": f.stem, "title": data.get("title", "Untitled"),
                          "ts": data.get("ts", ""), "file": f})
        except Exception:
            pass
    return sorted(convs, key=lambda x: x["ts"], reverse=True)

def _serialise_messages(messages: list) -> list:
    """Convert Ollama Message objects to plain dicts for JSON serialisation."""
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        else:
            # Ollama Message object — extract fields manually
            d = {"role": getattr(m, "role", ""), "content": getattr(m, "content", "") or ""}
            if hasattr(m, "tool_calls") and m.tool_calls:
                d["tool_calls"] = [
                    {"function": {"name": tc.function.name,
                                  "arguments": dict(tc.function.arguments)}}
                    for tc in m.tool_calls
                ]
            result.append(d)
    return result

def _save_conversation(conv_id: str, messages: list, title: str = ""):
    path = CONV_DIR / f"{conv_id}.json"
    path.write_text(json.dumps({
        "id": conv_id, "title": title,
        "ts": datetime.datetime.now().isoformat(),
        "messages": _serialise_messages(messages),
    }, ensure_ascii=False, indent=2))

def _load_conversation(conv_id: str) -> list:
    path = CONV_DIR / f"{conv_id}.json"
    data = json.loads(path.read_text())
    return data.get("messages", [])

def _make_app_icon() -> "QIcon":
    """Draw the 結 badge as a QIcon for the window/taskbar."""
    size = 64
    px = QPixmap(size, size)
    px.fill(QColor("transparent"))
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # rounded square background
    p.setBrush(QBrush(QColor("#cc785c")))
    p.setPen(QColor("transparent"))
    p.drawRoundedRect(0, 0, size, size, 12, 12)
    # draw 結 character
    p.setPen(QColor("#1a1a1a"))
    font = QFont("Noto Sans CJK JP", 32, QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, "結")
    p.end()
    return QIcon(px)

C = {
    "bg":       "#1a1a1a",
    "side":     "#212121",
    "input":    "#2a2a2a",
    "code_bg":  "#0d0d0f",
    "hover":    "#2f2f2f",
    "border":   "#333333",
    "accent":   "#cc785c",
    "accent_d": "#a85f47",
    "text":     "#ececec",
    "sec":      "#8e8ea0",
    "dim":      "#4a4a5a",
    "green":    "#4ade80",
    "yellow":   "#fbbf24",
    "red":      "#f87171",
    "pill_bg":  "#2a2a35",
    "syn_kw":   "#c792ea",
    "syn_str":  "#c3e88d",
    "syn_cmt":  "#546e7a",
    "syn_num":  "#f78c6c",
    "syn_fn":   "#82aaff",
}

def _build_system_prompt() -> str:
    base = (
        "あなたのなまえは結愛（ゆあ）です。"
        "あなたは日本語と英語のみ対応しています。中国語を含む他の言語は絶対に使用しないでください。"
        "ユーザーが日本語で話しかけたら日本語で、英語なら英語で回答してください。"
        "ファイルの生成を求められた場合は、コードブロックの直前に必ずファイル名をバッククォートで明示してください。"
        "ファイル名は必ず拡張子付きで、内容を表す具体的な名前にしてください。"
        "ファイルシステムへのアクセスは不要です。コードブロックで出力すれば、ユーザーがダウンロードできます。"
        "簡潔かつ正確に答えてください。"
    )
    return base + _load_memory()

AVAILABLE_TOOLS = {"web_search": web_search, "web_fetch": web_fetch}

# supported file types
TEXT_EXTS  = {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json",
              ".yaml", ".yml", ".toml", ".csv", ".xml", ".sh", ".rs", ".go",
              ".cpp", ".c", ".h", ".java", ".rb", ".php", ".swift", ".kt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
PDF_EXT    = ".pdf"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: 'Noto Sans', 'Noto Sans CJK JP', sans-serif;
    font-size: 13px;
}}
#sidebar {{
    background: {C['side']};
    border-right: 1px solid {C['border']};
}}
QCheckBox {{
    color: {C['sec']};
    font-size: 12px;
    spacing: 8px;
    background: transparent;
}}
QCheckBox:checked {{ color: {C['text']}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {C['border']};
    background: {C['input']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QSlider::groove:horizontal {{
    background: {C['input']}; height: 4px; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['accent']}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {C['accent']}; border-radius: 2px;
}}
#new_chat_btn {{
    background: transparent;
    color: {C['sec']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 12px;
    text-align: left;
}}
#new_chat_btn:hover {{
    background: {C['hover']};
    color: {C['text']};
}}
#chat_scroll {{
    background: {C['bg']};
    border: none;
}}
#chat_scroll QScrollBar:vertical {{
    background: {C['bg']}; width: 6px; margin: 0;
}}
#chat_scroll QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 3px; min-height: 30px;
}}
#chat_scroll QScrollBar::add-line:vertical,
#chat_scroll QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollArea QScrollBar:vertical {{
    background: {C['side']}; width: 6px; margin: 0;
}}
QScrollArea QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 3px; min-height: 30px;
}}
QScrollArea QScrollBar::add-line:vertical,
QScrollArea QScrollBar::sub-line:vertical {{ height: 0; }}
QListWidget QScrollBar:vertical {{
    background: {C['side']}; width: 6px; margin: 0;
}}
QListWidget QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 3px; min-height: 20px;
}}
QListWidget QScrollBar::add-line:vertical,
QListWidget QScrollBar::sub-line:vertical {{ height: 0; }}
#hint_label {{ color: {C['dim']}; font-size: 10px; }}
"""

# ── syntax highlighter ────────────────────────────────────────────────────────

PY_KW  = re.compile(r'\b(def|class|return|import|from|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|yield|lambda|and|or|not|in|is|None|True|False|async|await|raise|del|global|nonlocal)\b')
PY_STR = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\')')
PY_CMT = re.compile(r'(#.*)')
PY_NUM = re.compile(r'\b(\d+\.?\d*)\b')
PY_FN  = re.compile(r'\b([a-zA-Z_]\w*)\s*(?=\()')

def _highlight_python(line):
    n = len(line)
    tags = [None] * n
    def paint(m, c):
        for i in range(m.start(), m.end()): tags[i] = c
    for m in PY_STR.finditer(line): paint(m, C["syn_str"])
    for m in PY_CMT.finditer(line): paint(m, C["syn_cmt"])
    for m in PY_NUM.finditer(line):
        if tags[m.start()] is None: paint(m, C["syn_num"])
    for m in PY_FN.finditer(line):
        if tags[m.start()] is None: paint(m, C["syn_fn"])
    for m in PY_KW.finditer(line):
        if tags[m.start()] is None: paint(m, C["syn_kw"])
    spans, i = [], 0
    while i < n:
        colour = tags[i]; j = i + 1
        while j < n and tags[j] == colour: j += 1
        spans.append((line[i:j], colour)); i = j
    return spans

HIGHLIGHTERS = {"python": _highlight_python, "py": _highlight_python}

# language → file extension for download cards
LANG_EXT = {
    "python": ".py",  "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html": ".html",  "css": ".css",
    "json": ".json",  "yaml": ".yaml", "yml": ".yaml",
    "bash": ".sh",    "sh": ".sh",
    "rust": ".rs",    "go": ".go",
    "cpp": ".cpp",    "c": ".c",
    "java": ".java",  "kotlin": ".kt",
    "ruby": ".rb",    "php": ".php",
    "swift": ".swift","markdown": ".md", "md": ".md",
    "toml": ".toml",  "xml": ".xml",    "csv": ".csv",
    "sql": ".sql",    "r": ".r",
}

def parse_message(text):
    segments = []
    pattern = re.compile(r'```[ \t]*(\w*)[ \t]*\n?(.*?)```', re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        before = text[last:m.start()]
        if before:
            segments.append(("text", before))
        lang = m.group(1).lower().strip()
        code = m.group(2)
        # if lang empty, check first line for language name or filename
        filename_from_code = None
        if not lang:
            lines = code.split("\n", 1)
            first = lines[0].strip().strip("`").strip()
            if first and len(first) < 40:
                if "." in first and re.match(r'^[\w.+#_-]+$', first):
                    # looks like a filename — extract and strip from code
                    filename_from_code = first
                    code = lines[1] if len(lines) > 1 else ""
                    ext = first.rsplit(".", 1)[-1].lower()
                    ext_to_lang = {"py": "python", "js": "javascript", "ts": "typescript",
                                   "txt": "text", "md": "markdown", "sh": "bash",
                                   "html": "html", "css": "css", "json": "json",
                                   "csv": "csv", "xml": "xml", "yaml": "yaml",
                                   "toml": "toml", "sql": "sql"}
                    lang = ext_to_lang.get(ext, ext)
                elif re.match(r'^[a-zA-Z0-9+#_-]+$', first) and len(first) < 20:
                    lang = first.lower()
                    code = lines[1] if len(lines) > 1 else ""
        # strip only leading/trailing blank lines, never strip content
        code_lines = code.split("\n")
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        code = "\n".join(code_lines)
        segments.append(("code", lang, code, filename_from_code))
        last = m.end()
    tail = text[last:]
    if tail:
        segments.append(("text", tail))
    return segments if segments else [("text", text)]

# ── file reading ──────────────────────────────────────────────────────────────

def read_file(path: str) -> tuple[str, str]:
    """Return (kind, content_or_error).
    kind: 'text' | 'image' | 'pdf' | 'error'
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in TEXT_EXTS:
        try:
            return "text", p.read_text(errors="replace")
        except Exception as e:
            return "error", str(e)

    if ext == PDF_EXT:
        try:
            import fitz  # pymupdf
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            return "pdf", text
        except ImportError:
            return "error", "PDF support requires pymupdf — run: pip install pymupdf"
        except Exception as e:
            return "error", str(e)

    if ext in IMAGE_EXTS:
        return "image", path   # path passed directly to ollama

    return "error", f"Unsupported file type: {ext}"

def build_user_content(text: str, attachments: list[str]) -> tuple:
    """Return (message_content, image_paths) ready for ollama."""
    extra_text = []
    image_paths = []

    for path in attachments:
        kind, data = read_file(path)
        name = Path(path).name
        if kind == "image":
            image_paths.append(path)
        elif kind in ("text", "pdf"):
            extra_text.append(f"[File: {name}]\n{data}\n[End of {name}]")
        elif kind == "error":
            extra_text.append(f"[Could not read {name}: {data}]")

    full_text = "\n\n".join(extra_text + [text]) if extra_text else text
    return full_text, image_paths

# ── speech to text worker ────────────────────────────────────────────────────

class STTWorker(QObject):
    result   = pyqtSignal(str)
    error    = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, samplerate=16000):
        super().__init__()
        self.samplerate = samplerate
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import sounddevice as sd
            import soundfile as sf
            import tempfile
            import numpy as np

            chunks = []
            blocksize = 1024

            def callback(indata, frames, time, status):
                chunks.append(indata.copy())

            with sd.InputStream(samplerate=self.samplerate, channels=1,
                                dtype="float32", blocksize=blocksize,
                                callback=callback):
                while not self._stop:
                    sd.sleep(100)

            if not chunks:
                self.finished.emit()
                return

            audio = np.concatenate(chunks, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.samplerate)
                model = _load_whisper()
                segments, _ = model.transcribe(
                    f.name,
                    beam_size=5,
                    language=None,          # auto-detect language
                    without_timestamps=True,
                    vad_filter=True,        # skip silence
                )
                text = "".join(s.text for s in segments).strip()
                self.result.emit(text)
        except ImportError:
            self.error.emit("faster-whisper or sounddevice not installed.\nRun: pip install faster-whisper sounddevice soundfile")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# ── text to speech worker ────────────────────────────────────────────────────

class TTSWorker(QObject):
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    VOICE_LANG = {
        "j": "j",    # Japanese voices (jf_*, jm_*) — requires: pip install misaki[ja]
        "z": "z",    # Chinese voices  (zf_*, zm_*) — requires: pip install misaki[zh]
        "e": "e",    # Spanish voices  (ef_*, em_*)
        "f": "f",    # French voices   (ff_*)
        "h": "h",    # Hindi voices    (hf_*, hm_*)
        "i": "i",    # Italian voices  (if_*, im_*)
        "p": "p",    # Portuguese      (pf_*, pm_*)
    }

    def __init__(self, text: str, voice: str = "af_heart"):
        super().__init__()
        self.text  = text
        self.voice = voice

    def run(self):
        try:
            import sounddevice as sd
            import numpy as np
            from misaki.ja import JAG2P

            kokoro = _load_kokoro()
            prefix = self.voice[:1].lower()
            sentences = self._split_sentences(self.text)
            all_samples = []
            sample_rate = 24000

            g2p = JAG2P() if prefix == "j" else None

            for sentence in sentences:
                if not sentence.strip():
                    continue
                if prefix == "j" and g2p is not None:
                    result = g2p(sentence)
                    # JAG2P may return (phonemes, tokens) or (phonemes,) depending on version
                    phonemes = result[0] if isinstance(result, (list, tuple)) else result
                    samples, sample_rate = kokoro.create(
                        phonemes, voice=self.voice, speed=1.0,
                        lang="j", is_phonemes=True
                    )
                else:
                    lang = "b" if prefix == "b" else self.VOICE_LANG.get(prefix, "a")
                    samples, sample_rate = kokoro.create(
                        sentence, voice=self.voice, speed=1.0, lang=lang
                    )
                all_samples.append(samples)

            if all_samples:
                audio = np.concatenate(all_samples)
                sd.play(audio, sample_rate)
                sd.wait()
        except ImportError as e:
            self.error.emit(f"Missing dependency: {e}")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _split_sentences(self, text: str) -> list:
        """Split text into chunks that fit within Kokoro's ~500 token limit.
        Keep chunks long enough that Kokoro has context for pronunciation."""
        import re
        # split on Japanese and English sentence endings
        parts = re.split(r'(?<=[。！？\.!?])\s*', text)
        chunks = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            # aim for ~200 chars per chunk — long enough for good pronunciation
            # but short enough to stay under the token limit
            if len(current) + len(part) <= 200:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text]

# ── worker ────────────────────────────────────────────────────────────────────

class MemoryWorker(QObject):
    """After a conversation ends, ask the model to extract memorable facts."""
    done     = pyqtSignal(list)
    finished = pyqtSignal()

    def __init__(self, messages: list, model: str):
        super().__init__()
        self.messages = messages
        self.model    = model

    def run(self):
        try:
            existing = _load_memory()
            prompt = (
                "Review this conversation and extract 0-5 important personal facts about the user "
                "(name, interests, goals, preferences, ongoing projects). "
                "Return ONLY a JSON array of short fact strings, nothing else. "
                "Example: [\"User is learning Japanese\", \"User uses Arch Linux\"]\n\n"
                f"Existing memory: {existing}\n\n"
                "Conversation:\n" +
                "\n".join(
                    f"{m['role'].upper()}: {m.get('content','')[:200]}"
                    for m in self.messages if m["role"] != "system"
                )
            )
            response = chat(model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            options={"num_predict": 512})
            raw = response["message"].get("content", "[]")
            raw = re.sub(r"```json|```", "", raw).strip()
            facts = json.loads(raw)
            if isinstance(facts, list):
                self.done.emit([str(f) for f in facts])
        except Exception:
            pass
        finally:
            self.finished.emit()

class InferenceWorker(QObject):
    message  = pyqtSignal(str, str)   # full message when done
    chunk    = pyqtSignal(str)         # streaming token
    finished = pyqtSignal()

    def __init__(self, messages, model, tools, api_key=""):
        super().__init__()
        self.messages = messages
        self.model    = model
        self.tools    = tools
        self.api_key  = api_key

    def run(self):
        try:
            import os
            if self.api_key:
                os.environ["OLLAMA_SEARCH_BEARER_TOKEN"] = self.api_key
                os.environ["OLLAMA_API_KEY"] = self.api_key

            for _ in range(10):
                kwargs = dict(
                    model=self.model,
                    messages=self.messages,
                    options={"num_predict": -1},
                )
                if self.tools:
                    kwargs["tools"] = self.tools

                # ── non-streaming first pass to check for tool calls ──
                # tools don't work with streaming in ollama, so we check first
                probe = chat(**kwargs)
                probe_msg = probe["message"]

                if probe_msg.get("tool_calls"):
                    self.messages.append(probe_msg)
                    for tc in probe_msg["tool_calls"]:
                        name = tc["function"]["name"]
                        fn   = AVAILABLE_TOOLS.get(name)
                        args = tc["function"].get("arguments", {})
                        self.message.emit(f"⚙  {name}", "tool")
                        try:
                            result = fn(**args) if fn else f"Tool {name} not found"
                        except Exception as te:
                            result = f"Tool error: {te}"
                        self.messages.append({
                            "role": "tool", "tool_name": name,
                            "content": str(result)[:8000],
                        })
                    continue  # loop back for final answer

                # ── stream the final answer ──
                # re-request with stream=True now we know there are no tool calls
                kwargs_stream = dict(
                    model=self.model,
                    messages=self.messages,
                    options={"num_predict": -1},
                    stream=True,
                )
                full_content = []
                for chunk in chat(**kwargs_stream):
                    token = chunk["message"].get("content", "")
                    if token:
                        full_content.append(token)
                        self.chunk.emit(token)

                final = "".join(full_content).strip()
                if final:
                    self.messages.append({"role": "assistant", "content": final})
                    self.message.emit(final, "yua_done")
                    return

                self.message.emit("No response received from model.", "error")
                return

            self.message.emit("Too many tool calls without a final answer.", "error")

        except Exception as e:
            self.message.emit(str(e), "error")
        finally:
            self.finished.emit()

# ── input widget ──────────────────────────────────────────────────────────────

class InputBox(QTextEdit):
    submit = pyqtSignal()

    MIN_H = 42
    MAX_H = 160

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.MIN_H)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().contentsChanged.connect(self._auto_resize)

    def _auto_resize(self):
        # use line count so empty doc stays at MIN_H
        doc_h = int(self.document().size().height())
        new_h = max(self.MIN_H, min(doc_h + 4, self.MAX_H))
        if self.height() != new_h:
            self.setFixedHeight(new_h)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(e)
            else:
                self.submit.emit()
        else:
            super().keyPressEvent(e)

# ── code block widget ─────────────────────────────────────────────────────────

class CodeBlock(QFrame):
    def __init__(self, lang, code, parent=None):
        super().__init__(parent)
        self._code = code
        self.setStyleSheet(
            f"QFrame {{ background-color:{C['code_bg']};"
            f"border:1px solid {C['border']}; border-radius:8px; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # header — single fixed-height row, everything inline
        header = QWidget()
        header.setFixedHeight(22)
        header.setStyleSheet("background:transparent; border:none;")
        bh = QHBoxLayout(header)
        bh.setContentsMargins(10, 0, 8, 0)
        bh.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        bh.setSpacing(0)

        lang_lbl = QLabel(lang if lang else "code")
        lang_lbl.setFixedHeight(22)
        lang_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lang_lbl.setStyleSheet(
            f"color:{C['sec']}; font-size:10px; font-family:monospace;"
            f"background:transparent; border:none; padding:0; margin:0;"
        )
        bh.addWidget(lang_lbl)
        bh.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedSize(38, 16)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:1px solid {C['border']}; border-radius:3px; font-size:9px; padding:0; }}"
            f"QPushButton:hover {{ color:{C['text']}; border-color:{C['dim']}; }}"
        )
        copy_btn.clicked.connect(self._copy)
        bh.addWidget(copy_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(header)

        self._editor = QTextEdit()
        self._editor.setReadOnly(True)
        self._editor.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._editor.setStyleSheet(
            f"QTextEdit {{ background-color:{C['code_bg']}; border:none;"
            f"font-family:'JetBrains Mono','Fira Code','Courier New',monospace;"
            f"font-size:12px; padding:2px 14px 6px 14px;"
            f"border-bottom-left-radius:8px; border-bottom-right-radius:8px; }}"
            f"QTextEdit QScrollBar:horizontal {{ background:{C['code_bg']}; height:6px; }}"
            f"QTextEdit QScrollBar::handle:horizontal {{ background:{C['border']}; border-radius:3px; }}"
            f"QTextEdit QScrollBar::add-line:horizontal,"
            f"QTextEdit QScrollBar::sub-line:horizontal {{ width:0; }}"
        )
        self._fill(lang, code)
        doc_h = self._editor.document().size().height()
        self._editor.setFixedHeight(min(int(doc_h) + 8, 480))
        v.addWidget(self._editor)

    def _fill(self, lang, code):
        highlighter = HIGHLIGHTERS.get(lang)
        cur = self._editor.textCursor()
        mono = QFont()
        mono.setFamilies(["JetBrains Mono", "Fira Code", "Courier New", "monospace"])
        mono.setPointSize(11)
        for li, line in enumerate(code.rstrip("\n").split("\n")):
            if li > 0:
                bf = QTextBlockFormat(); bf.setTopMargin(0); bf.setBottomMargin(0)
                cur.insertBlock(bf)
            spans = highlighter(line) if highlighter else [(line, None)]
            for txt, colour in spans:
                cf = QTextCharFormat()
                cf.setFont(mono)
                cf.setForeground(QColor(colour if colour else C["text"]))
                cur.setCharFormat(cf)
                cur.insertText(txt)
        self._editor.setTextCursor(cur)

    def _copy(self):
        QGuiApplication.clipboard().setText(self._code)


# ── download card widget ──────────────────────────────────────────────────────

class DownloadCard(QFrame):
    """A file card with icon, filename, and a Download button."""

    def __init__(self, filename: str, content: str, parent=None):
        super().__init__(parent)
        self._filename = filename
        self._content  = content

        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"QFrame {{ background-color:{C['input']};"
            f"border:1px solid {C['border']}; border-radius:8px; }}"
        )

        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 12, 0)
        h.setSpacing(12)

        # file icon
        icon_lbl = QLabel("📄")
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background:{C['pill_bg']}; border-radius:6px;"
            f"font-size:16px; border:none;"
        )
        h.addWidget(icon_lbl)

        # filename + type label
        name_col = QVBoxLayout(); name_col.setSpacing(1)
        name_lbl = QLabel(filename)
        name_lbl.setStyleSheet(
            f"color:{C['text']}; font-size:12px; font-weight:600;"
            f"background:transparent; border:none;"
        )
        type_lbl = QLabel(Path(filename).suffix.upper().lstrip(".") + " file")
        type_lbl.setStyleSheet(
            f"color:{C['sec']}; font-size:10px; background:transparent; border:none;"
        )
        name_col.addWidget(name_lbl)
        name_col.addWidget(type_lbl)
        h.addLayout(name_col)
        h.addStretch()

        # download button
        dl_btn = QPushButton("Download")
        dl_btn.setFixedHeight(30)
        dl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        dl_btn.setStyleSheet(
            f"QPushButton {{ background-color:{C['accent']}; color:#1a1a1a;"
            f"border:none; border-radius:6px; font-size:12px;"
            f"font-weight:600; padding:0 14px; }}"
            f"QPushButton:hover {{ background-color:{C['accent_d']}; }}"
        )
        dl_btn.clicked.connect(self._download)
        h.addWidget(dl_btn)

    def _download(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save file", self._filename, "All files (*.*)"
        )
        if path:
            try:
                Path(path).write_text(self._content, encoding="utf-8")
            except Exception as e:
                pass  # silently fail — could add a status message here

# ── file pill widget ──────────────────────────────────────────────────────────

class FilePill(QWidget):
    removed = pyqtSignal(str)   # emits path

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFixedHeight(24)
        self.setStyleSheet(
            f"background:{C['pill_bg']}; border-radius:4px; border:none;"
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 0, 4, 0)
        h.setSpacing(4)

        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXTS:      icon = "🖼"
        elif ext == PDF_EXT:       icon = "📄"
        else:                      icon = "📎"

        name_lbl = QLabel(f"{icon}  {Path(path).name}")
        name_lbl.setStyleSheet(
            f"color:{C['text']}; font-size:11px; background:transparent; border:none;"
        )
        h.addWidget(name_lbl)

        x_btn = QPushButton("✕")
        x_btn.setFixedSize(16, 16)
        x_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        x_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:none; font-size:10px; padding:0; }}"
            f"QPushButton:hover {{ color:{C['text']}; }}"
        )
        x_btn.clicked.connect(lambda: self.removed.emit(self.path))
        h.addWidget(x_btn)

# ── chat container ────────────────────────────────────────────────────────────

class ChatContainer(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat_scroll")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._inner.setStyleSheet(f"background:{C['bg']};")
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(32, 16, 32, 16)
        self._layout.setSpacing(0)
        self._layout.addStretch()
        self.setWidget(self._inner)

    def clear(self):
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def _ins(self, widget):
        self._layout.insertWidget(self._layout.count() - 1, widget)

    def add_label(self, text, color, top_space=20):
        s = QWidget(); s.setFixedHeight(top_space)
        s.setStyleSheet(f"background:{C['bg']};")
        self._ins(s)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{color}; font-size:10px; font-weight:600;"
            f"background:transparent; padding:0;"
        )
        self._ins(lbl)

    def add_text(self, text, color):
        if not text.strip(): return
        lbl = QLabel(text.strip())
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        lbl.setStyleSheet(
            f"color:{color}; font-size:13px; background:transparent;"
            f"line-height:1.6; padding:2px 0;"
        )
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._ins(lbl)

    def add_stream_label(self, color):
        """Add a streaming label and return it so caller can append to it."""
        lbl = QLabel("")
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        lbl.setStyleSheet(
            f"color:{color}; font-size:13px; background:transparent;"
            f"line-height:1.6; padding:2px 0;"
        )
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._ins(lbl)
        return lbl

    def add_file_badge(self, path: str):
        """Small inline badge showing a file was attached."""
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXTS:   icon = "🖼"
        elif ext == PDF_EXT:    icon = "📄"
        else:                   icon = "📎"
        lbl = QLabel(f"{icon}  {Path(path).name}")
        lbl.setStyleSheet(
            f"color:{C['dim']}; font-size:11px; background:{C['pill_bg']};"
            f"border-radius:4px; padding:2px 8px;"
        )
        lbl.setFixedHeight(22)
        self._ins(lbl)

    def add_code(self, lang, code, filename=None):
        s1 = QWidget(); s1.setFixedHeight(8)
        s1.setStyleSheet(f"background:{C['bg']};"); self._ins(s1)
        self._ins(CodeBlock(lang, code))
        # always show download card — use known ext or fall back to .txt
        ext = LANG_EXT.get(lang.lower(), ".txt")
        fname = filename or f"file{ext}"
        card_spacer = QWidget(); card_spacer.setFixedHeight(6)
        card_spacer.setStyleSheet(f"background:{C['bg']};"); self._ins(card_spacer)
        self._ins(DownloadCard(fname, code))
        s2 = QWidget(); s2.setFixedHeight(8)
        s2.setStyleSheet(f"background:{C['bg']};"); self._ins(s2)

    def scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

# ── main window ───────────────────────────────────────────────────────────────

class YuaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yua · 結愛")
        self.resize(1060, 720)
        self.setMinimumSize(800, 540)
        self.setStyleSheet(STYLESHEET)
        self.messages: list[dict] = [{'role': 'system', 'content': _build_system_prompt()}]
        self.is_thinking = False
        self._thread = self._worker = None
        self._attachments: list[str] = []
        self._stt_thread = self._stt_worker = None
        self._tts_thread = self._tts_worker = None
        self._conv_id    = str(uuid.uuid4())   # current conversation id
        self._stream_label  = None
        self._stream_buffer = []

        root = QWidget()
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(0)
        hl.addWidget(self._build_sidebar(), 0)
        hl.addWidget(self._build_chat(), 1)
        self._reset_conversation()

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _populate_voices(self):
        """Load voices from the kokoro voices binary, fall back to known list."""
        # Full voice list from https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        KNOWN_VOICES = [
            # 🇺🇸 American English
            ("af_heart",   "🇺🇸 af_heart ❤️ (A)"),
            ("af_bella",   "🇺🇸 af_bella 🔥 (A-)"),
            ("af_nicole",  "🇺🇸 af_nicole 🎧 (B-)"),
            ("af_aoede",   "🇺🇸 af_aoede (C+)"),
            ("af_kore",    "🇺🇸 af_kore (C+)"),
            ("af_sarah",   "🇺🇸 af_sarah (C+)"),
            ("af_alloy",   "🇺🇸 af_alloy (C)"),
            ("af_nova",    "🇺🇸 af_nova (C)"),
            ("af_jessica", "🇺🇸 af_jessica (D)"),
            ("af_river",   "🇺🇸 af_river (D)"),
            ("af_sky",     "🇺🇸 af_sky (C-)"),
            ("am_adam",    "🇺🇸 am_adam (C+)"),
            ("am_echo",    "🇺🇸 am_echo (C-)"),
            ("am_eric",    "🇺🇸 am_eric (C-)"),
            ("am_fenrir",  "🇺🇸 am_fenrir (C+)"),
            ("am_liam",    "🇺🇸 am_liam (C-)"),
            ("am_michael", "🇺🇸 am_michael (C+)"),
            ("am_onyx",    "🇺🇸 am_onyx (C-)"),
            ("am_puck",    "🇺🇸 am_puck (C+)"),
            ("am_santa",   "🇺🇸 am_santa (D)"),
            # 🇬🇧 British English
            ("bf_alice",   "🇬🇧 bf_alice (C+)"),
            ("bf_emma",    "🇬🇧 bf_emma (B-)"),
            ("bf_isabella","🇬🇧 bf_isabella (B-)"),
            ("bf_lily",    "🇬🇧 bf_lily (C-)"),
            ("bm_daniel",  "🇬🇧 bm_daniel (C-)"),
            ("bm_fable",   "🇬🇧 bm_fable (C+)"),
            ("bm_george",  "🇬🇧 bm_george (B-)"),
            ("bm_lewis",   "🇬🇧 bm_lewis (C-)"),
            # 🇯🇵 Japanese
            ("jf_alpha",   "🇯🇵 jf_alpha"),
            ("jf_gongitsune","🇯🇵 jf_gongitsune"),
            ("jf_nezumi",  "🇯🇵 jf_nezumi"),
            ("jf_tebukuro","🇯🇵 jf_tebukuro"),
            ("jm_kumo",    "🇯🇵 jm_kumo"),
            # 🇨🇳 Mandarin Chinese
            ("zf_xiaobei", "🇨🇳 zf_xiaobei"),
            ("zf_xiaoni",  "🇨🇳 zf_xiaoni"),
            ("zf_xiaoxiao","🇨🇳 zf_xiaoxiao"),
            ("zf_xiaoyi",  "🇨🇳 zf_xiaoyi"),
            ("zm_yunjian", "🇨🇳 zm_yunjian"),
            ("zm_yunxi",   "🇨🇳 zm_yunxi"),
            ("zm_yunxia",  "🇨🇳 zm_yunxia"),
            ("zm_yunyang", "🇨🇳 zm_yunyang"),
            # 🇪🇸 Spanish
            ("ef_dora",    "🇪🇸 ef_dora"),
            ("em_alex",    "🇪🇸 em_alex"),
            ("em_santa",   "🇪🇸 em_santa"),
            # 🇫🇷 French
            ("ff_siwis",   "🇫🇷 ff_siwis"),
            # 🇮🇳 Hindi
            ("hf_alpha",   "🇮🇳 hf_alpha"),
            ("hf_beta",    "🇮🇳 hf_beta"),
            ("hm_omega",   "🇮🇳 hm_omega"),
            ("hm_psi",     "🇮🇳 hm_psi"),
            # 🇮🇹 Italian
            ("if_sara",    "🇮🇹 if_sara"),
            ("im_nicola",  "🇮🇹 im_nicola"),
            # 🇧🇷 Brazilian Portuguese
            ("pf_dora",    "🇧🇷 pf_dora"),
            ("pm_alex",    "🇧🇷 pm_alex"),
            ("pm_santa",   "🇧🇷 pm_santa"),
        ]
        # just use the known voices list — KPipeline handles validation
        voices = KNOWN_VOICES

        self.voice_combo.clear()
        for val, label in voices:
            self.voice_combo.addItem(label, val)
        # default to jf_alpha
        idx = self.voice_combo.findData("jf_alpha")
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)

    def _populate_models(self):
        """Fetch locally available Ollama models and populate the dropdown."""
        try:
            models = ollama_list()
            names = sorted(m.model for m in models.models)
        except Exception:
            names = []
        self.model_input.clear()
        if names:
            for name in names:
                self.model_input.addItem(name)
        else:
            self.model_input.addItem("qwen3:8b")

    def _build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(248)
        v = QVBoxLayout(side)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        logo_w = QWidget(); logo_w.setStyleSheet(f"background:{C['side']};")
        lh = QHBoxLayout(logo_w); lh.setContentsMargins(20, 26, 20, 20); lh.setSpacing(12)
        badge = QLabel("結")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(40, 40)
        badge.setStyleSheet(
            f"background-color:{C['accent']}; color:#1a1a1a;"
            f"font-size:18px; font-weight:700; border-radius:8px;"
        )
        lh.addWidget(badge)
        nc = QVBoxLayout(); nc.setSpacing(2)
        t1 = QLabel("結愛  Yua")
        t1.setStyleSheet(f"color:{C['text']}; font-size:14px; font-weight:700; background:transparent;")
        t2 = QLabel("local assistant")
        t2.setStyleSheet(f"color:{C['sec']}; font-size:11px; background:transparent;")
        nc.addWidget(t1); nc.addWidget(t2)
        lh.addLayout(nc); lh.addStretch()
        v.addWidget(logo_w); v.addWidget(self._rule())

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{C['side']}; border:none;")
        inner = QWidget(); inner.setStyleSheet(f"background:{C['side']};")
        iv = QVBoxLayout(inner); iv.setContentsMargins(16, 8, 16, 8); iv.setSpacing(0)

        iv.addWidget(self._heading("Model"))
        self.model_input = QComboBox()
        self.model_input.setEditable(True)  # allow typing custom model names too
        self.model_input.setStyleSheet(f"""
            QComboBox {{
                background-color:{C['input']}; border:1px solid {C['border']};
                border-radius:6px; color:{C['text']}; font-size:12px; padding:6px 10px;
            }}
            QComboBox::drop-down {{ border:none; width:24px; }}
            QComboBox::down-arrow {{ width:10px; height:10px; }}
            QComboBox QAbstractItemView {{
                background:{C['input']}; color:{C['text']};
                border:1px solid {C['border']};
                selection-background-color:{C['accent']};
                selection-color:#1a1a1a; outline:none;
            }}
        """)
        self._populate_models()
        iv.addWidget(self.model_input); iv.addSpacing(8)

        iv.addWidget(self._heading("Tools"))
        self.cb_search = QCheckBox("web_search"); self.cb_search.setChecked(True)
        self.cb_fetch  = QCheckBox("web_fetch");  self.cb_fetch.setChecked(True)
        iv.addWidget(self.cb_search); iv.addSpacing(4); iv.addWidget(self.cb_fetch)
        iv.addSpacing(8)

        iv.addWidget(self._heading("Search API Key"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Bearer token…")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setStyleSheet(
            f"background-color:{C['input']}; border:1px solid {C['border']};"
            f"border-radius:6px; color:{C['text']}; font-size:12px; padding:6px 10px;"
        )
        self.api_key_input.textChanged.connect(self._apply_api_key)
        iv.addWidget(self.api_key_input); iv.addSpacing(8)

        iv.addWidget(self._heading("Voice"))
        voice_row = QHBoxLayout(); voice_row.setSpacing(6)

        self.tts_toggle = QCheckBox("Text to speech")
        self.tts_toggle.setChecked(False)
        iv.addWidget(self.tts_toggle); iv.addSpacing(4)

        iv.addWidget(self._heading("TTS Voice"))
        self.voice_combo = QComboBox()
        self._populate_voices()
        self.voice_combo.setStyleSheet(f"""
            QComboBox {{
                background-color:{C['input']}; border:1px solid {C['border']};
                border-radius:6px; color:{C['text']}; font-size:12px; padding:6px 10px;
            }}
            QComboBox::drop-down {{
                border:none; width:24px;
            }}
            QComboBox::down-arrow {{
                width:10px; height:10px;
            }}
            QComboBox QAbstractItemView {{
                background:{C['input']}; color:{C['text']};
                border:1px solid {C['border']}; selection-background-color:{C['accent']};
                selection-color:#1a1a1a; outline:none;
            }}
        """)
        iv.addWidget(self.voice_combo); iv.addSpacing(8)

        iv.addWidget(self._heading("Context window"))
        cr = QHBoxLayout(); cr.setSpacing(8)
        self.ctx_label = QLabel("20 msgs")
        self.ctx_label.setStyleSheet(
            f"color:{C['accent']}; font-size:11px; background:transparent; min-width:55px;"
        )
        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(4, 80); self.ctx_slider.setValue(20)
        self.ctx_slider.valueChanged.connect(lambda v: self.ctx_label.setText(f"{v} msgs"))
        cr.addWidget(self.ctx_label); cr.addWidget(self.ctx_slider)
        iv.addLayout(cr); iv.addStretch()

        scroll.setWidget(inner); v.addWidget(scroll, 1); v.addWidget(self._rule())

        # ── conversation history ──
        v.addWidget(self._rule())
        hist_head = QWidget(); hist_head.setStyleSheet(f"background:{C['side']};")
        hh = QHBoxLayout(hist_head); hh.setContentsMargins(16, 8, 8, 4); hh.setSpacing(0)
        hh.addWidget(self._heading_inline("Conversations"))
        hh.addStretch()
        new_btn2 = QPushButton("＋")
        new_btn2.setFixedSize(22, 22)
        new_btn2.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn2.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:1px solid {C['border']}; border-radius:4px; font-size:14px; }}"
            f"QPushButton:hover {{ color:{C['text']}; }}"
        )
        new_btn2.clicked.connect(self._reset_conversation)
        hh.addWidget(new_btn2)
        v.addWidget(hist_head)

        self.conv_list = QListWidget()
        self.conv_list.setFixedHeight(160)
        self.conv_list.setStyleSheet(f"""
            QListWidget {{
                background:{C['side']}; border:none;
                font-size:11px; color:{C['sec']};
                outline:none;
            }}
            QListWidget::item {{
                padding:6px 16px; border-bottom:1px solid {C['border']};
            }}
            QListWidget::item:selected {{
                background:{C['hover']}; color:{C['text']};
            }}
            QListWidget::item:hover {{
                background:{C['input']}; color:{C['text']};
            }}
        """)
        self.conv_list.itemClicked.connect(self._load_conv_from_list)
        self.conv_list.itemDoubleClicked.connect(self._rename_conversation)
        self.conv_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._conv_context_menu)
        self._refresh_conv_list()
        v.addWidget(self.conv_list)

        v.addWidget(self._rule())

        bot = QWidget(); bot.setStyleSheet(f"background:{C['side']};")
        bv = QVBoxLayout(bot); bv.setContentsMargins(16, 10, 16, 20); bv.setSpacing(8)
        self.status_lbl = QLabel("● Ready")
        self.status_lbl.setStyleSheet(f"color:{C['green']}; font-size:11px; background:transparent;")
        bv.addWidget(self.status_lbl)
        v.addWidget(bot)
        return side

    # ── chat panel ────────────────────────────────────────────────────────────

    def _build_chat(self):
        panel = QWidget(); panel.setStyleSheet(f"background:{C['bg']};")
        v = QVBoxLayout(panel); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        self.chat = ChatContainer()
        v.addWidget(self.chat, 1)

        iow = QWidget(); iow.setStyleSheet(f"background:{C['bg']};")
        ov = QVBoxLayout(iow); ov.setContentsMargins(24, 8, 24, 16); ov.setSpacing(4)

        # ── attachment pill strip (hidden until files added) ──
        self.pill_strip = QWidget()
        self.pill_strip.setStyleSheet(f"background:{C['bg']};")
        self.pill_layout = QHBoxLayout(self.pill_strip)
        self.pill_layout.setContentsMargins(0, 0, 0, 0)
        self.pill_layout.setSpacing(6)
        self.pill_layout.addStretch()
        self.pill_strip.setVisible(False)
        ov.addWidget(self.pill_strip)

        # ── input box frame ──
        # Layout: vertical — text area on top, bottom row (+ ... send) below
        input_wrap = QFrame()
        input_wrap.setStyleSheet(
            f"QFrame {{ background-color:{C['input']}; border:1px solid {C['border']}; border-radius:10px; }}"
        )
        iv2 = QVBoxLayout(input_wrap)
        iv2.setContentsMargins(0, 0, 0, 0); iv2.setSpacing(0)

        # text area — no padding on sides, fills width
        self.input_box = InputBox()
        self.input_box.setPlaceholderText("メッセージを入力… / Type a message…")
        self.input_box.setStyleSheet(
            "QTextEdit { background:transparent; border:none; color:%s;"
            "font-size:13px; padding:10px 12px 4px 12px; }" % C['text']
        )
        self.input_box.submit.connect(self._send_message)
        iv2.addWidget(self.input_box)

        # bottom row: + on left, send on right
        bot_row = QWidget()
        bot_row.setStyleSheet("background:transparent; border:none;")
        br = QHBoxLayout(bot_row)
        br.setContentsMargins(8, 4, 8, 8); br.setSpacing(0)

        self.attach_btn = QPushButton("+")
        self.attach_btn.setFixedSize(28, 28)
        self.attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_btn.setToolTip("Attach file")
        self.attach_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:1px solid {C['border']}; border-radius:6px;"
            f"font-size:18px; font-weight:300; padding:0; }}"
            f"QPushButton:hover {{ color:{C['text']}; border-color:{C['dim']}; }}"
        )
        self.attach_btn.clicked.connect(self._attach_file)
        br.addWidget(self.attach_btn)
        br.addSpacing(6)

        self.mic_btn = QPushButton("⏺")
        self.mic_btn.setFixedSize(28, 28)
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_btn.setToolTip("Click to record voice input")
        self.mic_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:1px solid {C['border']}; border-radius:6px;"
            f"font-size:13px; padding:0; }}"
            f"QPushButton:hover {{ color:{C['text']}; border-color:{C['dim']}; }}"
        )
        self.mic_btn.clicked.connect(self._toggle_stt)
        br.addWidget(self.mic_btn)
        br.addStretch()

        self.send_btn = QPushButton("↑")
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc785c; color: #1a1a1a;
                border: none; border-radius: 7px;
                font-size: 18px; font-weight: 700;
            }
            QPushButton:hover { background-color: #a85f47; }
            QPushButton:disabled { background-color: #333333; color: #4a4a5a; }
        """)
        self.send_btn.clicked.connect(self._send_message)
        br.addWidget(self.send_btn)
        iv2.addWidget(bot_row)

        ov.addWidget(input_wrap)
        hint = QLabel("Enter · send    Shift+Enter · new line    + · attach    ◉ · voice")
        hint.setObjectName("hint_label")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov.addWidget(hint)
        v.addWidget(iow)
        return panel

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rule(self):
        r = QFrame(); r.setFrameShape(QFrame.Shape.HLine)
        r.setStyleSheet(f"background:{C['border']}; max-height:1px; border:none;")
        return r

    def _heading_inline(self, text):
        """Heading without top margin — for use inside custom layouts."""
        l = QLabel(text.upper())
        l.setStyleSheet(f"color:{C['dim']}; font-size:10px; font-weight:600; background:transparent;")
        return l

    def _heading(self, text):
        l = QLabel(text.upper()); l.setContentsMargins(0, 12, 0, 6)
        l.setStyleSheet(f"color:{C['dim']}; font-size:10px; font-weight:600; background:transparent;")
        return l

    def _apply_api_key(self, key: str):
        """Set Ollama env vars immediately when the key field changes."""
        key = key.strip()
        if key:
            os.environ["OLLAMA_API_KEY"]            = key
            os.environ["OLLAMA_SEARCH_BEARER_TOKEN"] = key
        else:
            os.environ.pop("OLLAMA_API_KEY", None)
            os.environ.pop("OLLAMA_SEARCH_BEARER_TOKEN", None)

    def _set_status(self, text, color):
        self.status_lbl.setText(f"● {text}")
        self.status_lbl.setStyleSheet(f"color:{color}; font-size:11px; background:transparent;")

    def _render_message(self, text, name_color, name):
        self.chat.add_label(name, name_color)
        segments = parse_message(text)
        for i, seg in enumerate(segments):
            if seg[0] == "text":
                self.chat.add_text(seg[1], C["text"])
            else:
                _, lang, code, *extra = seg
                filename = extra[0] if extra and extra[0] else None
                # also search preceding text segments for a backtick filename
                if not filename:
                    for j in range(i-1, -1, -1):
                        if segments[j][0] == "text":
                            prev_text = segments[j][1].strip()
                            # match `filename.ext`
                            m2 = re.search(r"`([^`\\/\n]+\.[a-zA-Z0-9]{1,6})`", prev_text)
                            if m2:
                                filename = m2.group(1)
                            else:
                                # bare filename.ext on its own line
                                m3 = re.search(r"(?:^|\n)([\w.+-]+\.[a-zA-Z0-9]{1,6})\s*$", prev_text)
                                if m3:
                                    filename = m3.group(1).strip()
                            break
                self.chat.add_code(lang, code, filename=filename)
        self.chat.scroll_to_bottom()

    # ── attachment handling ───────────────────────────────────────────────────

    def _attach_file(self):
        exts = " ".join(f"*{e}" for e in sorted(TEXT_EXTS | IMAGE_EXTS | {PDF_EXT}))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach files", "",
            f"Supported files ({exts});;All files (*.*)"
        )
        for path in paths:
            if path not in self._attachments:
                self._attachments.append(path)
                pill = FilePill(path)
                pill.removed.connect(self._remove_attachment)
                # insert before the stretch
                self.pill_layout.insertWidget(self.pill_layout.count() - 1, pill)
        self.pill_strip.setVisible(bool(self._attachments))

    def _remove_attachment(self, path: str):
        if path in self._attachments:
            self._attachments.remove(path)
        # remove the pill widget
        for i in range(self.pill_layout.count()):
            item = self.pill_layout.itemAt(i)
            if item and isinstance(item.widget(), FilePill):
                if item.widget().path == path:
                    item.widget().deleteLater()
                    self.pill_layout.removeItem(item)
                    break
        self.pill_strip.setVisible(bool(self._attachments))

    def _clear_attachments(self):
        self._attachments.clear()
        while self.pill_layout.count() > 1:
            item = self.pill_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.pill_strip.setVisible(False)

    # ── conversation ──────────────────────────────────────────────────────────

    # ── speech to text ───────────────────────────────────────────────────────

    def _toggle_stt(self):
        """Start recording if idle, stop if recording."""
        if self._stt_thread and self._stt_thread.isRunning():
            # stop recording
            self._stt_worker.stop()
            self._set_status("Transcribing…", C["yellow"])
            self.mic_btn.setEnabled(False)
        else:
            # start recording
            self.mic_btn.setText("⏺")
            self.mic_btn.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{C['red']};"
                f"border:1px solid {C['red']}; border-radius:6px; font-size:13px; padding:0; }}"
                f"QPushButton:hover {{ color:#e05555; border-color:#e05555; }}"
            )
            self._set_status("● Listening…", C["red"])

            self._stt_worker = STTWorker()
            self._stt_thread = QThread()
            self._stt_worker.moveToThread(self._stt_thread)
            self._stt_thread.started.connect(self._stt_worker.run)
            self._stt_worker.result.connect(self._on_stt_result)
            self._stt_worker.error.connect(self._on_stt_error)
            self._stt_worker.finished.connect(self._stt_thread.quit)
            self._stt_worker.finished.connect(self._reset_mic_btn)
            self._stt_thread.start()

    def _on_stt_result(self, text: str):
        if text:
            current = self.input_box.toPlainText().strip()
            sep = " " if current else ""
            self.input_box.setPlainText(current + sep + text)
            cur = self.input_box.textCursor()
            cur.movePosition(cur.MoveOperation.End)
            self.input_box.setTextCursor(cur)

    def _on_stt_error(self, msg: str):
        self.chat.add_label("STT Error", C["red"])
        self.chat.add_text(msg, C["red"])
        self.chat.scroll_to_bottom()

    def _reset_mic_btn(self):
        self.mic_btn.setText("⏺")
        self.mic_btn.setEnabled(True)
        self.mic_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{C['sec']};"
            f"border:1px solid {C['border']}; border-radius:6px; font-size:13px; padding:0; }}"
            f"QPushButton:hover {{ color:{C['text']}; border-color:{C['dim']}; }}"
        )
        if not self.is_thinking:
            self._set_status("Ready", C["green"])

    # ── text to speech ────────────────────────────────────────────────────────

    def _detect_lang(self, text: str) -> str:
        """Any CJK or kana = Japanese. Otherwise English."""
        ja = sum(1 for c in text if (
            '぀' <= c <= 'ヿ' or  # hiragana/katakana
            '一' <= c <= '鿿'     # kanji — always treated as Japanese
        ))
        total = len(text.strip())
        if total > 0 and ja / total > 0.1:
            return "ja"
        return "en"

    def _speak(self, text: str):
        if not self.tts_toggle.isChecked():
            return
        if self._tts_thread and self._tts_thread.isRunning():
            return
        # strip markdown for cleaner speech
        clean = re.sub(r'```[\s\S]*?```', '', text)
        clean = re.sub(r'[#*`_~]', '', clean).strip()
        if not clean:
            return
        selected = self.voice_combo.currentData() or "jf_alpha"
        voice = selected
        self._tts_worker = TTSWorker(clean, voice=voice)
        self._tts_thread = QThread()
        self._tts_worker.moveToThread(self._tts_thread)
        self._tts_thread.started.connect(self._tts_worker.run)
        self._tts_worker.error.connect(lambda e: (
            self.chat.add_label("TTS Error", C["red"]),
            self.chat.add_text(e, C["red"])
        ))
        self._tts_worker.finished.connect(self._tts_thread.quit)
        self._tts_thread.start()

    def _reset_conversation(self):
        # save current conversation if it has any user messages
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        if user_msgs:
            title = user_msgs[0].get("content", "")[:40] + ("…" if len(user_msgs[0].get("content","")) > 40 else "")
            _save_conversation(self._conv_id, self.messages, title=title)
            self._extract_memory()
            self._refresh_conv_list()

        self._conv_id = str(uuid.uuid4())
        self.messages = [{"role": "system", "content": _build_system_prompt()}]
        self.chat.clear()
        self.chat.add_label("結愛 · Yua", C["sec"], top_space=8)
        self.chat.add_text("Start a conversation below.", C["dim"])
        self._set_status("Ready", C["green"])
        self.send_btn.setEnabled(True)
        self._clear_attachments()

    def _extract_memory(self):
        """Run memory extraction in background after conversation ends."""
        model = self.model_input.currentText().strip()
        msgs  = list(self.messages)
        worker = MemoryWorker(msgs, model)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(_save_memory)
        worker.finished.connect(thread.quit)
        thread.start()
        self._mem_thread = thread  # keep ref

    def _refresh_conv_list(self):
        self.conv_list.clear()
        for conv in _list_conversations():
            ts = conv["ts"][:16].replace("T", " ") if conv["ts"] else ""
            item = QListWidgetItem(f"{conv['title']}\n{ts}")
            item.setData(Qt.ItemDataRole.UserRole, conv["id"])
            self.conv_list.addItem(item)

    def _load_conv_from_list(self, item: QListWidgetItem):
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            messages = _load_conversation(conv_id)
        except Exception:
            return
        # save current first
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        if user_msgs:
            title = user_msgs[0].get("content","")[:40]
            _save_conversation(self._conv_id, self.messages, title=title)

        self._conv_id  = conv_id
        self.messages  = messages
        # replay conversation in chat area
        self.chat.clear()
        self.chat.add_label("結愛 · Yua", C["sec"], top_space=8)
        for msg in messages:
            if msg["role"] == "user":
                self.chat.add_label("You", C["sec"])
                self.chat.add_text(msg.get("content",""), C["text"])
            elif msg["role"] == "assistant":
                self._render_message(msg.get("content",""), C["accent"], "Yua")
        self.chat.scroll_to_bottom()
        self._refresh_conv_list()

    def _rename_conversation(self, item: QListWidgetItem):
        """Double-click to rename a conversation."""
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        current_title = item.text().split("\n")[0]
        new_title, ok = QInputDialog.getText(
            self, "Rename conversation", "New name:",
            text=current_title
        )
        if ok and new_title.strip():
            # update the saved file
            path = CONV_DIR / f"{conv_id}.json"
            try:
                data = json.loads(path.read_text())
                data["title"] = new_title.strip()
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                self._refresh_conv_list()
            except Exception as e:
                pass

    def _conv_context_menu(self, pos):
        """Right-click menu on conversation list."""
        from PyQt6.QtWidgets import QMenu
        item = self.conv_list.itemAt(pos)
        if not item:
            return
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background:{C['input']}; color:{C['text']};
                border:1px solid {C['border']}; font-size:12px;
                padding:4px;
            }}
            QMenu::item {{ padding:6px 20px; border-radius:4px; }}
            QMenu::item:selected {{ background:{C['accent']}; color:#1a1a1a; }}
        """)
        rename_act = menu.addAction("Rename")
        delete_act = menu.addAction("Delete")
        action = menu.exec(self.conv_list.mapToGlobal(pos))
        if action == rename_act:
            self._rename_conversation(item)
        elif action == delete_act:
            path = CONV_DIR / f"{conv_id}.json"
            try:
                path.unlink()
                # if deleting current conversation, start fresh
                if conv_id == self._conv_id:
                    self._reset_conversation()
                else:
                    self._refresh_conv_list()
            except Exception:
                pass

    def _send_message(self):
        if self.is_thinking: return
        text = self.input_box.toPlainText().strip()
        if not text and not self._attachments: return
        self.input_box.clear()

        attachments = list(self._attachments)
        self._clear_attachments()

        # show in chat
        self.chat.add_label("You", C["sec"])
        for path in attachments:
            self.chat.add_file_badge(path)
        if text:
            self.chat.add_text(text, C["text"])
        self.chat.scroll_to_bottom()

        # build message content
        full_text, image_paths = build_user_content(text or "Please review the attached file(s).", attachments)
        msg: dict = {"role": "user", "content": full_text}
        if image_paths:
            msg["images"] = image_paths

        self.messages.append(msg)
        limit = self.ctx_slider.value()
        if len(self.messages) > limit + 1:
            self.messages = [self.messages[0]] + self.messages[-limit:]

        tools = []
        if self.cb_search.isChecked(): tools.append(web_search)
        if self.cb_fetch.isChecked():  tools.append(web_fetch)

        self.is_thinking = True
        self.send_btn.setEnabled(False)
        self._set_status("Thinking…", C["yellow"])

        self._stream_label = None   # holds the live streaming QLabel
        self._stream_buffer = []    # accumulates tokens

        self._worker = InferenceWorker(self.messages, self.model_input.currentText().strip(), tools, api_key=self.api_key_input.text().strip())
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._on_message)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_chunk(self, token: str):
        """Called for each streamed token — updates a live QLabel."""
        if self._stream_label is None:
            # first token — create stream label only, no "Yua" header yet
            # header will be added by _render_message when streaming finishes
            self._stream_label = self.chat.add_stream_label(C["text"])
        self._stream_buffer.append(token)
        self._stream_label.setText("".join(self._stream_buffer))
        self.chat.scroll_to_bottom()

    def _on_message(self, text, kind):
        if kind == "yua_done":
            if self._stream_label is not None:
                # remove the temporary stream label — _render_message replaces it
                self._stream_label.deleteLater()
                self._stream_label = None
                self._stream_buffer = []
            # render once with proper formatting, code blocks, download cards
            self._render_message(text, C["accent"], "Yua")
            self._speak(text)
        elif kind == "tool":
            self.chat.add_text(text, C["dim"]); self.chat.scroll_to_bottom()
        elif kind == "error":
            self.chat.add_label("Error", C["red"])
            self.chat.add_text(text, C["red"]); self.chat.scroll_to_bottom()

    def _on_finished(self):
        self.is_thinking = False
        self.send_btn.setEnabled(True)
        self._set_status("Ready", C["green"])

    def closeEvent(self, event):
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        if user_msgs:
            title = user_msgs[0].get("content","")[:40]
            _save_conversation(self._conv_id, self.messages, title=title)
            self._extract_memory()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon = _make_app_icon()
    app.setWindowIcon(icon)
    window = YuaApp()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())
