# 結愛 Yua

A local AI assistant powered by [Ollama](https://ollama.com), built with PyQt6.

## Features

- 💬 Chat with any locally installed Ollama model
- 🌐 Web search via Ollama's built-in web_search tool
- 📎 File attachments (text, PDF, images)
- 🎤 Speech to text via faster-whisper (Japanese & English)
- 🔊 Text to speech via Kokoro TTS (Japanese & English voices)
- 💾 Conversation history with rename/delete
- 🧠 Persistent memory — Yua remembers facts across conversations
- 📥 File generation with download cards
- 🌙 Dark theme

## Requirements

- Linux (tested on Arch/EndeavourOS)
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- At least one Ollama model pulled (e.g. `ollama pull qwen3:8b`)

## Install

```bash
git clone https://github.com/LyriaLim/Yua-Chat.git
cd Yua-Chat
chmod +x install.sh
./install.sh
```

The installer will:
1. Create a virtual environment at `~/.yua/venv`
2. Install all Python dependencies
3. Download Kokoro TTS model files (~320MB)
4. Create a desktop launcher

## Run

```bash
~/.yua/run.sh
```

Or find **Yua** in your application launcher.

## Update

```bash
git pull
./install.sh
```

## Uninstall

```bash
./uninstall.sh
```

Your conversations and memory (`~/.yua/conversations/`, `~/.yua/memory.json`) are kept unless you run `rm -rf ~/.yua`.

## Optional: Speech features

Speech to text and text to speech require extra packages installed by the installer automatically. For Japanese TTS to work correctly:

```bash
pip install misaki[ja]
```

## Web search

Yua uses Ollama's built-in `web_search` tool. To enable it, set your search API key in the sidebar or in your Ollama service:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_API_KEY=your_key_here"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

## Data

All data is stored in `~/.yua/`:
- `conversations/` — saved chat history (JSON)
- `memory.json` — extracted memory facts
- `models/` — Kokoro TTS model files
