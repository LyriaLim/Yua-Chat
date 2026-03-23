#!/usr/bin/env bash
set -e

INSTALL_DIR="$HOME/.yua"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  結愛 Yua — Installer"
echo "  ─────────────────────────────────────"
echo ""

# ── check dependencies ────────────────────────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "  ✗ $1 not found. $2"
        exit 1
    fi
    echo "  ✓ $1"
}

echo "  Checking dependencies..."
check_cmd python3   "Install Python 3.10+ from your package manager."
check_cmd pip       "Install pip: sudo pacman -S python-pip  or  sudo apt install python3-pip"
check_cmd ollama    "Install Ollama from https://ollama.com"
check_cmd git       "Install git: sudo pacman -S git  or  sudo apt install git"

# check Python version >= 3.10
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "  ✗ Python 3.10+ required (found $PY_VER)"
    exit 1
fi
echo "  ✓ Python $PY_VER"
echo ""

# ── create venv ───────────────────────────────────────────────────────────────

echo "  Setting up virtual environment at $INSTALL_DIR/venv ..."
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

# ── install Python dependencies ───────────────────────────────────────────────

echo "  Installing Python packages (this may take a few minutes)..."
pip install -r "$REPO_DIR/requirements.txt" --quiet
echo "  ✓ Python packages installed"
echo ""

# ── download Kokoro TTS model files ──────────────────────────────────────────

MODELS_DIR="$INSTALL_DIR/models"
mkdir -p "$MODELS_DIR"

download_if_missing() {
    local url="$1"
    local dest="$2"
    local name="$3"
    if [ -f "$dest" ]; then
        echo "  ✓ $name (already downloaded)"
    else
        echo "  ↓ Downloading $name..."
        wget -q --show-progress -O "$dest" "$url" || \
        curl -L --progress-bar -o "$dest" "$url"
        echo "  ✓ $name"
    fi
}

echo "  Downloading Kokoro TTS model files..."
download_if_missing \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
    "$MODELS_DIR/kokoro-v1.0.onnx" \
    "kokoro-v1.0.onnx (~310MB)"

download_if_missing \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
    "$MODELS_DIR/voices-v1.0.bin" \
    "voices-v1.0.bin (~12MB)"

echo ""

# ── copy app files ────────────────────────────────────────────────────────────

echo "  Installing Yua..."
cp "$REPO_DIR/yua.py" "$INSTALL_DIR/yua.py"

# create launcher script
cat > "$INSTALL_DIR/run.sh" << 'LAUNCHER'
#!/usr/bin/env bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR"
python "$INSTALL_DIR/yua.py" "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/run.sh"

# ── create desktop entry ──────────────────────────────────────────────────────

DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"

cat > "$DESKTOP_DIR/yua.desktop" << DESKTOP
[Desktop Entry]
Name=Yua
GenericName=AI Assistant
Comment=結愛 — Local AI assistant powered by Ollama
Exec=$INSTALL_DIR/run.sh
Icon=$INSTALL_DIR/icon.png
Terminal=false
Type=Application
Categories=Utility;Application;
Keywords=AI;assistant;ollama;chat;japanese;
StartupWMClass=yua
DESKTOP

# generate a simple icon using Python
python3 - << 'ICONPY'
import os, sys
try:
    from PIL import Image, ImageDraw, ImageFont
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # rounded square background
    margin = 20
    draw.rounded_rectangle([margin, margin, size-margin, size-margin],
                             radius=40, fill="#cc785c")
    # try to draw 結 character
    try:
        font = ImageFont.truetype("/usr/share/fonts/noto/NotoSansCJK-Bold.ttc", 140)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/NotoSansCJK-Bold.ttc", 140)
        except Exception:
            font = ImageFont.load_default()
    draw.text((size//2, size//2), "結", font=font, fill="#1a1a1a", anchor="mm")
    install_dir = os.path.expanduser("~/.yua")
    img.save(f"{install_dir}/icon.png")
    print("  ✓ Icon generated")
except ImportError:
    print("  ℹ  Pillow not installed — skipping icon generation")
ICONPY

echo ""
echo "  ─────────────────────────────────────"
echo "  ✓ Yua installed successfully!"
echo ""
echo "  To launch:"
echo "    $INSTALL_DIR/run.sh"
echo ""
echo "  Or find 'Yua' in your application launcher."
echo ""
echo "  To update, run this installer again."
echo "  ─────────────────────────────────────"
echo ""
