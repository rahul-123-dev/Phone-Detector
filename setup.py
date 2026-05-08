import subprocess
import sys
import os
import platform
import glob

VENV_DIR = ".venv"
REQUIRED_PYTHON = (3, 13)

DEPENDENCIES = [
    "opencv-python",
    "pygame",
    "numpy",
]

MODEL_FILES = [
    "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb",
    "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
]

# Any audio format pygame supports
SUPPORTED_AUDIO = ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.m4a"]

ASSETS_DIR = "Assets"

# ==========================================
# COLORS (auto-disable on Windows if no ANSI)
# ==========================================

def supports_color():
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("TERM") is not None
    return True

USE_COLOR = supports_color()

def c(code, text):
    return f"{code}{text}\033[0m" if USE_COLOR else text

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def ok(msg):     print(f"  {c(GREEN, '✔')}  {msg}")
def fail(msg):   print(f"  {c(RED, '✘')}  {msg}")
def warn(msg):   print(f"  {c(YELLOW, '⚠')}  {msg}")
def info(msg):   print(f"  {c(CYAN, '→')}  {msg}")
def header(msg): print(f"\n{c(BOLD+CYAN, msg)}")

# ==========================================
# PLATFORM HELPERS
# ==========================================

OS = platform.system()  # "Windows" | "Linux" | "Darwin"

def get_venv_python():
    if OS == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def get_venv_pip():
    if OS == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")

def get_activate_cmd():
    if OS == "Windows":
        return f".venv\\Scripts\\activate"
    return f"source .venv/bin/activate"

def run(cmd, capture=False):
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    subprocess.run(cmd, check=True)

# ==========================================
# STEP 1 — SYSTEM PYTHON VERSION
# ==========================================

def check_system_python():
    header("[ 1/6 ] Checking system Python version")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"

    if (v.major, v.minor) >= REQUIRED_PYTHON:
        ok(f"Python {version_str} — OK")
        return True
    else:
        fail(f"Python {version_str} found — need {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+")
        links = {
            "Windows": "https://www.python.org/downloads/windows/",
            "Darwin":  "https://www.python.org/downloads/macos/",
            "Linux":   "sudo apt install python3.13  OR  https://www.python.org/downloads/",
        }
        print(f"\n  {c(YELLOW, 'Download:')} {links.get(OS, 'https://www.python.org/downloads/')}")
        return False

# ==========================================
# STEP 2 — VENV
# ==========================================

def setup_venv():
    header("[ 2/6 ] Virtual environment")
    venv_python = get_venv_python()

    if os.path.exists(VENV_DIR) and os.path.isfile(venv_python):
        result = run([venv_python, "--version"], capture=True)
        if result.returncode == 0:
            ok(f".venv exists — {result.stdout.strip()}")
        else:
            warn("Broken .venv detected — recreating")
            _remove_venv()
            _create_venv()
    else:
        if os.path.exists(VENV_DIR):
            warn("Incomplete .venv found — recreating")
            _remove_venv()
        _create_venv()

def _remove_venv():
    import shutil
    shutil.rmtree(VENV_DIR, ignore_errors=True)

def _create_venv():
    info(f"Creating .venv ({OS})...")
    try:
        run([sys.executable, "-m", "venv", VENV_DIR])
        ok(".venv created successfully")
    except subprocess.CalledProcessError:
        fail("venv creation failed")
        if OS == "Linux":
            print(f"\n  {c(YELLOW, 'Try:')} sudo apt install python3-venv python3.13-venv")
        elif OS == "Darwin":
            print(f"\n  {c(YELLOW, 'Try:')} brew install python@3.13")
        sys.exit(1)

# ==========================================
# STEP 3 — UPGRADE PIP
# ==========================================

def upgrade_pip():
    header("[ 3/6 ] Upgrading pip")
    pip = get_venv_pip()
    try:
        run([pip, "install", "--upgrade", "pip", "-q"])
        result = run([pip, "--version"], capture=True)
        ok(result.stdout.strip())
    except subprocess.CalledProcessError:
        warn("pip upgrade failed — continuing anyway")

# ==========================================
# STEP 4 — INSTALL DEPENDENCIES
# ==========================================

def install_dependencies():
    header("[ 4/6 ] Installing dependencies")
    pip = get_venv_pip()

    # opencv-python has a known issue on some Linux distros
    # opencv-python-headless is more compatible on servers
    pkgs = list(DEPENDENCIES)
    if OS == "Linux":
        try:
            run(["python3", "-c", "import cv2"], capture=True)
        except Exception:
            pass

    for pkg in pkgs:
        pkg_name = pkg.split("==")[0].split(">=")[0].split("<=")[0]
        result = run([pip, "show", pkg_name], capture=True)

        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    ver = line.split(":", 1)[1].strip()
                    ok(f"{pkg_name:<25} already installed  (v{ver})")
                    break
        else:
            info(f"Installing {pkg_name}...")
            try:
                install_result = run([pip, "install", pkg, "-q"], capture=True) if False else None
                subprocess.run([pip, "install", pkg], check=True)
                ok(f"{pkg_name:<25} installed")
            except subprocess.CalledProcessError:
                fail(f"{pkg_name} — FAILED")
                # Platform-specific hints
                if pkg_name == "opencv-python" and OS == "Linux":
                    warn("Try: sudo apt install python3-opencv  OR  pip install opencv-python-headless")
                elif pkg_name == "pygame" and OS == "Linux":
                    warn("Try: sudo apt install python3-pygame libsdl2-dev")
                elif pkg_name == "pygame" and OS == "Darwin":
                    warn("Try: brew install sdl2 sdl2_mixer  then pip install pygame")

# ==========================================
# STEP 5 — FIND AUDIO FILE IN ASSETS
# ==========================================

def find_audio_file():
    header("[ 5/6 ] Checking Assets folder for audio")

    # Create Assets dir if missing
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        warn(f"'{ASSETS_DIR}/' folder created — put an audio file inside it")
        return None

    found_files = []
    for pattern in SUPPORTED_AUDIO:
        # Case-insensitive search on all platforms
        matches = glob.glob(os.path.join(ASSETS_DIR, pattern))
        # Also check uppercase extensions
        matches += glob.glob(os.path.join(ASSETS_DIR, pattern.upper()))
        matches += glob.glob(os.path.join(ASSETS_DIR, pattern.replace("*.", "*.").swapcase()))
        found_files.extend(matches)

    # Deduplicate
    found_files = list(dict.fromkeys(found_files))

    if not found_files:
        fail(f"No audio file found in '{ASSETS_DIR}/'")
        warn(f"Supported formats: mp3, wav, ogg, flac, aac, m4a")
        warn(f"Put any audio file in '{ASSETS_DIR}/' and re-run setup.py")
        return None

    if len(found_files) == 1:
        audio = found_files[0]
        size_kb = os.path.getsize(audio) / 1024
        ok(f"Found: {audio}  ({size_kb:.0f} KB)")
        _write_config(audio)
        return audio

    # Multiple files found — pick first, list others
    audio = found_files[0]
    ok(f"Using:  {audio}")
    for extra in found_files[1:]:
        info(f"Also found: {extra}  (not used)")

    _write_config(audio)
    return audio

def _write_config(audio_path):
    """Write a small config.py so main.py auto-picks the right audio."""
    # Normalize path separators for the OS
    normalized = os.path.normpath(audio_path)
    config_content = f'''# Auto-generated by setup.py — do not edit manually
SOUND_FILE = {repr(normalized)}
'''
    with open("config.py", "w") as f:
        f.write(config_content)
    info(f"config.py written — SOUND_FILE = {repr(normalized)}")

# ==========================================
# STEP 6 — CHECK MODEL FILES
# ==========================================

def check_model_files():
    header("[ 6/6 ] Checking model files")
    all_good = True

    for f in MODEL_FILES:
        f = os.path.normpath(f)
        if os.path.isfile(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            ok(f"{f}  ({size_mb:.1f} MB)")
        else:
            fail(f"MISSING: {f}")
            all_good = False

    if not all_good:
        sep = c(YELLOW, "━" * 52)
        dl_cmd = _model_download_commands()
        print(f"""
{sep}
{c(BOLD, 'Download model files:')}

{dl_cmd}
{sep}""")

def _model_download_commands():
    url_tar  = "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
    url_pbtxt = "https://gist.githubusercontent.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7/raw/2a20064a9d33b893dd95d2567da126d0ecd03e85/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    if OS == "Windows":
        return (
            f"  {c(CYAN, '# PowerShell:')}\n"
            f"  Invoke-WebRequest -Uri \"{url_tar}\" -OutFile model.tar.gz\n"
            f"  tar -xzf model.tar.gz\n"
            f"  Invoke-WebRequest -Uri \"{url_pbtxt}\" -OutFile ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        )
    else:
        return (
            f"  {c(CYAN, '# Linux / macOS:')}\n"
            f"  wget \"{url_tar}\"\n"
            f"  tar -xzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz\n"
            f"  wget -O ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt \\\n"
            f"    \"{url_pbtxt}\""
        )

# ==========================================
# FINAL SUMMARY
# ==========================================

def print_summary():
    activate = get_activate_cmd()
    venv_python = get_venv_python()
    sep = c(GREEN, "━" * 52)

    print(f"""
{sep}
{c(BOLD+GREEN, '  ✔  Setup complete!')}  [{OS}]

  {c(BOLD, 'Run the detector:')}

    {c(CYAN, activate)}
    {c(CYAN, 'python main.py')}

  {c(BOLD, 'Or directly:')}

    {c(CYAN, f'{venv_python} main.py')}
{sep}
""")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print(f"\n{c(BOLD+CYAN, '  Phone Detector — Setup')}")
    print(f"  OS: {OS} | Arch: {platform.machine()} | Python {sys.version.split()[0]}\n")

    if not check_system_python():
        sys.exit(1)

    setup_venv()
    upgrade_pip()
    install_dependencies()
    find_audio_file()
    check_model_files()
    print_summary()