import os
import subprocess
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
TOKEN_TABLE = TRAIN_DIR.parent / "tokens" / "table.json"
# A Piper medium voice ONNX for the init tests; the laptop default does not exist on rented boxes.
AMY_ONNX = Path(os.environ.get(
    "GIGAPIPER_TEST_VOICE_ONNX",
    "/home/david/AndroidStudioProjects/bucket/tts/1/en/en_US/amy/medium/en_US-amy-medium.onnx",
))


def pytest_configure(config):
    sys.path.insert(0, str(TRAIN_DIR))
    kernel_dir = TRAIN_DIR / "gigatrain" / "vits" / "monotonic_align"
    if not list(kernel_dir.glob("core*.so")):
        subprocess.run([sys.executable, str(kernel_dir / "setup.py")], check=True, cwd=TRAIN_DIR)
