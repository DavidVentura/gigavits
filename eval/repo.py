"""Access to the sibling folders this harness builds on (tokens/, quality/), which are not packages."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
TOKENS = ROOT / "tokens"
QUALITY = ROOT / "quality"


def tokens_module(name: str) -> ModuleType:
    # remap.py imports `tokenizer` by its bare name, so tokens/ has to be importable as a directory.
    if str(TOKENS) not in sys.path:
        sys.path.append(str(TOKENS))
    return importlib.import_module(name)


def quality_score() -> ModuleType:
    # Loaded under its own name: eval/score.py would otherwise shadow quality/score.py.
    name = "quality_score"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, QUALITY / "score.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
