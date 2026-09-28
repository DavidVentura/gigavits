import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKENS_DIR = REPO / "tokens"
DURATIONS_DIR = REPO / "durations"

# tokens/ and durations/ are sibling folders of flat modules (tokenizer, remap, expose_durations),
# shared with the training code; they are not packages.
for _shared in (TOKENS_DIR, DURATIONS_DIR):
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
