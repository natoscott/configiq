from __future__ import annotations

import sys
from pathlib import Path

TESTED_MODELS_DIR = Path(__file__).resolve().parents[1]
if str(TESTED_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTED_MODELS_DIR))
