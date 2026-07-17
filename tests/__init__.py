"""Debug entrypoint bootstrap for execution from ``morph_graph/``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)
