"""Pytest bootstrap for the eval suites: puts the project root on sys.path so
`import evals.metrics` and `import recon3d...` work from any invocation dir."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
