"""Wrapper around IsaacLab's generate_dataset.py. Same pattern as annotate_demos.py
wrapper: defer `envs` import until after AppLauncher has launched Isaac Sim.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

GENERATE = pathlib.Path("/home/ubuntu/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py")
source = GENERATE.read_text()

MARKER = "import isaaclab_mimic.envs  # noqa: F401"
assert MARKER in source, (
    f"expected marker '{MARKER}' not found in {GENERATE} — upstream "
    "generate_dataset.py layout changed, update this wrapper."
)
patched = source.replace(
    MARKER,
    MARKER + "\nimport envs  # noqa: F401  # injected by vla_kitting wrapper",
)

exec(compile(patched, str(GENERATE), "exec"), {"__name__": "__main__", "__file__": str(GENERATE)})
