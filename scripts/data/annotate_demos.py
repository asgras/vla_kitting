"""Wrapper around IsaacLab's annotate_demos.py that injects our `envs` package
import AFTER the Isaac Sim app launches (importing envs before then pulls in
isaaclab.assets which requires omni.physics — only available once the app is
running).

Usage:
    ./isaaclab.sh -p scripts/data/annotate_demos.py \\
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \\
        --input_file datasets/teleop/cube_scripted.hdf5 \\
        --output_file datasets/teleop/cube_annotated.hdf5 \\
        --auto --headless --enable_cameras
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ANNOTATE = pathlib.Path("/home/ubuntu/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py")
source = ANNOTATE.read_text()

# Inject our gym registration after isaaclab_mimic.envs is imported.
# The reference line reliably appears in the file; if Isaac Lab refactors it,
# the fail-loud assertion below will catch it.
MARKER = "import isaaclab_mimic.envs  # noqa: F401"
assert MARKER in source, (
    f"expected marker '{MARKER}' not found in {ANNOTATE} — upstream "
    "annotate_demos.py layout changed, update this wrapper."
)
patched = source.replace(
    MARKER,
    MARKER + "\nimport envs  # noqa: F401  # injected by vla_kitting wrapper",
)

# Execute as __main__ so argparse and the if __name__ block run normally.
exec(compile(patched, str(ANNOTATE), "exec"), {"__name__": "__main__", "__file__": str(ANNOTATE)})
