"""Baseline sanity check that project scaffolding is in place."""
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_expected_dirs_exist():
    for d in ["envs", "scripts/validate", "scripts/teleop", "scripts/data",
              "scripts/train", "scripts/assembly", "assets", "datasets",
              "checkpoints", "logs", "reports", "tests", "configs"]:
        assert (REPO / d).is_dir(), f"missing {d}"


def test_justfile_present():
    assert (REPO / "justfile").is_file()


def test_progress_marker_present():
    assert (REPO / "logs/PROGRESS.json").is_file()
