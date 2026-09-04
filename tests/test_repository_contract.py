from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_references_existing_writing_and_implementation_specifications() -> None:
    manifest = yaml.safe_load((ROOT / "repo_manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["authority"]["implementation_standard"] == ("docs/standards/implementation.md")
    assert manifest["authority"]["implementation_specification"] == "docs/SPECIFICATIONS.md"
    assert (ROOT / manifest["authority"]["implementation_standard"]).is_file()
    assert (ROOT / manifest["authority"]["implementation_specification"]).is_file()
    assert "simulation" in manifest["areas"]
    assert "ml" in manifest["areas"]


def test_repository_context_check_and_selection_are_deterministic() -> None:
    check = subprocess.run(
        [sys.executable, "scripts/repo_context.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    first = subprocess.run(
        [
            sys.executable,
            "scripts/repo_context.py",
            "--path",
            "src/execsim/simulator/core.py",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(first.args, cwd=ROOT, capture_output=True, text=True, check=False)

    assert check.returncode == 0, check.stderr
    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert '"simulation"' in first.stdout
