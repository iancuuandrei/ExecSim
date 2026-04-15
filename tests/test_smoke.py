from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else os.pathsep.join([src_path, existing_pythonpath])
    )
    return env


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "execsim.cli", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_test_env(),
        check=False,
    )

    assert result.returncode == 0
    assert "show-config" in result.stdout
    assert "smoke" in result.stdout
    assert "download-data" in result.stdout
    assert "build-manifest" in result.stdout
    assert "validate-data" in result.stdout


def test_cli_smoke_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "execsim.cli", "smoke"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_test_env(),
        check=False,
    )

    assert result.returncode == 0
    assert "smoke: ok" in result.stdout
