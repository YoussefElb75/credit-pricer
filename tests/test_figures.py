"""The figure script runs end to end (images are not compared)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("QuantLib")

ROOT = Path(__file__).resolve().parents[1]


def test_make_figures_runs(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "make_figures.py"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert len(pngs) == 18  # nine figures, light and dark
    for p in tmp_path.glob("*.png"):
        assert p.stat().st_size > 10_000
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
