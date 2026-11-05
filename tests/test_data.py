from __future__ import annotations

from pathlib import Path

import pytest

from credit_pricer.data import (
    DEFAULT_DATA_DIR,
    load_index,
    load_rate_quotes,
    load_single_names,
    load_tranche_quotes,
)


def test_every_file_is_labelled_synthetic() -> None:
    files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
    assert len(files) == 5
    for f in files:
        assert f.read_text(encoding="utf-8").startswith("# SYNTHETIC - illustrative only")


def test_loaders() -> None:
    deposits, swaps = load_rate_quotes()
    assert deposits and swaps
    names = load_single_names()
    assert len(names) == 3
    for q in names.values():
        assert q.tenors == sorted(q.tenors)
        assert all(0 < s < 0.1 for s in q.spreads)
    idx = load_index()
    assert idx.spreads.shape == (125, len(idx.tenors))
    assert idx.coupon == pytest.approx(0.01)
    tr = load_tranche_quotes()
    assert tr.quotes[0].attach == 0.0
    assert len(tr.generating_correlations) == len(tr.quotes)


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rate_quotes(tmp_path / "nope")
