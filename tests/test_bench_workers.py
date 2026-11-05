import os

import pandas as pd
import pytest

from bench_workers import add_speedup, malloc_settings


def test_malloc_settings_default_when_none_set(monkeypatch):
    for k in [k for k in os.environ if k.startswith("MALLOC_")]:
        monkeypatch.delenv(k)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    assert malloc_settings() == "default"


def test_malloc_settings_lists_variables_sorted(monkeypatch):
    monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", "134217728")
    monkeypatch.setenv("MALLOC_MMAP_THRESHOLD_", "67108864")
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    assert malloc_settings() == (
        "MALLOC_MMAP_THRESHOLD_=67108864;MALLOC_TRIM_THRESHOLD_=134217728"
    )


def test_speedup_uses_the_one_worker_run_of_the_same_sample():
    # Rows deliberately not in workers order, and two sample sizes.
    table = pd.DataFrame(
        {
            "sample": [100, 100, 200, 200],
            "workers": [4, 1, 1, 4],
            "seconds": [1.0, 2.0, 8.0, 4.0],
        }
    )
    out = add_speedup(table)
    assert out["speedup"].tolist() == pytest.approx([2.0, 1.0, 1.0, 2.0])
    assert out["efficiency"].tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])
