import pandas as pd
import pytest

from bench_workers import add_speedup


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
