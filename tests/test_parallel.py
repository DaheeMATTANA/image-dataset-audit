from parallel_utils import parallel_map


def square(x: int) -> int:
    return x * x


def test_same_results_with_one_or_many_workers():
    items = list(range(100))
    serial = parallel_map(square, items, workers=1)
    parallel = parallel_map(square, items, workers=4)
    assert sorted(serial) == sorted(parallel)
