"""Run a function over many items, in one process or in a pool of workers."""

from collections.abc import Callable, Iterable
from multiprocessing import get_context

from tqdm import tqdm


def parallel_map[T, R](
    func: Callable[[T], R],
    items: list[T],
    workers: int = 1,
    chunksize: int = 64,
    desc: str = "",
    unit: str = "it",
) -> list[R]:
    """Apply func to every item; result order is not guaranteed, so sort if needed.

    func must be defined at the top level of a module so worker processes can import it.
    """
    if workers == 1:
        return [func(x) for x in tqdm(items, desc=desc, unit=unit)]
    with get_context("forkserver").Pool(processes=workers) as pool:
        results: Iterable[R] = pool.imap_unordered(func, items, chunksize=chunksize)
        return list(
            tqdm(
                results,
                total=len(items),
                desc=f"{desc} ({workers} workers)",
                unit=unit,
            )
        )
