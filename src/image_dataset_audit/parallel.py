"""Run a function over many items, in one process or in a pool of workers."""

from collections.abc import Callable, Iterable
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

START_METHOD = "forkserver"


def parallel_map[T, R](
    func: Callable[[T], R],
    items: list[T],
    workers: int = 1,
    chunksize: int = 64,
    desc: str = "",
    unit: str = "it",
    threads: bool = False,
) -> list[R]:
    """Apply func to every item; result order is not guaranteed, so sort if needed.

    workers are processes by default, or threads of this process if threads=True.
    func must be defined at the top level of a module so worker processes can import it.
    """
    kind = "thread" if threads else "worker"
    label = f"{desc} ({workers} {kind}{'s' if workers > 1 else ''})"

    if workers == 1:
        return [func(x) for x in tqdm(items, desc=label, unit=unit)]
    pool_class = ThreadPool if threads else get_context(START_METHOD).Pool
    with pool_class(processes=workers) as pool:
        results: Iterable[R] = pool.imap_unordered(func, items, chunksize=chunksize)
        return list(
            tqdm(
                results,
                total=len(items),
                desc=label,
                unit=unit,
            )
        )
