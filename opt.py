from typing import Iterable, Tuple


def opt_cache_simulation(request_trace: Iterable[int], cache_size: int,) -> int:
    """
    Returns optimal number of cache misses for a given request trace and cache size

    request_trace: list of (page) tuples
    """
    cache = set()
    cache_misses = 0

    for i, request in enumerate(request_trace):
        if request not in cache:
            cache_misses += 1
            if len(cache) >= cache_size:
                future_requests = request_trace[i+1:]
                farthest = None
                for cached_item in cache:
                    try:
                        distance = future_requests.index(cached_item)
                    except ValueError:
                        farthest = cached_item
                        break
                    if farthest is None or distance > future_requests.index(farthest):
                        farthest = cached_item
                cache.remove(farthest)
            cache.add(request)

    return cache_misses
