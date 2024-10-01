import itertools
import sys

import traces.mem_hist
import typing
import alg

import tqdm

def simulate(cache_size: int, algo: alg.alg.AlgBase, pages_seq: typing.Iterable[int], prefetch_size: int = 0) -> int:
    """
    Simulates algorithm algo on sequence of pages pages_seq and return the number of misses
    :param prefetch_size: maximum number of pages prefetched per request
    :param cache_size: size of the cache in number of pages
    :param algo: an online learning algorithm with prediction and sampled updates
    :param pages_seq: iterable of ints - sequence of page numbers
    :return: the total number of cache misses
    """
    misses = 0
    mem_hist = traces.mem_hist.MemHist()
    iters = 0
    cache = set()
    cache_prev = set()
    bar = tqdm.tqdm(pages_seq)
    for p in bar:
        bar.set_description(f"misses={misses} hit_ratio={round(1.0 - misses / iters if iters > 0 else 1, 2)}")
        if p not in cache:
            misses += 1
            if len(cache) == cache_size:
                cache.remove(algo.predict_evict(cache))
            cache.add(p)
        mem_hist.next_step(p)
        for q in cache_prev.difference(cache):
            mem_hist.evict_from_cache(q)
        for q in cache.difference(cache_prev):
            mem_hist.put_into_cache(q)
        cache_prev = cache.copy()
        algo.update(mem_hist)
        for page in reversed(list(itertools.islice(algo.prefetch_suggest(), prefetch_size))):
            if page not in cache:
                if len(cache) == cache_size:
                    page_to_evict = algo.predict_evict(cache)
                    cache.remove(page_to_evict)
                cache.add(page)
        iters += 1
        if iters % 10 == 0:
            print("iter=", iters, "misses=", misses,
                  "hit_ratio=", round(1.0 - misses / iters, 3),
                  "distinct_pages/iter=", round(len(mem_hist.get_pages_seen_so_far()) / iters, 3))
    return misses
