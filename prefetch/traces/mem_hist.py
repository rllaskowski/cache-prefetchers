import bisect
import math
import operator


def update_set(s, ops):
    s = set(s)
    for inserts, removes in ops:
        s -= set(removes)
        s |= set(inserts)
    return s


class MemHist:
    def __init__(self):
        self.history = []
        self.page_lists = dict()
        self.cache_checkpoints = [(0, set(), [])]
        self.cache_operations = 0

    def _init_ops(self):
        self.cache_checkpoints[-1][2].append(([], []))

    def _try_make_cache_checkpoint(self):
        _, cache, ops = self.cache_checkpoints[-1]
        if self.cache_operations >= 4 * (len(cache) + 1):
            cache = update_set(cache, ops)
            self.cache_checkpoints.append((len(self.history), cache, []))
            self.cache_operations = 0
        self._init_ops()

    def next_step(self, page):
        self._try_make_cache_checkpoint()
        t = len(self.history)
        if page not in self.page_lists:
            self.page_lists[page] = [t]
        else:
            self.page_lists[page].append(t)
        self.history.append(page)

    def evict_from_cache(self, page):
        self.cache_operations += 1
        self.cache_checkpoints[-1][2][-1][1].append(page)

    def put_into_cache(self, page):
        self.cache_operations += 1
        self.cache_checkpoints[-1][2][-1][0].append(page)

    def get_cache_at(self, time):
        t = len(self.history) - time
        i = bisect.bisect(self.cache_checkpoints, t, key=operator.itemgetter(0))
        t0, cache, ops = self.cache_checkpoints[i - 1]
        return update_set(cache, ops[0: t - t0])

    def get_history_size(self, time):
        return len(self.history) - time

    def get_history_at(self, time):
        return reversed(self.history[0: self.get_history_size(time)])

    def get_next_request_time_at(self, time, page):
        if page not in self.page_lists:
            return math.inf
        t = len(self.history) - time
        pl = self.page_lists[page]
        i = bisect.bisect(pl, t)
        if i < len(pl):
            return pl[i] - i
        else:
            return math.inf

    def get_pages_seen_so_far(self):
        return self.page_lists.keys()

    def get_page_freq(self, page):
        if page in self.page_lists:
            return len(self.page_lists[page])
        else:
            return 0

    def get_page_last_occurrence(self, page):
        if page in self.page_lists:
            return self.page_lists[page][-1]
        else:
            return None
