import random

import numpy as np
import pulp as pl
from traces.mem_hist import MemHist
from probmodel.probmodel import ProbModel


class AlgBase:

    # supposed to be passed history *after* the new request
    def update(self, history: MemHist):
        raise NotImplementedError('')

    # assumes the history from last update
    def predict_evict(self, cache):
        raise NotImplementedError('')

    # return an iterable of suggested pages to prefetch, sorted by priority
    def prefetch_suggest(self):
        return []


def dominating_distribution(solver, n, w):
    model = pl.LpProblem(sense=pl.LpMinimize)
    x = []
    for i in range(0, n):
        x.append(pl.LpVariable(name='x' + str(i), lowBound=0))
    c = pl.LpVariable(name='c', lowBound=0)
    for i in range(0, n):
        s = pl.LpAffineExpression()
        for j in range(0, n):
            if i != j and w[j, i] != 0.0:
                s += w[j, i] * x[i]
        model += s <= c
    s = pl.LpAffineExpression()
    for i in range(0, n):
        s += x[i]
    model += s == 1
    model += c
    model.solve(solver=solver)
    return np.array([x[i].value() for i in range(0, n)], dtype=w.dtype)


def prefetch_most_probable(model: ProbModel, prob_threshold):
    distr = model.distr_next()
    if prob_threshold > 0.0:
        distr = {k: v for k, v in distr.items() if v >= prob_threshold}
    return sorted(distr, key=distr.get, reverse=True)


class DomDistAlg(AlgBase):
    def __init__(self, model: ProbModel, prefetch_model: ProbModel = None, prob_threshold=0.0):
        self.model = model
        self.prefetch_model = prefetch_model
        self.prob_threshold = prob_threshold
        #self.solver = pl.getSolver('SCIP_CMD', msg=False)
        self.solver = pl.getSolver('GLPK_CMD', msg=False)
    def update(self, history: MemHist):
        self.model.update(history)
        if self.prefetch_model is not None:
            self.prefetch_model.update(history)

    def predict_evict(self, cache):
        m = len(cache)
        cache = list(cache)
        w = np.zeros((m, m), dtype=np.float32)
        for i in range(m - 1):
            w[i, i] = 0.0
            for j in range(i + 1, m):
                x = self.model.prob_before(cache[i], cache[j])
                y = self.model.prob_before(cache[j], cache[i])
                x = (x + 1.0 - y) / 2.0
                w[i, j] = x
                w[j, i] = 1.0 - x
        dist = dominating_distribution(self.solver, m, w)
        return random.choices(cache, weights=dist)[0]

    def prefetch_suggest(self):
        if self.prefetch_model is None:
            return super().prefetch_suggest()
        else:
            return prefetch_most_probable(self.prefetch_model, self.prob_threshold)


class RandomAlg(AlgBase):

    def __init__(self):
        pass

    def update(self, history: MemHist):
        ...

    def predict_evict(self, cache):
        return random.choice(list(cache))

    def prefetch_suggest(self):
        return []


class LRUAlg(AlgBase):
    def __init__(self):
        self.recent_history = None

    def update(self, history: MemHist):
        self.recent_history = history

    def predict_evict(self, cache):
        remaining = set(cache)
        for page in self.recent_history.get_history_at(0):
            remaining.discard(page)
            if len(remaining) <= 1:
                break
        return min(remaining)


class MQAlg(LRUAlg):
    def __get_page_key(self, page):
        i = 0
        f = self.recent_history.get_page_freq(page)
        t = self.recent_history.get_page_last_occurrence(page)
        if t is None:
            t = -1
        while f > 0:
            f >>= 1
            i += 1
        s = self.recent_history.get_history_size(0)
        life_time = max(6, s / (3 * 1024))
        age = s - t
        i = max(0, i - age // life_time)
        return i, t

    def predict_evict(self, cache):
        return min(cache, key=lambda p: self.__get_page_key(p))

# evicts the one with latest expected next request
# prefetches according to the probability of appearing next
class METAlg(AlgBase):
    def __init__(self, model: ProbModel):
        self.model = model

    def update(self, history: MemHist):
        self.model.update(history)

    def predict_evict(self, cache):
        return max(cache, key=self.model.expected_next)

    def prefetch_suggest(self):
        distr = self.model.distr_next()
        return sorted(distr, key=distr.get, reverse=True)


class LRUPrefetchMostProbable(LRUAlg):
    def __init__(self, model: ProbModel, prob_threshold=0.0):
        super(LRUPrefetchMostProbable, self).__init__()
        self.model = model
        self.prob_threshold = prob_threshold

    def update(self, history: MemHist):
        super(LRUPrefetchMostProbable, self).update(history)
        self.model.update(history)

    def prefetch_suggest(self):
        return prefetch_most_probable(self.model, self.prob_threshold)


class MQPrefetchMostProbable(MQAlg):
    def __init__(self, model: ProbModel, prob_threshold=0.0):
        super(MQPrefetchMostProbable, self).__init__()
        self.model = model
        self.prob_threshold = prob_threshold

    def update(self, history: MemHist):
        super(MQPrefetchMostProbable, self).update(history)
        self.model.update(history)

    def prefetch_suggest(self):
        return prefetch_most_probable(self.model, self.prob_threshold)


class DomPrefetchMostProbable(DomDistAlg):
    def __init__(self, model: ProbModel):
        super(DomPrefetchMostProbable, self).__init__(model)

    def prefetch_suggest(self):
        distr = self.model.distr_next()
        return sorted(distr, key=distr.get, reverse=True)
