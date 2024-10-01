import math

from traces.mem_hist import MemHist


class ProbModel:

    # update the model with the most recent history
    # supposed to be called after every change to the history
    def update(self, history: MemHist):
        raise NotImplementedError('')

    # prob that page will appear in the next request
    def prob_next(self, page):
        raise NotImplementedError('')

    # prob that page_a will be requested before page_b
    def prob_before(self, page_a, page_b):
        raise NotImplementedError('')

    # 1-based expected time when page is requested
    def expected_next(self, page):
        raise NotImplementedError('')

    # return a distribution on a pages that can appear next
    def distr_next(self):
        raise NotImplementedError('')

    # return the set of known pages
    def known_pages(self):
        raise NotImplementedError('')


class PredictNext(ProbModel):

    def __init__(self):
        self.last_page = None

    def update(self, history: MemHist):
        self.last_page = next(history.get_history_at(0))

    def distr_next(self):
        if self.last_page is None:
            return {}
        else:
            return {self.last_page + 256: 1.0}


class UniformFixed(ProbModel):
    def __init__(self, page_set):
        self.n = len(page_set)
        self.page_set = page_set
        self.distr = dict.fromkeys(page_set, 1 / self.n)

    def update(self, history: MemHist):
        pass

    def prob_next(self, page):
        return 1 / self.n

    def prob_before(self, page_a, page_b):
        return 0.5

    def expected_next(self, page):
        return self.n

    def distr_next(self):
        return self.distr

    def known_pages(self):
        return self.page_set


class ExplicitDist(ProbModel):
    def __init__(self, initial_prob_dist: dict):
        self.prob_weight = initial_prob_dist
        self.total_weight = sum(self.prob_weight)

    def update(self, history: MemHist):
        pass

    def prob_next(self, page):
        if page in self.prob_weight:
            return self.prob_weight[page] / self.total_weight
        else:
            return 0

    def prob_before(self, page_a, page_b):
        if page_a == page_b:
            return 0.0
        elif page_a in self.prob_weight or page_b in self.prob_weight:
            return self.prob_next(page_a) / (self.prob_next(page_a) + self.prob_next(page_b))
        else:
            # should this be zero?
            return 0.5

    def expected_next(self, page):
        if page in self.prob_weight:
            return self.total_weight / self.prob_next(page)
        else:
            return math.inf

    def distr_next(self):
        return {k: (v / self.total_weight) for k, v in self.prob_weight.items()}

    def known_pages(self):
        return self.prob_weight.keys()


class ProportionalToOccurrences(ExplicitDist):
    def __init__(self):
        super(ProportionalToOccurrences, self).__init__(dict())

    def update(self, history: MemHist):
        last_page = next(history.get_history_at(0))
        if last_page not in self.prob_weight:
            self.prob_weight[last_page] = 1
        else:
            self.prob_weight[last_page] += 1
        self.total_weight += 1


class UniformOnSeen(ExplicitDist):
    def __init__(self):
        super(UniformOnSeen, self).__init__(dict())

    def update(self, history: MemHist):
        last_page = next(history.get_history_at(0))
        if last_page not in self.prob_weight:
            self.prob_weight[last_page] = 1
            self.total_weight += 1


class Mixed(ProbModel):
    def __init__(self, models):
        self.models = models

    def update(self, history: MemHist):
        for _, model in self.models:
            model.update(history)

    def distr_next(self):
        d = {}
        for model_weight, model in self.models:
            for page, prob in model.distr_next().items():
                d[page] = d.get(page, 0.0) + model_weight * prob
        return d
