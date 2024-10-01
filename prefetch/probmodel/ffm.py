import math
import numpy as np
import mmh3
import itertools
from probmodel.probmodel import ProbModel
from traces.mem_hist import MemHist


def phi_ffm(w, x):
    assert(isinstance(w, np.ndarray) and w.ndim == 3)
    res = 0.0
    for j1 in range(0, len(x) - 1):
        i1 = x[j1]
        for j2 in range(j1 + 1, len(x)):  # todo: vectorize this loop
            i2 = x[j2]
            res += np.dot(w[i1, j2], w[i2, j1])
    return res


def get_x(mangled_history, *args):
    h = list(mangled_history)
    h.extend(args)
    return np.array(h, dtype=np.int32)


class FFMBase(ProbModel):
    def __init__(self, n, f, k, lambda_, eta, history_size, samples_per_step):
        self.n = n
        self.f = f
        self.k = k
        self.lambda_ = lambda_
        self.eta = eta
        self.history_size = history_size
        self.samples_per_step = samples_per_step
        self.most_recent_mangled_history = []
        self.rng = np.random.default_rng()

    def _hash_page(self, p):
        return mmh3.hash(p.to_bytes(8, byteorder='big'), signed=False) % self.n

    def _mangle_history(self, history_slice):
        return map(self._hash_page, itertools.islice(history_slice, self.history_size))

    def update(self, history: MemHist):
        self.most_recent_mangled_history = self._mangle_history(history.get_history_at(0))
        hist_size = history.get_history_size(0)
        for i in range(self.samples_per_step):
            t = int(hist_size * (1.0 - math.sqrt(self.rng.random())))
            t = min(t, hist_size - 1)
            self.update_t(history, t)

    def update_t(self, history: MemHist, t):
        raise NotImplementedError('')


class FFMSingle(FFMBase):
    def __init__(self, n, f, k, lambda_, eta, history_size, samples_per_step):
        super().__init__(n, f, k, lambda_, eta, history_size, samples_per_step)
        self.w = self.rng.random(size=(n, f, k), dtype=np.float32) / math.sqrt(k)
        self.G = np.ones(shape=self.w.shape, dtype=self.w.dtype)

    def _prob(self, x):
        phi = phi_ffm(self.w, x)
        if phi < 0:
            e = math.exp(phi)
            return e / (e + 1.0)
        else:
            return 1.0 / (1.0 + math.exp(-phi))

    def prob(self, mangled_history, *args):
        return self._prob(get_x(mangled_history, *args))

    def _update(self, y, x):
        kappa = y - self._prob(x)
        j_lists = [[] for _ in range(self.n)]
        for j in range(len(x)):
            j_lists[x[j]].append(j)
        i_list = [i for i in range(self.n) if j_lists[i]]
        s = np.zeros(shape=(len(i_list), self.k), dtype=self.w.dtype)
        for si in range(len(i_list)):
            i = i_list[si]
            for j in j_lists[i]:
                s[si] += self.w[i, j]
        for i in i_list:
            for si in range(len(i_list)):
                ii = i_list[si]
                if i == ii and len(j_lists[ii]) == 1:
                    continue
                for j in j_lists[ii]:
                    # ii = x[j]
                    d_phi = s[si]
                    if i == ii:
                        d_phi -= self.w[ii, j]
                    g = self.lambda_ * self.w[i, j] + kappa * d_phi
                    self.G[i, j] += np.square(g)
                    self.w[i, j] -= self.eta * (g / np.sqrt(self.G[i, j]))

    def update_mangled(self, mangled_history, mangled_cache_times):
        m = len(mangled_cache_times)
        for i in range(m - 1):
            p, t_p = mangled_cache_times[i]
            for j in range(i + 1, m):
                q, t_q = mangled_cache_times[j]
                if t_p < t_q:
                    y = 0.0
                elif t_p == t_q:
                    y = 0.5
                else:
                    y = 1.0
                self._update(y, get_x(mangled_history, p, q))
                self._update(1.0 - y, get_x(mangled_history, q, p))

    def _mangle_update(self, history, cache_times):
        mangled_history = self._mangle_history(history)
        orig_cache_page = dict()
        page_time = dict()
        for p, t in cache_times:
            mangled_p = self._hash_page(p)
            if mangled_p not in orig_cache_page or t < page_time[mangled_p]:
                orig_cache_page[mangled_p] = p
                page_time[mangled_p] = t
        self.update_mangled(mangled_history, list(page_time.items()))

    def update_t(self, history: MemHist, t):
        cache = history.get_cache_at(t)
        cache_times = [(p, history.get_next_request_time_at(t, p)) for p in cache]
        self._mangle_update(history.get_history_at(t), cache_times)

    def prob_before(self, page_a, page_b):
        return self.prob(self.most_recent_mangled_history, self._hash_page(page_a), self._hash_page(page_b))


class FFMDistribution(FFMBase):
    def __init__(self, n, f, k, lambda_, eta, history_size, samples_per_step):
        super().__init__(n, f, k, lambda_, eta, history_size, samples_per_step)
        self.pages_history = [None] * n
        self.mangled_prediction = []
        self.wa = self.rng.random(size=(n, k), dtype=np.float32) / math.sqrt(k)
        self.wb = self.rng.random(size=(n + 1, f, k), dtype=np.float32) / math.sqrt(k)
        self.Ga = np.ones(shape=self.wa.shape, dtype=self.wa.dtype)
        self.Gb = np.ones(shape=self.wb.shape, dtype=self.wb.dtype)

    def _softmax(self, x):
        z = np.tensordot(self.wa[x], self.wb, axes=([0, 1], [1, 2]))
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def distr_next(self):
        x = get_x(self.most_recent_mangled_history)
        if len(x) < self.history_size:
            return {}
        p = self._softmax(x)
        return {self.pages_history[i]: p[i] for i in range(self.n) if self.pages_history[i]}

    def _update(self, y, x):
        kappa = self._softmax(x)
        kappa[y] -= 1.0
        j_lists = [[] for _ in range(self.n)]
        for j in range(len(x)):
            j_lists[x[j]].append(j)
        i_list = [i for i in range(self.n) if j_lists[i]]
        for si in range(len(i_list)):
            i = i_list[si]
            s = np.zeros(shape=(self.n + 1, self.k), dtype=self.wb.dtype)
            for j in j_lists[i]:
                s += self.wb[:, j]
            ga = self.lambda_ * self.wa[i] + np.dot(kappa, s)
            self.Ga[i] += np.square(ga)
            self.wa[i] -= self.eta * (ga / np.sqrt(self.Ga[i]))
        for j in range(self.f):
            gb = self.lambda_ * self.wb[:, j] + np.outer(kappa, self.wa[x[j]])
            self.Gb[:, j] += np.square(gb)
            self.wb[:, j] -= self.eta * (gb / np.sqrt(self.Gb[:, j]))

    def update_t(self, history: MemHist, t):
        if t == 0 or history.get_history_size(t) < self.history_size:
            return
        self._update(self.mangled_prediction[-t],
                     get_x(self._mangle_history(history.get_history_at(t))))

    def update(self, history: MemHist):
        p = next(history.get_history_at(0))
        h = self._hash_page(p)
        self.mangled_prediction.append(h if self.pages_history[h] == p else self.n)
        self.pages_history[h] = p
        super().update(history)
