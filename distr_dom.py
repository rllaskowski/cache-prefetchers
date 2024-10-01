from base64 import b16encode
from calendar import c
from copy import deepcopy
from math import e
from collections import defaultdict
import re
from IPython import embed
from matplotlib.pylab import f
from sympy import Q, field, im, li
import torch
from dataclasses import dataclass
import torch.nn as nn
import mmh3
import numpy as np
import random
from eviction import find_dom_distribution
import bisect
import itertools
import copy
import tqdm
import math
import cache
from fieldfm import FieldAwareFactorizationMachineModel
import pulp as pl


def _to_tensor_batched(x: any, batch_dims: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.dim() < batch_dims:
        x = x.unsqueeze(0)

    return x


class FFM(nn.Module):
    def __init__(
        self,
        n_pages,
        cache_size,
        history_size,
        emb_size,
        n_deltas,
        delta_emb_size,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.n_dims = []
        self.emb_size = emb_size
        self.n_deltas = n_deltas
        self.delta_emb_size = delta_emb_size

        if delta_emb_size > 0 and n_deltas > 0:
            field_dims = [n_pages, n_deltas, n_pages, n_deltas]
            field_dims += [n_pages] * history_size + [n_deltas] * history_size
        else:
            field_dims = [n_pages] * (history_size + 2)

        self.ffm = FieldAwareFactorizationMachineModel(
            field_dims=field_dims, embed_dim=emb_size
        )

    def forward(self, a, delta_a, b, delta_b, history, delta_history):
        a = _to_tensor_batched(a, 1).unsqueeze(1)
        b = _to_tensor_batched(b, 1).unsqueeze(1)
        history = _to_tensor_batched(history, 2)
        if self.delta_emb_size > 0 and self.n_deltas > 0:
            delta_a = _to_tensor_batched(delta_a, 1).unsqueeze(1)
            delta_b = _to_tensor_batched(delta_b, 1).unsqueeze(1)
            delta_history = _to_tensor_batched(delta_history, 2)
            ffm_input = torch.cat([a, delta_a, b, delta_b, history, delta_history], dim=1)
        else:
            ffm_input = torch.cat([a, b, history], dim=1)

        return self.ffm(ffm_input)


class Early(nn.Module):
    def __init__(
        self,
        n_pages,
        emb_size,
        delta_emb_size,
        hidden_size,
        n_deltas=100,
        dropout=0.1,
        train_next_delta=False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat_emb_size = emb_size + delta_emb_size
        self.emb_size = emb_size
        self.delta_emb_size = delta_emb_size
        self.train_next_delta = train_next_delta

        self.embedding = nn.Embedding(n_pages, emb_size)
        self.delta_embedding = nn.Embedding(n_deltas, delta_emb_size)

        self.query = nn.Linear(self.concat_emb_size, hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)

        self.rnn = nn.GRU(
            self.concat_emb_size, hidden_size, batch_first=True, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.ff1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ff2 = nn.Linear(hidden_size * 2, n_deltas)

        for layer in [self.query, self.key, self.ff1, self.ff2]:
            torch.nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, a, delta_a, b, delta_b, history, delta_history):
        a = _to_tensor_batched(a, 1)
        b = _to_tensor_batched(b, 1)
        delta_a = _to_tensor_batched(delta_a, 1)
        delta_b = _to_tensor_batched(delta_b, 1)
        history = _to_tensor_batched(history, 2)
        delta_history = _to_tensor_batched(delta_history, 2)

        emb_a = torch.cat(
            [self.embedding(a), self.delta_embedding(delta_a)], dim=1
        ).unsqueeze(1)
        emb_b = torch.cat(
            [self.embedding(b), self.delta_embedding(delta_b)], dim=1
        ).unsqueeze(1)

        history_emb = torch.cat(
            [self.embedding(history), self.delta_embedding(delta_history)], dim=2
        )

        if isinstance(self.rnn, nn.LSTM):
            _, (history_hn, __) = self.rnn(history_emb)
            history_hidden = history_hn.squeeze(0).unsqueeze(1)
        elif isinstance(self.rnn, nn.GRU):
            _, history_hn = self.rnn(history_emb)
            history_hidden = history_hn.squeeze(0).unsqueeze(1)
        else:
            raise ValueError("Invalid RNN type")

        # history_hidden = history_hn
        # history_hidden = history_emb
        query_a_b = self.query(torch.cat([emb_a, emb_b], dim=1))
        query_a_b = self.dropout(query_a_b)
        # query_a_b = torch.cat([emb_a, emb_b], dim=1)
        # query_a_b = self.dropout(history_hidden)

        #key = self.layer_norm(history_hidden)
        key = history_hidden
        scores = query_a_b @ key.transpose(1, 2)
        scores /= torch.sqrt(torch.tensor(self.hidden_size).float())
        out = scores.squeeze(-1)

        out = torch.softmax(out, dim=1)

        if not self.training or not self.train_next_delta:
            return out[:, 0]

        out2 = self.ff1(history_hidden.squeeze(1))
        out2 = self.ff2(torch.relu(out2))
        out2 = torch.softmax(out2, dim=1)

        return out[:, 0], out2


class Mapping:
    def __init__(self, max_size: int, hash_mapping: bool = False):
        self.max_size = max_size
        self.hash_mapping = hash_mapping
        self.mapping: dict[int, int] = {}

    def map(self, key):
        if isinstance(key, list) or isinstance(key, tuple):
            return self._map_list(key)

        if self.hash_mapping:
            return mmh3.hash(str(key)) % self.max_size if self.max_size > 0 else 0

        if key not in self.mapping:
            self.mapping[key] = min(len(self.mapping), self.max_size - 1)

        return self.mapping[key]

    def _map_list(self, keys):
        return [self.map(x) for x in keys]

    def __call__(self, key):
        return self.map(key)

    def __len__(self):
        return self.max_size


def to_delta_sequence(sequence: list[int], addr: int) -> list[int]:
    return [s - addr for s in sequence]


def to_delta_sequence_next(sequence: list[int]) -> int:
    return [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]


def get_first_occurance(occurences: list[int], min_t: int) -> int:
    idx = bisect.bisect_left(occurences, min_t)
    if idx == len(occurences):
        return float("inf")

    occ = occurences[idx]
    next_occ = occurences[idx + 1] if idx + 1 < len(occurences) else float("inf")

    assert min_t <= occ < next_occ, f"min_t: {min_t}, occ: {occ}"

    return occ


def find_mistakes(
    sequence: list[int],
    cache_history: list[tuple],
    occurences: dict[int, list[int]],
    history_size: int,
    samples: int,
    addr_mapping: Mapping,
    delta_mapping: Mapping,
    model: nn.Module,
    interval: int,
    classify_cutoff: float = 0.35,
):

    mistakes = []
    T = len(cache_history)

    model_inputs = []
    ys = []
    for _ in range(samples):

        t = random.randint(
            max(T - 2 - 2*interval + 1, history_size), T - 2 - interval
        )

        cache_t = cache_history[t]

        if len(cache_t) < 2:
            continue

        history_t = sequence[t - history_size + 1 : t + 1]
        history_t_mapped = addr_mapping(history_t)
        delta_history_t = delta_mapping(to_delta_sequence_next(sequence[t - history_size: t + 1]))

        for a, b in itertools.combinations(cache_t, 2):
            a_mapped = addr_mapping(a)
            b_mapped = addr_mapping(b)

            a_first_occ = get_first_occurance(occurences[a_mapped], t + 1)
            b_first_occ = get_first_occurance(occurences[b_mapped], t + 1)

            model_input = {
                "a": a_mapped,
                "b": b_mapped,
                "delta_a": delta_mapping(a - sequence[t]),
                "delta_b": delta_mapping(b - sequence[t]),
                "history": history_t_mapped,
                "delta_history": delta_history_t,
            }


            if a_first_occ < b_first_occ:
                y = 1
            elif a_first_occ == b_first_occ:
                y = 0.5
            else:
                y = 0

            ys.append(y)
            model_inputs.append(model_input)


    inputs = to_batched_dict(model_inputs)

    model.eval()
    with torch.no_grad():
        probs = model(**inputs)


    for y, prob in zip(ys, probs):
        c = classify_cutoff
        if prob < 0.5-c:
            y_model = 0
        elif prob < 0.5 + c:
            y_model = 0.5
        else:
            y_model = 1

        #y_model = int(random.random() < prob)

        #y_model = int(prob > 0.5)

        if y != y_model:
            mistakes.append((True, y, y_model))
        else:
            mistakes.append((False, y, y_model))


    return mistakes



def get_samples(
    sequence: list[int],
    T: int,
    cache_history: list[tuple],
    occurences: dict[int, list],
    cache_size: int,
    history_size: int,
    delta_mapping: int,
    addr_mapping: int,
    interval=80,
    n_samples=1000,
    cache_replacement=5,
):

    samples = []

    while len(samples) < n_samples:
        hist_size = T
        #t = int(hist_size * (1.0 - math.sqrt(random.random())))
        #t = min(t, T-2)
        #t = max(t, history_size)
        t = random.randint(
            max(T - 2 - interval * 2 + 1, history_size), T - 2 - interval
        )

        cache_t = cache_history[t]

        if cache_replacement > 0:
            to_replace_idxs = random.sample(range(len(cache_t)), cache_replacement)
            to_replace = [cache_t[i] for i in to_replace_idxs]
            cache_t = [x for x in cache_t if x not in to_replace]
            new_options = tuple(set(range(len(addr_mapping))) - set(cache_t))
            new_pages = random.sample(new_options, cache_replacement)
            cache_t += new_pages

        cache_t = random.sample(cache_t, len(cache_t))

        if len(cache_t) < 2:
            continue

        last_addr = sequence[t]
        history_t = sequence[t + 1 - history_size: t + 1]

        history_t_mapped = addr_mapping(history_t)
        delta_history_t = to_delta_sequence_next(sequence[t - history_size: t + 1])
        delta_history_t_mapped = delta_mapping(delta_history_t)

        cache_t_mapped = addr_mapping(cache_t)
        delta_cache_t = to_delta_sequence(cache_t, last_addr)
        delta_cache_t_mapped = delta_mapping(delta_cache_t)


        for a_i, b_i in itertools.combinations(range(len(cache_t)), 2):
            a_mapped = cache_t_mapped[a_i]
            b_mapped = cache_t_mapped[b_i]

            a_delta_mapped = delta_cache_t_mapped[a_i]
            b_delta_mapped = delta_cache_t_mapped[b_i]

            a_first_occ = get_first_occurance(occurences[a_mapped], t + 1)
            b_first_occ = get_first_occurance(occurences[b_mapped], t + 1)

            if a_first_occ < b_first_occ:
                y = 1
            elif a_first_occ == b_first_occ:
                y = 0.5
            else:
                y = 0

            samples.append(
                {
                    "a": a_mapped,
                    "b": b_mapped,
                    "delta_a": a_delta_mapped,
                    "delta_b": b_delta_mapped,
                    "history": history_t_mapped,
                    "delta_history": delta_history_t_mapped,
                    "next_delta": delta_mapping(sequence[t + 1] - last_addr),
                    "prob": y,
                }
            )
            samples.append(
                {
                    "a": b_mapped,
                    "b": a_mapped,
                    "delta_a": b_delta_mapped,
                    "delta_b": a_delta_mapped,
                    "history": history_t_mapped,
                    "delta_history": delta_history_t_mapped,
                    "next_delta": delta_mapping(sequence[t + 1] - last_addr),
                    "prob": 1 - y,
                }
        )

    return samples


def to_batched_dict(list_dict):
    def _map(k):
        if k == "prob":
            dtype = torch.float
        else:
            dtype = torch.long

        return torch.tensor([d[k] for d in list_dict], dtype=dtype)

    keys = list(list_dict[0].keys())
    return {k: _map(k) for k in keys}


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list[tuple],
    n_deltas: int,
    batch_size=32,
    train_next_delta=False,
):

    loss_a_b_sum = 0
    loss_delta_sum = 0
    random_loss_sum = 0
    criterion_a_b = nn.BCEWithLogitsLoss()
    criterion_delta = nn.CrossEntropyLoss()

    model.train()
    for i in range(0, len(samples), batch_size):
        batch = to_batched_dict(samples[i : i + batch_size])
        prob_target = batch.pop("prob")
        next_delta = batch.pop("next_delta")


        model_out = model(**batch)
        random_model_out = torch.ones_like(prob_target) * 0.5

        if train_next_delta:
            probs_a_b, distr_delta = model_out
        else:
            probs_a_b = model_out

        loss_a_b = criterion_a_b(probs_a_b, prob_target)
        loss_a_b_sum += loss_a_b.item()
        random_loss_sum += criterion_a_b(random_model_out, prob_target).item()
        loss = loss_a_b

        if train_next_delta:
            next_delta_target = torch.zeros((batch_size, n_deltas))
            next_delta_target[range(len(next_delta)), next_delta] = 1
            loss_delta = criterion_delta(distr_delta, next_delta)

            loss += 0.5 * loss_delta
            loss_delta_sum += loss_delta.item()


        loss.backward()
        optimizer.step()

    batches = math.ceil(len(samples) / batch_size)
    loss_a_b_sum /= batches
    loss_delta_sum /= batches
    random_loss_sum /= batches

    return loss_a_b_sum, loss_delta_sum, random_loss_sum


def dominating_distribution(solver, pages, w):
    for page_i in pages:
        for page_j in pages:
            if page_i == page_j:
                continue
            assert math.isclose(w[(page_i, page_j)] + w[(page_j, page_i)], 1.0)


    model = pl.LpProblem(sense=pl.LpMinimize)
    x = []
    for i in range(len(pages)):
        x.append(pl.LpVariable(name='x' + str(i), lowBound=0))
    c = pl.LpVariable(name='c', lowBound=0)
    for i, page_i in enumerate(pages):
        s = pl.LpAffineExpression()
        for j, page_j in enumerate(pages):
            if page_i != page_j and w[(page_j, page_i)] != 0.0:
                s += w[(page_j, page_i)] * x[i]
        model += s <= c
    s = pl.LpAffineExpression()
    for i, page in enumerate(pages):
        s += x[i]
    model += s == 1
    model += c
    model.solve(solver=solver)

    distr =  np.array([x[i].value() for i in range(len(pages))], dtype=np.float64)
    return (distr / distr.sum()).tolist()


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def test_on_sequence(
    sequence_rw: list[tuple[str, int]],
    cache_size,
    n_pages,
    n_deltas,
    train_interval=50,
    history_size=1,
    greedy=True,
    emb_size=10,
    hidden_size=64,
    lr=1e-4,
    wd=0.1,
    dropout=0.1,
    train_samples=800,
    delta_emb_size=10,
    cache_replacement=5,
    batch_size=32,
    train_next_delta=False,
    hash_mapping=False,
    model_class="ffm",
    from_model=None,
    random_evictor_seed=42,
    count_mistakes=False,
    classify_cutoff=0.35,
):
    solver = pl.getSolver('GLPK_CMD', msg=False)
    delta_mapping = Mapping(n_deltas, hash_mapping)
    addr_mapping = Mapping(n_pages, hash_mapping)
    model_config = dict(
        n_pages=n_pages,
        cache_size=cache_size,
        history_size=history_size,
        emb_size=emb_size,
        delta_emb_size=delta_emb_size,
        hidden_size=hidden_size,
        n_deltas=n_deltas,
        dropout=dropout,
        train_next_delta=train_next_delta,
    )

    if from_model is not None:
        model = from_model
    else:
        if model_class == "ffm":
            model = FFM(**model_config)
        else:
            model = Early(**model_config)

    if isinstance(model, Early):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)

    misses = 0
    misses_history = []
    losses_a_b = []
    random_losses = []
    losses_delta = []
    cache_history = []
    occurences = defaultdict(list)
    cache = set()
    sequence = [x[1] for x in sequence_rw if x[0] == "R"]
    pbar = tqdm.tqdm(list(enumerate(sequence_rw)))
    evict_counts = defaultdict(int)
    mistakes_0 = []
    mistakes_05 = []
    mistakes_1 = []

    random_cache = set()
    random_cache_misses = 0

    random_generator = random.Random(random_evictor_seed)
    entropy_sum = 0
    entropy_count = 0
    variance_sum = 0

    def get_to_evict(t) -> int:
        if t <= history_size:
            return random.choice(list(cache)), 0, 0

        model_input = []

        cache_tuple = cache_history[t]
        page_pairs = [(a,b) for a, b in itertools.permutations(cache_tuple, 2) if a != b]

        history = sequence[t - history_size + 1 : t + 1]
        history_mapped = addr_mapping(history)
        delta_history = to_delta_sequence_next(sequence[t - history_size: t + 1])
        delta_history_mapped = delta_mapping(delta_history)
        last_addr = sequence[t]

        for a, b in page_pairs:
            if a == b:
                continue

            a_mapped = addr_mapping(a)
            b_mapped = addr_mapping(b)
            a_delta_mapped = delta_mapping(a - last_addr)
            b_delta_mapped = delta_mapping(b - last_addr)

            model_input.append(
                {
                    "a": a_mapped,
                    "b": b_mapped,
                    "delta_a": a_delta_mapped,
                    "delta_b": b_delta_mapped,
                    "history": history_mapped,
                    "delta_history": delta_history_mapped,
                }
            )

        model.eval()
        with torch.no_grad():
            probs = model(**to_batched_dict(model_input))
            probs = probs.tolist()

        pair_probabilities = {}
        for (a, b), prob in zip(page_pairs, probs):
            pair_probabilities[(a, b)] = prob


        in_edges = defaultdict(int)
        probs_mean = defaultdict(float)
        for (a, b) in page_pairs:
            if a > b:
                continue

            probs_mean[(a, b)] = (pair_probabilities[(a, b)] + 1 - pair_probabilities[(b, a)]) / 2
            probs_mean[(b, a)] = 1 - probs_mean[(a, b)]
            c = classify_cutoff
            in_edges[a] += pair_probabilities[(a, b)] > 0.5 + c
            in_edges[b] += pair_probabilities[(b, a)] > 0.5 + c
            if 0.5-c < in_edges[a] < 0.5+c:
                in_edges[a] += 1
                in_edges[b] += 1

        distr = dominating_distribution(solver, cache_tuple, probs_mean)
        entropy = -sum(p * math.log(p) for p in distr if p > 0)
        variance = sum(p * (1 - p) for p in distr)

        if greedy:
            #to_evict_idx = max(range(len(cache_tuple)), key=lambda i: in_edges[cache_tuple[i]])
            #to_evict_idx = np.argmax(distr)
            x = [in_edges[cache_tuple[i]] for i in range(len(cache_tuple))]
            to_evict_idx = random.choices(range(len(cache_tuple)), k=1, weights=softmax(x))[0]
        else:
            to_evict_idx = np.random.choice(len(cache_tuple), p=distr)

        return cache_tuple[to_evict_idx], entropy, variance

    for t, addr in pbar:
        addr_mapped = addr_mapping.map(addr)
        occurences[addr_mapped].append(t)
        cache_history.append(tuple(cache))

        if len(cache) < cache_size:
            cache.add(addr)

        elif addr not in cache:
            to_evict, distr_entropy, distr_variance = get_to_evict(t)
            entropy_sum += distr_entropy
            variance_sum += distr_variance
            entropy_count += 1
            evict_counts[addr_mapping(to_evict)] += 1
            cache.remove(to_evict)
            cache.add(addr)
            misses += 1

        if len(random_cache) < cache_size:
            random_cache.add(addr)

        elif addr not in random_cache:
            to_evict = random_generator.choice(list(random_cache))
            random_cache_misses += 1
            random_cache.remove(to_evict)
            random_cache.add(addr)

        misses_history.append(misses)

        if count_mistakes and t % train_interval == 0 and t >= train_interval * 3 and t > history_size * 2:
            _mistakes = find_mistakes(
                sequence,
                cache_history,
                occurences,
                history_size,
                samples=32,
                addr_mapping=addr_mapping,
                delta_mapping=delta_mapping,
                model=model,
                interval=train_interval,
                classify_cutoff=classify_cutoff,
            )
            for m in _mistakes:
                if m[1] == 0:
                    mistakes_0.append(m[0])
                elif m[1] == 0.5:
                    mistakes_05.append(m[0])
                else:
                    mistakes_1.append(m[0])
        if t % train_interval == 0 and t >= train_interval * 2 and t > history_size * 2:
            samples = get_samples(
                sequence,
                t,
                cache_history,
                occurences,
                cache_size,
                history_size,
                delta_mapping,
                addr_mapping,
                n_samples=train_samples,
                interval=train_interval,
                cache_replacement=cache_replacement,
            )
            loss_a_b, loss_delta, random_loss = train_model(
                model,
                optimizer,
                samples,
                batch_size=batch_size,
                n_deltas=n_deltas,
                train_next_delta=train_next_delta,
            )
            losses_a_b.append(loss_a_b)
            losses_delta.append(loss_delta)
            random_losses.append(random_loss)

        ratio = misses / random_cache_misses if random_cache_misses != 0 else 1
        entropy_mean = entropy_sum / entropy_count if entropy_count != 0 else 0
        most_common = max(evict_counts, key=evict_counts.get) if evict_counts else None
        most_common_counts = evict_counts[most_common] if most_common else 0
        all_counts = sum(evict_counts.values()) if evict_counts else 1
        hit_ratio = 1 - misses / (t + 1)
        variance_mean = variance_sum / entropy_count if entropy_count != 0 else 0


        all_mistakes = sum(mistakes_0) + sum(mistakes_05) + sum(mistakes_1)
        all_mistaktes_mean = all_mistakes / (len(mistakes_0) + len(mistakes_05) + len(mistakes_1) or 1)
        mistakes_0_mean = sum(mistakes_0) / (len(mistakes_0) or 1)
        mistakes_05_mean = sum(mistakes_05) / (len(mistakes_05) or 1)
        mistakes_1_mean = sum(mistakes_1) / (len(mistakes_1) or 1)


        mistakes_0 = mistakes_0[-20:]
        mistakes_05 = mistakes_05[-20:]
        mistakes_0 = mistakes_1[-20:]

        pbar.set_description(
            f"Misses: {misses}, "
            f"Random misses: {random_cache_misses}, "
            f"misses/random: {ratio:.3f}, "
            f"Entropy: {entropy_mean:.3f}, "
            f"All Mistakes: {all_mistaktes_mean:.3f}, "
            f"Mistakes 0: {mistakes_0_mean:.3f}, "
            f"Mistakes 0.5: {mistakes_05_mean:.3f}, "
            f"Mistakes 1: {mistakes_1_mean:.3f}, "
            f"Hits: {hit_ratio:.3f}, "
            f"Loss p(a,b): {losses_a_b[-1] if losses_a_b else 0}, "
            f"Loss delta: {losses_delta[-1] if losses_delta else 0}, "
            f"Random loss: {random_losses[-1] if random_losses else 0}"
        )

    return misses, misses_history, (losses_a_b, losses_delta), evict_counts, model
