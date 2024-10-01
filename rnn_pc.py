from base64 import b16encode
from calendar import c
from copy import deepcopy
from math import e
from collections import defaultdict
import re
from IPython import embed
from matplotlib.pylab import f
from pyparsing import C
from sympy import Q, field, im, li
import torch
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
import random
import bisect
import itertools
import tqdm
import math
import pulp as pl
from mapping import Mapping


def _to_tensor_batched(x: any, batch_dims: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.dim() < batch_dims:
        x = x.unsqueeze(0)

    return x


class AddrHistory:
    def __init__(self, history_size: int, addr_mapping: Mapping) -> None:
        self.addrs = []
        self.pcs = []
        self.history_size = history_size
        self.addr_mapping = addr_mapping
        self.occurences = defaultdict(list)

    def update(self, addr: int, pc: int):
        self.addrs.append(addr)
        self.pcs.append(pc)
        self.occurences[addr].append(len(self.addrs) - 1)

    def get_addr_history(self, t, length=None):
        length = length or self.history_size
        return self.addrs[t + 1 - length : t + 1]

    def get_pc_history(self, t, length=None):
        length = length or self.history_size
        return self.pcs[t + 1 - self.history_size : t + 1]

    def __getitem__(self, t):
        return self.get_addr_history(t), self.get_pc_history(t)

    def sample(self, min_t=0, max_t=None):
        min_t = max(min_t, self.history_size)
        max_t = max_t or len(self.addrs)
        max_t = min(max_t, len(self.addrs))

        t = random.randint(min_t, max_t - 1)

        return self[t]

    def get_deltas(self, t, length=None):
        length = length or (self.history_size + 1)
        return to_delta_sequence_next(self.get_addr_history(t, length))

    def get_first_occurance(self, a: int, min_t: int) -> int:
        occurences = self.occurences[a]
        idx = bisect.bisect_left(occurences, min_t)
        if idx == len(occurences):
            return float("inf")

        occ = occurences[idx]
        next_occ = occurences[idx + 1] if idx + 1 < len(occurences) else float("inf")

        assert min_t <= occ < next_occ, f"min_t: {min_t}, occ: {occ}"

        return occ


class CacheHistory:
    def __init__(self, max_size: int) -> None:
        self.history = []
        self.current = []
        self.max_size = max_size

    def add(self, addr: int, pc: int):
        if addr in self:
            self.current = [x for x in self.current if x[0] != addr]

        self.current.append((addr, pc))
        assert (
            len(self.current) <= self.max_size
        ), f"Cache size exceeded: {len(self.current)}"

    def evict(self, addr: int):
        assert addr in self, f"Address {addr} not in cache"
        previous_len = len(self.current)
        self.current = [x for x in self.current if x[0] != addr]
        assert len(self.current) == previous_len - 1, f"Eviction failed"

    def get_addr_history(self, t):
        return tuple([x[0] for x in self.history[t]])

    def get_pc_history(self, t):
        return tuple([x[1] for x in self.history[t]])

    def get_pc(self, addr):
        for a, pc in self.current:
            if a == addr:
                return pc

        raise ValueError(f"Address {addr} not in cache")

    def save(self):
        self.history.append(tuple(self.current))

    def __contains__(self, addr: int):
        return len([x for x in self.current if x[0] == addr]) > 0

    def __getitem__(self, t):
        return self.history[t]

    def __len__(self):
        return len(self.current)


def get_model_input(
    a: int,
    b: int,
    pc_a: int,
    pc_b: int,
    t: int,
    history: AddrHistory,
    delta_mapping: Mapping,
    addr_mapping: Mapping,
    pc_mapping: Mapping,
):
    history_t = history.get_addr_history(t)
    pc_history_t = history.get_pc_history(t)
    delta_history_t = history.get_deltas(t)

    return {
        "a": addr_mapping(a),
        "b": addr_mapping(b),
        "delta_a": delta_mapping(a - history_t[-1]),
        "delta_b": delta_mapping(b - history_t[-1]),
        "history": addr_mapping(history_t),
        "delta_history": delta_mapping(delta_history_t),
        "pc_history": pc_mapping(pc_history_t),
        "pc_a": pc_mapping(pc_a),
        "pc_b": pc_mapping(pc_b),
    }


def get_target_output(a: int, b: int, t: int, history: AddrHistory) -> float:
    a_first_occ = history.get_first_occurance(a, t + 1)
    b_first_occ = history.get_first_occurance(b, t + 1)

    if a_first_occ < b_first_occ:
        y = 1
    elif a_first_occ == b_first_occ:
        y = 0.5
    else:
        y = 0

    return y


class PCRnn(nn.Module):
    def __init__(
        self,
        emb_size,
        delta_emb_size,
        pc_emb_size,
        hidden_size,
        addr_mapping: Mapping,
        delta_mapping: Mapping,
        pc_mapping: Mapping,
        dropout=0.1,
        train_next_delta=False,
        train_next_pc=False,
        rnn_layers=2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat_emb_size = emb_size + delta_emb_size + pc_emb_size
        self.emb_size = emb_size
        self.delta_emb_size = delta_emb_size
        self.pc_emb_size = pc_emb_size
        self.train_next_delta = train_next_delta
        self.train_next_pc = train_next_pc
        self.addr_mapping = addr_mapping
        self.delta_mapping = delta_mapping
        self.pc_mapping = pc_mapping

        self.n_pages = addr_mapping.max_size
        self.n_deltas = delta_mapping.max_size
        self.n_pcs = pc_mapping.max_size

        self.embedding = nn.Embedding(self.n_pages, emb_size)
        self.delta_embedding = nn.Embedding(self.n_deltas, delta_emb_size)
        self.pc_embedding = nn.Embedding(self.n_pcs, pc_emb_size)

        self.query = nn.Linear(self.concat_emb_size, hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)

        self.rnn = nn.GRU(
            self.concat_emb_size, hidden_size, batch_first=True, dropout=dropout,
            num_layers=rnn_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.ff = nn.Linear(hidden_size, hidden_size*2)
        self.ff_deltas = nn.Linear(hidden_size*2, self.n_deltas)
        self.ff_pcs = nn.Linear(hidden_size*2, self.n_pcs)

        for layer in [self.query, self.key, self.ff_deltas, self.ff_pcs, self.ff]:
            torch.nn.init.xavier_uniform_(layer.weight.data)

    def forward(
        self, a, b, delta_a, delta_b, pc_a, pc_b, history, delta_history, pc_history
    ):
        a = _to_tensor_batched(a, 1)
        b = _to_tensor_batched(b, 1)
        delta_a = _to_tensor_batched(delta_a, 1)
        delta_b = _to_tensor_batched(delta_b, 1)
        history = _to_tensor_batched(history, 2)
        delta_history = _to_tensor_batched(delta_history, 2)
        pc_history = _to_tensor_batched(pc_history, 2)
        pc_a = _to_tensor_batched(pc_a, 1)
        pc_b = _to_tensor_batched(pc_b, 1)

        emb_a = torch.cat(
            [self.embedding(a), self.delta_embedding(delta_a), self.pc_embedding(pc_a)],
            dim=1,
        ).unsqueeze(1)
        emb_b = torch.cat(
            [self.embedding(b), self.delta_embedding(delta_b), self.pc_embedding(pc_b)],
            dim=1,
        ).unsqueeze(1)

        history_emb = torch.cat(
            [
                self.embedding(history),
                self.delta_embedding(delta_history),
                self.pc_embedding(pc_history),
            ],
            dim=2,
        )

        if isinstance(self.rnn, nn.LSTM):
            _, (history_hn, __) = self.rnn(history_emb)
            history_hidden = history_hn.squeeze(0).unsqueeze(1)
        elif isinstance(self.rnn, nn.GRU):
            _, history_hn = self.rnn(history_emb)
            history_hidden = history_hn[-1].unsqueeze(1)
        else:
            raise ValueError("Invalid RNN type")

        # history_hidden = history_hn
        # history_hidden = history_emb
        query_a_b = self.query(torch.cat([emb_a, emb_b], dim=1))
        # query_a_b = self.dropout(history_hidden)

        # key = self.layer_norm(history_hidden)
        key = self.key(history_hidden)
        scores = query_a_b @ key.transpose(1, 2)
        scores /= torch.sqrt(torch.tensor(self.hidden_size).float())
        out = scores.squeeze(-1)

        out = torch.softmax(out, dim=1)

        if self.training:
            out_dict = {"prob_ab": out}
        else:
            out_dict = {"prob_ab": out[:, 0]}

        if not self.training and not (self.train_next_delta or self.train_next_pc):
            return out_dict

        ff = history_hidden.squeeze(1)

        if self.train_next_delta:
            out_deltas = self.ff_deltas(ff)
            out_deltas = torch.softmax(out_deltas, dim=1)
            out_dict["next_delta"] = out_deltas

        if self.train_next_pc:
            out_pcs = self.ff_pcs(ff)
            out_pcs = torch.softmax(out_pcs, dim=1)
            out_dict["next_pc"] = out_pcs

        return out_dict


def to_delta_sequence(sequence: list[int], addr: int) -> list[int]:
    return [s - addr for s in sequence]


def to_delta_sequence_next(sequence: list[int]) -> int:
    return [sequence[i] - sequence[i - 1] for i in range(1, len(sequence))]


def find_mistakes(
    history: AddrHistory,
    cache_history: CacheHistory,
    samples: int,
    addr_mapping: Mapping,
    delta_mapping: Mapping,
    pc_mapping: Mapping,
    model: nn.Module,
    interval: int,
    classify_cutoff: float = 0.35,
):

    mistakes = []
    T = len(history.addrs)

    model_inputs = []
    ys = []
    for _ in range(samples):

        t = random.randint(
            max(T - 2 - 2 * interval + 1, history.history_size), T - 2 - interval
        )

        cache_t = cache_history[t]

        if len(cache_t) < 2:
            continue

        for (a, pc_a), (b, pc_b) in itertools.combinations(cache_t, 2):
            model_input = get_model_input(
                a=a,
                b=b,
                pc_a=pc_a,
                pc_b=pc_b,
                t=t,
                history=history,
                delta_mapping=delta_mapping,
                addr_mapping=addr_mapping,
                pc_mapping=pc_mapping,
            )
            y = get_target_output(a, b, t, history)

            ys.append(y)
            model_inputs.append(model_input)

    inputs = to_batched_dict(model_inputs)

    model.eval()
    with torch.no_grad():
        probs = model(**inputs)["prob_ab"]

    for y, prob in zip(ys, probs):
        c = classify_cutoff
        if prob < 0.5 - c:
            y_model = 0
        elif prob < 0.5 + c:
            y_model = 0.5
        else:
            y_model = 1

        if y != y_model:
            mistakes.append((True, y, y_model))
        else:
            mistakes.append((False, y, y_model))

    return mistakes


def get_samples(
    addr_history: AddrHistory,
    cache_history: CacheHistory,
    delta_mapping: Mapping,
    addr_mapping: Mapping,
    pc_mapping: Mapping,
    interval=80,
    n_samples=1000,
    cache_replacement=5,
):

    samples = []

    while len(samples) < n_samples:
        T = len(addr_history.addrs)
        t = random.randint(
            max(T - 2 - interval * 2 + 1, addr_history.history_size), T - 2 - interval
        )

        cache_t = cache_history[t]

        cache_t = random.sample(cache_t, len(cache_t))

        if len(cache_t) < 2:
            continue

        for (a, pc_a), (b, pc_b) in itertools.combinations(cache_t, 2):
            y1 = get_target_output(a, b, t, addr_history)
            next_delta = delta_mapping(
                addr_history.addrs[t + 1] - addr_history.addrs[t]
            )
            next_pc = pc_mapping(addr_history.pcs[t + 1])
            model_input1 = get_model_input(
                a=a,
                b=b,
                pc_a=pc_a,
                pc_b=pc_b,
                t=t,
                history=addr_history,
                delta_mapping=delta_mapping,
                addr_mapping=addr_mapping,
                pc_mapping=pc_mapping,
            )

            samples.append(
                {"y": y1, **model_input1, "next_delta": next_delta, "next_pc": next_pc}
            )

            model_input2 = get_model_input(
                a=b,
                b=a,
                pc_a=pc_b,
                pc_b=pc_a,
                t=t,
                history=addr_history,
                delta_mapping=delta_mapping,
                addr_mapping=addr_mapping,
                pc_mapping=pc_mapping,
            )
            y2 = get_target_output(b, a, t, addr_history)

            assert math.isclose(y1 + y2, 1.0)

            samples.append(
                {"y": y2, **model_input2, "next_delta": next_delta, "next_pc": next_pc}
            )

    return samples


def to_batched_dict(list_dict):
    def _map(k):
        if k == "y":
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
    batch_size=32,
    loss_delta_weight=0.1,
    loss_pc_weight=0.1,
    loss_ab_weight=1.0,
):

    loss_a_b_sum = 0
    loss_pc_sum = 0
    loss_delta_sum = 0
    random_loss_sum = 0
    criterion_a_b = nn.CrossEntropyLoss()
    criterion_delta = nn.CrossEntropyLoss()
    criterion_pc = nn.CrossEntropyLoss()

    model.train()
    for i in range(0, len(samples), batch_size):
        batch = to_batched_dict(samples[i : i + batch_size])
        prob_target = batch.pop("y")
        prob_target = prob_target.unsqueeze(1)
        prob_target = torch.cat([prob_target, 1 - prob_target], dim=1)
        next_delta = batch.pop("next_delta")
        next_pc = batch.pop("next_pc")

        model_out = model(**batch)
        random_model_out = torch.rand_like(model_out["prob_ab"])
        random_model_out = torch.softmax(random_model_out, dim=1)

        probs_ab = model_out["prob_ab"]

        loss_a_b = criterion_a_b(probs_ab, prob_target)
        loss_a_b_sum += loss_a_b.item()
        random_loss_sum += criterion_a_b(random_model_out, prob_target).item()
        loss = loss_a_b * loss_ab_weight

        if "next_delta" in model_out:
            distr_delta = model_out["next_delta"]
            next_delta_target = torch.zeros_like(distr_delta)
            next_delta_target[range(len(next_delta)), next_delta] = 1
            loss_delta = criterion_delta(distr_delta, next_delta)

            loss += loss_delta * loss_delta_weight
            loss_delta_sum += loss_delta.item()

        if "next_pc" in model_out:
            distr_pc = model_out["next_pc"]
            next_pc_target = torch.zeros_like(distr_pc)
            next_pc_target[range(len(next_pc)), next_pc] = 1
            loss_pc = criterion_pc(distr_pc, next_pc_target)
            loss += loss_pc * loss_pc_weight
            loss_pc_sum += loss_pc.item()

        loss.backward()
        optimizer.step()

    batches = math.ceil(len(samples) / batch_size)
    loss_a_b_sum /= batches
    loss_delta_sum /= batches
    loss_pc_sum /= batches
    random_loss_sum /= batches

    return loss_a_b_sum, loss_delta_sum, loss_pc_sum, random_loss_sum


def dominating_distribution(solver, pages, w):
    for page_i in pages:
        for page_j in pages:
            if page_i == page_j:
                continue
            assert math.isclose(w[(page_i, page_j)] + w[(page_j, page_i)], 1.0)

    model = pl.LpProblem(sense=pl.LpMinimize)
    x = []
    for i in range(len(pages)):
        x.append(pl.LpVariable(name="x" + str(i), lowBound=0))
    c = pl.LpVariable(name="c", lowBound=0)
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

    distr = np.array([x[i].value() for i in range(len(pages))], dtype=np.float64)
    return (distr / distr.sum()).tolist()


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_to_evict(
    t: int,
    addr_mapping: Mapping,
    delta_mapping: Mapping,
    addr_history: AddrHistory,
    cache_history: CacheHistory,
    model,
    greedy=True,
    classify_cutoff=0.1,
    solver=None,
):
    cache_t = cache_history.get_addr_history(t - 1)

    if t < addr_history.history_size:
        return random.choice(cache_t)

    model_input = []

    page_pairs = [(a, b) for a, b in itertools.permutations(cache_t, 2)]

    for a, b in page_pairs:
        model_input.append(
            get_model_input(
                a=a,
                b=b,
                pc_a=cache_history.get_pc(a),
                pc_b=cache_history.get_pc(b),
                t=t,
                history=addr_history,
                delta_mapping=delta_mapping,
                addr_mapping=addr_mapping,
                pc_mapping=addr_mapping,
            )
        )

    model.eval()
    with torch.no_grad():
        probs = model(**to_batched_dict(model_input))["prob_ab"]
        probs = probs.tolist()

    pair_probabilities = {}
    for (a, b), prob in zip(page_pairs, probs):
        pair_probabilities[(a, b)] = prob

    in_edges = defaultdict(int)
    probs_mean = defaultdict(float)
    for (a, b) in page_pairs:
        if a > b:
            continue

        probs_mean[(a, b)] = (
            pair_probabilities[(a, b)] + 1 - pair_probabilities[(b, a)]
        ) / 2
        probs_mean[(b, a)] = 1 - probs_mean[(a, b)]
        c = classify_cutoff
        in_edges[a] += pair_probabilities[(a, b)] > 0.5 + c
        in_edges[b] += pair_probabilities[(b, a)] > 0.5 + c
        if 0.5 - c < in_edges[a] < 0.5 + c:
            in_edges[a] += 1
            in_edges[b] += 1

    distr = dominating_distribution(solver, cache_t, probs_mean)
    entropy = -sum(p * math.log(p) for p in distr if p > 0)
    variance = sum(p * (1 - p) for p in distr)

    if greedy:
        to_evict_idx = max(range(len(cache_t)), key=lambda i: in_edges[cache_t[i]])
        # to_evict_idx = np.argmax(distr)
        # x = [in_edges[cache_addrs_t[i]] for i in range(len(cache_addrs_t))]
        # to_evict_idx = random.choices(
        #    range(len(cache_addrs_t)), k=1, weights=softmax(x)
        # )[0]
    else:
        to_evict_idx = np.random.choice(len(cache_t), p=distr)

    return cache_t[to_evict_idx]


def test_on_sequence(
    sequence_rw: list[tuple[int, str, int]],
    cache_size,
    addr_mapping: Mapping,
    delta_mapping: Mapping,
    pc_mapping: Mapping,
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
    pc_emb_size=10,
    cache_replacement=5,
    batch_size=32,
    random_evictor_seed=42,
    count_mistakes=False,
    classify_cutoff=0.35,
    loss_delta_weight=0.1,
    loss_pc_weight=0.1,
    loss_ab_weight=1.0,
    rnn_layers=2,
):
    solver = pl.getSolver("GLPK_CMD", msg=False)
    model_config = dict(
        cache_size=cache_size,
        history_size=history_size,
        emb_size=emb_size,
        delta_emb_size=delta_emb_size,
        pc_emb_size=pc_emb_size,
        hidden_size=hidden_size,
        dropout=dropout,
        train_next_delta=loss_delta_weight > 0,
        train_next_pc=loss_pc_weight > 0,
        addr_mapping=addr_mapping,
        delta_mapping=delta_mapping,
        pc_mapping=pc_mapping,
        rnn_layers=rnn_layers,
    )

    model = PCRnn(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    misses = 0
    misses_history = []
    losses_a_b = []
    random_losses = []
    losses_delta = []
    losses_pc = []
    cache_history = CacheHistory(cache_size)
    addr_history = AddrHistory(history_size, addr_mapping)

    evict_counts = defaultdict(int)
    mistakes_0 = []
    mistakes_05 = []
    mistakes_1 = []

    random_cache = set()
    random_cache_misses = 0

    random_generator = random.Random(random_evictor_seed)
    entropy_sum = 0
    entropy_count = 0
    reads_sequence = [(x[2], x[0]) for x in sequence_rw if x[1] == "R"]
    pbar = tqdm.tqdm(list(enumerate(reads_sequence)))

    for t, (addr, pc) in pbar:
        addr_history.update(addr, pc)

        if len(cache_history) < cache_size or addr in cache_history:
            cache_history.add(addr, pc)
        else:
            to_evict = get_to_evict(
                t=t,
                addr_mapping=addr_mapping,
                delta_mapping=delta_mapping,
                addr_history=addr_history,
                cache_history=cache_history,
                model=model,
                greedy=greedy,
                classify_cutoff=classify_cutoff,
                solver=solver,
            )
            cache_history.evict(to_evict)
            cache_history.add(addr, pc)
            misses += 1

        cache_history.save()

        if len(random_cache) < cache_size:
            random_cache.add(addr)
        elif addr not in random_cache:
            to_evict = random_generator.choice(list(random_cache))
            random_cache_misses += 1
            random_cache.remove(to_evict)
            random_cache.add(addr)

        misses_history.append(misses)

        if (
            count_mistakes
            and t % train_interval == 0
            and t >= train_interval * 3
            and t > history_size * 2
        ):
            _mistakes = find_mistakes(
                addr_history,
                cache_history,
                train_samples,
                addr_mapping,
                delta_mapping,
                pc_mapping,
                model,
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

        if (
            t % train_interval == 0
            and t >= train_interval * 2
            and t > history_size * 2
        ):
            samples = get_samples(
                addr_history=addr_history,
                cache_history=cache_history,
                delta_mapping=delta_mapping,
                addr_mapping=addr_mapping,
                pc_mapping=pc_mapping,
                interval=train_interval,
                n_samples=train_samples,
                cache_replacement=cache_replacement,
            )
            loss_a_b, loss_delta, loss_pc, random_loss = train_model(
                model,
                optimizer,
                samples,
                batch_size=batch_size,
                loss_delta_weight=loss_delta_weight,
                loss_pc_weight=loss_pc_weight,
                loss_ab_weight=loss_ab_weight,
            )
            losses_a_b.append(loss_a_b)
            losses_delta.append(loss_delta)
            random_losses.append(random_loss)
            losses_pc.append(loss_pc)

        ratio = misses / random_cache_misses if random_cache_misses != 0 else 1
        entropy_mean = entropy_sum / entropy_count if entropy_count != 0 else 0
        hit_ratio = 1 - misses / (t + 1)

        all_mistakes = sum(mistakes_0) + sum(mistakes_05) + sum(mistakes_1)
        all_mistaktes_mean = all_mistakes / (
            len(mistakes_0) + len(mistakes_05) + len(mistakes_1) or 1
        )
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
            #f"Entropy: {entropy_mean:.3f}, "
            f"All Mistakes: {all_mistaktes_mean:.3f}, "
            f"Mistakes 0: {mistakes_0_mean:.3f}, "
            f"Mistakes 0.5: {mistakes_05_mean:.3f}, "
            f"Mistakes 1: {mistakes_1_mean:.3f}, "
            f"Hits: {hit_ratio:.3f}, "
            f"Loss p(a,b): {losses_a_b[-1] if losses_a_b else 0}, "
            f"Loss delta: {losses_delta[-1] if losses_delta else 0}, "
            f"Loss pc: {losses_pc[-1] if losses_pc else 0}, "
            f"Random loss: {random_losses[-1] if random_losses else 0}"
        )


    return misses, misses_history, (losses_a_b, losses_delta, losses_pc), evict_counts, model
