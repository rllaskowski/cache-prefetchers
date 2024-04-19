from base64 import b16encode
from calendar import c
from copy import deepcopy
from math import e
from collections import defaultdict
import re
from IPython import embed
from sympy import Q, field, im, li
import torch
from dataclasses import dataclass
import torch.nn as nn
from traitlets import default
from zmq import device
import numpy as np
import random
from eviction import find_dom_distribution
import bisect
import itertools
import copy
import tqdm

import cache
from fieldfm import FieldAwareFactorizationMachineModel


def to_tensor_batched(x, batch_dims):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.dim() < batch_dims:
        x = x.unsqueeze(0)

    return x


class Early(nn.Module):
    def __init__(
        self,
        n_pages,
        cache_size,
        history_size,
        emb_size,
        delta_emb_size,
        hidden_size,
        output_size,
        n_deltas=100,
        dropout=0.1,
        train_next_delta=False,
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

    def forward(self, cache, cache_delta, history, delta_history):
        cache = to_tensor_batched(cache, 1)
        cache_delta = to_tensor_batched(cache_delta, 1)
        history = to_tensor_batched(history, 2)
        delta_history = to_tensor_batched(delta_history, 2)

        emb_cache = torch.cat(
            [self.embedding(cache), self.delta_embedding(cache_delta)], dim=2
        )

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
        query_cache = self.query(emb_cache)
        # query_a_b = torch.cat([emb_a, emb_b], dim=1)
        # query_a_b = self.dropout(history_hidden)

        # key = self.layer_norm(history_hidden)
        key = history_hidden
        scores = query_cache @ key.transpose(1, 2)
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
            return key % self.max_size

        if key not in self.mapping:
            self.mapping[key] = min(len(self.mapping), self.max_size - 1)

        return self.mapping[key]

    def _map_list(self, keys):
        return [self.map(x) for x in keys]

    def __call__(self, key):
        return self.map(key)


def to_delta_sequence(sequence, addr):
    return [s - addr for s in sequence]


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
):

    samples = []

    while len(samples) < n_samples:
        t = random.randint(
            max(T - 2 - interval * 3 + 1, history_size), T - 2 - 2 * interval
        )

        cache_t = cache_history[t]

        if len(cache_t) < 2:
            continue

        last_addr = sequence[t]
        history_t = sequence[t - history_size + 1 : t + 1]

        history_t_mapped = addr_mapping(history_t)
        delta_history_t = to_delta_sequence(history_t, last_addr)
        delta_history_t_mapped = delta_mapping(delta_history_t)

        cache_t_mapped = addr_mapping(cache_t)
        delta_cache_t = to_delta_sequence(cache_t, last_addr)
        delta_cache_t_mapped = delta_mapping(delta_cache_t)

        first_occ = {}
        found = {}
        for addr_mapped in cache_t_mapped:
            first_occ[addr_mapped] = bisect.bisect_left(
                occurences[addr_mapped], t+1
            )
            if first_occ[addr_mapped] is None or 



        samples.append(
            {
                ":cache": cache_t_mapped,
                "cache_delta": delta_cache_t_mapped,
                "history": history_t_mapped,
                "delta_history": delta_history_t_mapped,
                "next_delta": delta_mapping(sequence[t + 1] - last_addr),
                "prob": y,
            }
        )

    return samples


def to_batched_dict(list_dict):
    return {k: torch.tensor([d[k] for d in list_dict]) for k in list_dict[0]}


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
    criterion_a_b = nn.BCELoss()
    criterion_delta = nn.CrossEntropyLoss()

    for i in range(0, len(samples), batch_size):
        batch = to_batched_dict(samples[i : i + batch_size])
        prob_target = torch.tensor(batch.pop("prob")).float()

        next_delta = batch.pop("next_delta")

        optimizer.zero_grad()
        model.train()
        model_out = model(**batch)
        random_model_out = torch.rand_like(prob_target)

        if train_next_delta:
            probs_a_b, distr_delta = model_out
        else:
            probs_a_b = model_out

        loss_a_b = criterion_a_b(probs_a_b, prob_target)
        loss_a_b_sum += loss_a_b.item()
        random_loss_sum += criterion_a_b(random_model_out, prob_target).item()
        loss = loss_a_b

        if train_next_delta:
            next_delta = torch.tensor(next_delta).long()
            next_delta_target = torch.zeros((batch_size, n_deltas))
            next_delta_target[range(len(next_delta)), next_delta] = 1
            loss_delta = criterion_delta(distr_delta, next_delta)

            loss += 0.5 * loss_delta
            loss_delta_sum += loss_delta.item()

        loss.backward()
        optimizer.step()

    return loss_a_b_sum, loss_delta_sum, random_loss_sum


def test_on_sequence(
    sequence,
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
    hash_elements=False,
    delta_emb_size=10,
    cache_replacement=5,
    batch_size=32,
    train_next_delta=False,
    hash_mapping=False,
    model_class='ffm'
):
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
        drouput=dropout,
        train_next_delta=train_next_delta,
    )
    if model_class == 'ffm':
        model = FFM(**model_config)
    else:
        model = Early(**model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    misses = 0
    misses_history = []
    losses_a_b = []
    random_losses = []
    losses_delta = []
    cache_history = []
    occurences = defaultdict(list)
    cache = set()
    pbar = tqdm.tqdm(list(enumerate(sequence)))
    evict_counts = defaultdict(int)

    def get_to_evict(t) -> int:
        model_input = []
        cache_tuple = cache_history[t]

        for a, b in itertools.combinations(cache_tuple, 2):
            if a == b:
                continue

            a_mapped = addr_mapping(a)
            b_mapped = addr_mapping(b)
            last_addr = sequence[t]
            a_delta_mapped = delta_mapping(a - last_addr)
            b_delta_mapped = delta_mapping(b - last_addr)
            history = sequence[t - history_size + 1 : t + 1]
            history_mapped = addr_mapping(history)
            delta_history = to_delta_sequence(history, last_addr)
            delta_history_mapped = delta_mapping(delta_history)

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

        with torch.no_grad():
            model.eval()
            probs = model(**to_batched_dict(model_input))
            probs = probs.tolist()

        linear_program_input = {}
        for (a, b), prob in zip(itertools.combinations(cache_tuple, 2), probs):
            linear_program_input[(a, b)] = prob
            linear_program_input[(b, a)] = 1 - prob

        distr = find_dom_distribution(linear_program_input, cache_tuple)

        if greedy:
            to_evict_idx = np.argmax(distr)
        else:
            to_evict_idx = np.random.choice(len(cache_tuple), p=distr)

        return cache_tuple[to_evict_idx]

    for t, addr in pbar:
        addr_mapped = addr_mapping.map(addr)
        occurences[addr_mapped].append(t)
        cache_history.append(tuple(cache))

        if len(cache) < cache_size:
            cache.add(addr)

        elif addr not in cache:
            to_evict = get_to_evict(t)
            evict_counts[to_evict] += 1
            cache.remove(to_evict)
            cache.add(addr)
            misses += 1

        misses_history.append(misses)

        if t % train_interval == 0 and t >= train_interval * 3:
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

        pbar.set_description(
            f"Misses: {misses}, "
            f"Loss p(a,b): {losses_a_b[-1] if losses_a_b else 0}, "
            f"Loss delta: {losses_delta[-1] if losses_delta else 0}, "
            f"Random loss: {random_losses[-1] if random_losses else 0}"
        )

    return misses, misses_history, losses_a_b, evict_counts
