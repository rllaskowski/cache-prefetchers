from __future__ import annotations

import numpy as np
from typing import List, Union, Iterable
from pathlib import Path
from vocab import PAD_TOKEN, Word, UNK_TOKEN
import pandas as pd
from vocab import Vocab
import tqdm
import math
import matplotlib.pyplot as plt
import torch
import random

COLORS = plt.cm.tab20.colors


def pad_sentences(sentences: List[List[Word]]) -> List[List[Word]]:
    max_len = max(len(s) for s in sentences)

    return [[s[i] if i < len(s) else PAD_TOKEN for i in range(max_len)] for s in sentences]


def load_deltas(path: Union[str, Path]) -> List[List[Word]]:
    data = []

    with open(path) as f:
        for line in f:
            values = line.strip().split(",")
            if len(values) > 0:
                data.append(values)

    return data


def generate_train_test_data(deltas, out_train_path, out_test_path):
    deltas = list(deltas)[:]
    random.shuffle(deltas)
    deltas_train = deltas[: 7 * len(deltas) // 10]
    deltas_test = deltas[7 * len(deltas) // 10 :]

    with open(out_train_path, "w") as file:
        for d in deltas_train:
            file.write(",".join(map(str, d)) + "\n")

    with open(out_test_path, "w") as file:
        for d in deltas_test:
            file.write(",".join(map(str, d)) + "\n")

    return deltas_train, deltas_test


def generate_vocab(
    delta_count: pd.Series,
    path: str = None,
    threshold: int = None,
    percent: float = None,
    num: int = None,
    with_padding: bool = True,
):
    assert threshold or percent or num

    if threshold:
        data = delta_count[delta_count > threshold]
    elif percent:
        data = delta_count.nlargest(int(delta_count.shape[0] * percent))
    elif num:
        data = delta_count.nlargest(num)

    vocab = Vocab(data.astype("str").index.to_numpy(), with_padding=with_padding)
    if path:
        vocab.save(path)

    return vocab


def load_cache_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", skipinitialspace=True, dtype=str)
    df = df.rename(columns=lambda c: c.strip())

    if "print data" in df.columns:
        df.columns = list(df.columns[1:]) + ["temp"]
        df = df.drop(columns=["temp"])
    else:
        df = df.drop(columns=df.columns[0])
        df = df.drop(columns=filter(lambda x: "Unnamed" in x, df.columns))

    df = df.applymap(lambda x: x.strip() if type(x) == str else x)

    return df


def get_addresses(df: pd.DataFrame) -> np.ndarray:
    read_miss = df[df["tp"] == "READ_MISS"]["objLba"].dropna().to_numpy()
    try:
        read_miss = read_miss.astype("int64")
    except:
        read_miss = np.vectorize(int)(read_miss, 16)

    return read_miss


def get_deltas(df: pd.DataFrame, filter_fn=None) -> np.ndarray:
    if filter_fn:
        return np.diff(list(filter(filter_fn, get_addresses(df))))
    return np.diff(get_addresses(df))


def generate_address_counts(out_path: str, cache_files) -> pd.Series:
    address_count = pd.Series([])

    for path in tqdm.tqdm(cache_files):
        try:
            addresses = get_addresses(load_cache_trace(path))
            # if len(addresses) > 0:
            # print(addresses)
        except Exception as exc:
            print(exc, path)
            continue
        address_count = address_count.add(pd.value_counts(addresses), fill_value=0)

    address_count.to_csv(out_path, header=False)

    return address_count


def generate_deltas(out_deltas_path: str, cache_files, address_filter_fn=None):
    all_deltas = []

    with open(out_deltas_path, "w") as deltas_file:
        for path in tqdm.tqdm(cache_files):
            try:
                deltas = get_deltas(load_cache_trace(path), address_filter_fn)
            except Exception as exc:
                print(exc, path)
                continue
            if deltas.shape[0] > 0:
                deltas_file.write(",".join(map(str, deltas)) + "\n")
                all_deltas.append(deltas)

    return all_deltas


def generate_delta_counts(deltas: Iterable, out_path: str = None) -> pd.Series:
    delta_count = pd.Series([], dtype="int64")

    for d in tqdm.tqdm(deltas):
        delta_count = delta_count.add(pd.value_counts(d), fill_value=0)

    if out_path:
        delta_count.to_csv(out_path, header=False)

    return delta_count


def deltas_to_addresses(deltas: Iterable):
    return np.cumsum(np.array(deltas, dtype="int64"))


def load_series(path: str) -> pd.Series:
    series = pd.read_csv(path, index_col=0, header=None, dtype="int64").squeeze("columns")

    return series.rename_axis(None)


def batch_iter(data, batch_size, shuffle=True, seq_len=64, show_progress=True):
    data = list(filter(lambda d: len(d) > seq_len, data))
    index_array = list((i, j) for i in range(len(data)) for j in range(len(data[i]) - seq_len))

    batch_num = math.ceil(len(index_array) / batch_size)

    if shuffle:
        np.random.shuffle(index_array)

    for i in tqdm.tqdm(range(batch_num), disable=not show_progress):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        examples = [data[idx][idx2 : idx2 + seq_len] for idx, idx2 in indices]
        targets = [[data[idx][idx2 + seq_len]] for idx, idx2 in indices]

        yield examples, targets


def cluster_batch_iter(data, batch_size: int, shuffle=True, seq_len=64):
    data = {c: list(filter(lambda d: len(d) > seq_len, data[c])) for c in range(6)}
    index_array = list(
        (c, i, j)
        for c in range(6)
        for i in range(len(data[c]))
        for j in range(len(data[c][i]) - seq_len)
    )

    batch_num = math.ceil(len(index_array) / batch_size)

    if shuffle:
        np.random.shuffle(index_array)

    for i in tqdm.tqdm(range(batch_num)):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        examples = {c: [] for c in data}
        targets = {c: [] for c in data}
        for c, idx1, idx2 in indices:
            examples[c].append(data[c][idx1][idx2 : idx2 + seq_len])
            targets[c].append([data[c][idx1][idx2 + seq_len]])

        yield examples, targets


def evaluate(model, data, vocab_out, criterion=None, n=10, batch_size=64, seq_len=64):
    model.eval()
    val_loss = 0
    val_correct = 0
    s = 0

    with torch.no_grad():
        for (inputs, labels) in batch_iter(data, batch_size, seq_len=seq_len, shuffle=False):
            outputs = model(inputs)
            labels = vocab_out.to_tensor(labels).squeeze(dim=1).to(outputs.device)

            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            outputs.data[:, vocab_out.word2id[UNK_TOKEN]] = -torch.inf
            _, topk_indices = torch.topk(outputs, k=n, dim=1)
            is_in_topk = torch.eq(topk_indices, labels.view(-1, 1))

            val_correct += is_in_topk.sum().item()
            s += 1

    val_accuracy = 100.0 * val_correct / (s * batch_size)

    return val_loss / s, val_accuracy


def cluster_evaluate(model, data, vocab_out, criterion=None, n=10, batch_size=64, seq_len=64):
    model.eval()
    val_loss = 0
    val_correct = 0
    s = 0

    with torch.no_grad():
        for (inputs, labels) in cluster_batch_iter(
            data, batch_size, seq_len=seq_len, shuffle=False
        ):
            outputs = model(inputs)
            labels_ = []
            for c in range(6):
                labels_.extend(labels[c])
            labels = vocab_out.to_tensor(labels_).squeeze(dim=1).to(outputs.device)

            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            outputs.data[:, vocab_out.word2id[UNK_TOKEN]] = -torch.inf
            _, topk_indices = torch.topk(outputs, k=n, dim=1)
            is_in_topk = torch.eq(topk_indices, labels.view(-1, 1))

            val_correct += is_in_topk.sum().item()
            s += 1

    val_accuracy = 100.0 * val_correct / (s * batch_size)

    return val_loss / s, val_accuracy


def scatter_cache_traces(paths: List[str], address_cluster=None):
    nrows = math.ceil(len(paths) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows), squeeze=False)

    for i, path in enumerate(paths):
        df = load_cache_trace(path)
        cache_addresses = get_addresses(df)

        if address_cluster:
            colors = [COLORS[address_cluster[a]] for a in cache_addresses]
            axes[i // 2][i % 2].scatter(range(len(cache_addresses)), cache_addresses, c=colors)
        else:
            axes[i // 2][i % 2].scatter(range(len(cache_addresses)), cache_addresses)

        axes[i // 2][i % 2].set_title(path)
        axes[i // 2][i % 2].set_xlabel("time step")
        axes[i // 2][i % 2].set_ylabel("address")

    plt.show()


def scatter_results(inputs, results, labels, vocab_out, n=10, axs=None):
    batch_size = len(inputs)

    if not axs:
        fig, axs = plt.subplots(
            nrows=batch_size,
            ncols=2,
            squeeze=False,
            figsize=(15, 5 * batch_size + 4 * (batch_size - 1)),
        )

    for b in range(batch_size):
        addresses = deltas_to_addresses(inputs[b])
        results[b][vocab_out.word2id[UNK_TOKEN]] = -torch.inf
        result_deltas = vocab_out.to_words([results[b].topk(n).indices])[0]
        result_addresses = [addresses[-1] + int(k) for k in result_deltas]
        label_address = int(labels[b][0]) + addresses[-1]

        axs[b][0].scatter(np.arange(len(addresses)), addresses)
        axs[b][0].scatter(
            [len(addresses)] * len(result_deltas), result_addresses, c=[10] * len(result_deltas)
        )
        axs[b][0].set_xlabel("timestep")
        axs[b][0].set_ylabel("address")
        axs[b][0].set_title(f"Model top {n} deltas : \n{result_deltas}")

        axs[b][1].scatter(np.arange(len(addresses)), addresses)
        axs[b][1].scatter([len(addresses)], [label_address], c=[10])
        axs[b][1].set_xlabel("timestep")
        axs[b][1].set_ylabel("address")
        axs[b][1].set_title(f"Target delta : \n{int(labels[b][0])}")

    return plt.show()


def scatter_cluster_data_results(data, model, cluster, vocab_out, batches=1, n=10, shuffle=True):
    batches = cluster_batch_iter(data, batches, shuffle=shuffle)
    batch = next(batches)
    inputs, labels = batch
    results = {c: model({c: inputs[c]}) for c in range(6)}

    return scatter_results(inputs[cluster], results[cluster], labels[cluster], vocab_out, n=n)


def scatter_data_results(data, model, vocab_out, batches=1, n=10, shuffle=True):
    batches = batch_iter(data, batches, shuffle=shuffle)
    batch = next(batches)
    inputs, labels = batch
    results = model(inputs)

    return scatter_results(inputs, results, labels, vocab_out, n=n)
