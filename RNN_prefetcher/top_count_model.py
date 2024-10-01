import pandas as pd
from vocab import Vocab, Word
from typing import List, Dict
import torch


class TopCountModel:
    def __init__(self, deltas_count: pd.Series, vocab_out: Vocab, k: int):
        self.deltas_count = deltas_count
        self.vocab_out = vocab_out
        self.topk = torch.zeros(len(vocab_out))

        for i, idx in enumerate(deltas_count.nlargest(k).index):
            self.topk[vocab_out.word2id[str(idx)]] = k + 1 - i

    def forward(self, x: List[List[Word]]) -> torch.Tensor:
        return self.topk.repeat(len(x), 1)

    def __call__(self, *args):
        return self.forward(*args)

    def eval(self):
        ...


class TopCountClusterModel:
    def __init__(self, delta_counts: Dict[int, pd.Series], vocab_out: Vocab, k: int):
        self.vocab_out = vocab_out
        self.topks = {c: torch.zeros(len(vocab_out)) for c in range(6)}

        for c in range(6):
            for i, idx in enumerate(delta_counts[c].nlargest(k).index):
                self.topks[c][vocab_out.word2id[str(idx)]] = k + 1 - i

    def forward(self, x: Dict[int, List[List[Word]]]) -> torch.Tensor:
        out = []

        for c in range(6):
            if c in x:
                out.append(self.topks[c].repeat(len(x[c]), 1))
        return torch.concat(out, dim=0)

    def __call__(self, *args):
        return self.forward(*args)

    def eval(self):
        ...
