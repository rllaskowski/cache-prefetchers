from __future__ import annotations

from typing import List, Union
import torch
import json


Word = Union[str, int]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocab:
    def __init__(self, corpus: List[Word], with_padding=True):
        self.word2id = {}
        if with_padding:
            self.word2id[PAD_TOKEN] = 0
            self.word2id[UNK_TOKEN] = 1
        else:
            self.word2id[UNK_TOKEN] = 0

        d = len(self.word2id)
        self.word2id.update({str(word): i + d for i, word in enumerate(corpus)})

        self.id2word = {item: key for key, item in self.word2id.items()}

    def to_tensor(self, sentences: List[List[Word]]) -> torch.Tensor:
        assert (
            len(set(len(s) for s in sentences)) == 1
        ), "Sentences must be padded to the same length"

        return torch.Tensor(
            [[self.word2id.get(str(w), self.word2id[UNK_TOKEN]) for w in s] for s in sentences],
        ).long()

    def to_words(self, tensor: torch.Tensor) -> List[List[Word]]:
        sentences = []
        for b in tensor:
            try:
                sentences.append([self.id2word[str(int(i))] for i in b])
            except:
                sentences.append([self.id2word[int(i)] for i in b])
        return sentences

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"word2id": self.word2id, "id2word": self.id2word}, f)

    @classmethod
    def load(cls, path: str) -> Vocab:
        with open(path, "r") as f:
            data = json.load(f)

        vocab = cls([])
        vocab.word2id = data["word2id"]
        vocab.id2word = data["id2word"]

        return vocab

    def __len__(self):
        return len(self.word2id)
