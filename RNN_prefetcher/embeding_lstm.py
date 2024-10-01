from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
from logger import FileLogger
import argparse
from utils import pad_sentences, load_deltas, batch_iter, evaluate
from vocab import Vocab, PAD_TOKEN, Word


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = None


class EmbeddingsLSTM(nn.Module):
    def __init__(
        self, hidden_size, output_size, num_layers, embedding_size, vocab_in, vocab_out, dropout
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        self.dropout = dropout

        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.embedding = nn.Embedding(
            len(vocab_in), embedding_size, padding_idx=vocab_in.word2id[PAD_TOKEN]
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: List[List[Word]]) -> torch.Tensor:
        lengths = [len(s) for s in x]

        out = self.vocab_in.to_tensor(pad_sentences(x)).to(DEVICE)
        out = self.embedding(out)
        out = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out

    @staticmethod
    def load(path: str) -> EmbeddingsLSTM:
        params = torch.load(path, map_location=lambda s, l: s)
        model = EmbeddingsLSTM(**params["kwargs"])
        model.load_state_dict(params["state_dict"])
        model.to(DEVICE)

        return model

    def save(self, path: str, save_params=True) -> None:
        params = {
            "state_dict": self.state_dict(),
        }
        if save_params:
            params["kwargs"] = {
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "num_layers": self.num_layers,
                "embedding_size": self.embedding_size,
                "vocab_out": self.vocab_out,
                "vocab_in": self.vocab_in,
                "dropout": self.dropout,
            }
        torch.save(params, path)


def train(model, data, criterion, optimizer, vocab_out, batch_size=64, seq_len=64):
    model.train()
    train_loss = 0
    s = 0
    for (inputs, labels) in batch_iter(data, batch_size, seq_len=seq_len):
        labels = vocab_out.to_tensor(labels).squeeze(dim=1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        s += 1

    return train_loss / s


def get_experiment_dir(arg_dir):
    experiment_dir = Path(arg_dir)

    i = 0
    while (experiment_dir / str(i)).exists():
        i += 1

    experiment_dir = experiment_dir / str(i)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir


def get_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Size of the hidden layer of the LSTM"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="Size of the word embedding"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch sizes during training")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument(
        "--vocab_in",
        type=str,
        default="./vocabs/e_lstm/vocab_in_th_10.json",
        help="Path to the input vocabulary",
    )
    parser.add_argument(
        "--vocab_out",
        type=str,
        default="./vocabs/e_lstm/vocab_out_percent_10.json",
        help="Path to the output vocabulary",
    )
    parser.add_argument(
        "--train_data", type=str, default="./data/e_lstm/train_deltas", help="Path to train data"
    )
    parser.add_argument(
        "--test_data", type=str, default="./data/e_lstm/test_deltas", help="Path to test data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--evaluate_n", type=int, default=10, help="N used during evaluation")
    parser.add_argument("--seq_len", type=int, default=64, help="Context sequence length")
    parser.add_argument(
        "--dir", type=str, default="./experiments/e_lstm/", help="Experiment dir path"
    )

    return parser.parse_args()


def main():
    args = get_args()

    experiment_dir = get_experiment_dir(args.dir)

    global logger
    logger = FileLogger("logger", filename=experiment_dir / "logs")

    with open(experiment_dir / "args.json", "w") as f:
        json.dump(args.__dict__, f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_data = load_deltas(args.train_data)
    test_data = load_deltas(args.test_data)

    vocab_in = Vocab.load(args.vocab_in)
    vocab_out = Vocab.load(args.vocab_out)

    model = EmbeddingsLSTM(
        hidden_size=args.hidden_size,
        output_size=len(vocab_out),
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        vocab_in=vocab_in,
        vocab_out=vocab_out,
    )

    model.to(DEVICE)

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}:")

        print("Train")
        train_loss = train(model, train_data, criterion, optimizer, vocab_out, seq_len=args.seq_len)
        logger.log_training_loss(epoch, train_loss)
        print(f"Train loss: {train_loss:.4f}")

        print("Evaluate")
        val_loss, val_accuracy = evaluate(
            model,
            test_data,
            vocab_out,
            criterion,
            n=args.evaluate_n,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        logger.log_validation_loss(epoch, val_loss)
        logger.log_accuracy(epoch, val_accuracy)

        print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.2f}%")

        model.save(experiment_dir / f"checkpoint_{epoch}.pt", save_params=False)

    model.save(experiment_dir / "model.pt")


if __name__ == "__main__":
    main()
