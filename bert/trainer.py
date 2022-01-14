import time
from datetime import datetime
from pathlib import Path

import torch

from torch import nn
from torch.utils.data import DataLoader

from bert.dataset import IMDBBertDataset
from bert.model import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def percentage(batch_size: int, max_len: int, current: int):
    batched_max = max_len // batch_size
    return round(current / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    s = (result.argmax(1) == target.squeeze(1)).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(result: torch.Tensor, target: torch.Tensor):
    s = (result.argmax(-1) == target).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)


class BertTrainer:

    def __init__(self, model: BERT, dataset: IMDBBertDataset,
                 checkpoint_dir: Path = None,
                 save_checkpoint_every: int = 100,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5,
                 ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoint_every = save_checkpoint_every

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.ml_criterion = nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.epochs = epochs

        self.current_epoch = 0

        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def print_summary(self):
        ds_len = len(self.dataset)

        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()

    def __call__(self):
        for self.current_epoch in range(self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)

    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")

        prev = time.time()
        for i, value in enumerate(self.loader):
            index = i + 1

            self.optimizer.zero_grad()

            inp, mask, mask_target, nsp_target = value
            token, nsp = self.model(inp, mask)

            loss_token = self.ml_criterion(token.transpose(1, 2), mask_target)  # 1D tensor as target is required
            loss_nsp = self.criterion(nsp, nsp_target.squeeze(1))  # 1D tensor as target is required

            loss = loss_token + loss_nsp

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                passed = percentage(self.batch_size, self._ds_len, index)
                s = f"{time.strftime('%H:%M:%S', elapsed)}"
                s += f" | Epoch {epoch + 1} | {index} / {self._batched_len} ({passed}%) | Loss {loss:6.2f}"

                if index % self._accuracy_every == 0:
                    s += f" | NSP accuracy {nsp_accuracy(nsp, nsp_target)} | " \
                         f"Token accuracy {token_accuracy(token, mask_target)}"
                print(s)

            if index % self.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, index, loss)
        return loss

    def save_checkpoint(self, epoch, step, loss):
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.utcnow().timestamp():.0f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()
