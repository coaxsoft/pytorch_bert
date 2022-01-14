import datetime

import torch

from pathlib import Path

from torch.utils.data import DataLoader

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from bert.trainer import BertTrainer

BASE_DIR = Path(__file__).resolve().parent

EMB_SIZE = 64
EPOCHS = 4
BATCH_SIZE = 8
NUM_HEADS = 4

CHECKPOINT_DIR = BASE_DIR.joinpath('data/bert_checkpoints')

timestamp = datetime.datetime.utcnow().timestamp()
LOG_DIR = BASE_DIR.joinpath(f'data/logs/bert_experiment_{timestamp}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    ds = IMDBBertDataset(BASE_DIR.joinpath('data/imdb.csv'), max_sentence_length=64, max_ds_length=1000)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    bert = BERT(len(ds.vocab), EMB_SIZE, 24, NUM_HEADS).to(device)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        save_checkpoint_every=500,
        print_progress_every=20,
        print_accuracy_every=50,
        batch_size=BATCH_SIZE,
        learning_rate=0.005,
        epochs=5
    )

    trainer.print_summary()
    trainer()
