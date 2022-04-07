import datetime

import torch

from pathlib import Path

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from bert.trainer import BertTrainer

BASE_DIR = Path(__file__).resolve().parent

EMB_SIZE = 64
HIDDEN_SIZE = 36
EPOCHS = 4
BATCH_SIZE = 12
NUM_HEADS = 4

CHECKPOINT_DIR = BASE_DIR.joinpath('data/bert_checkpoints')

timestamp = datetime.datetime.utcnow().timestamp()
LOG_DIR = BASE_DIR.joinpath(f'data/logs/bert_experiment_{timestamp}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Prepare dataset")
    ds = IMDBBertDataset(BASE_DIR.joinpath('data/imdb.csv'), ds_from=0, ds_to=1000)

    bert = BERT(len(ds.vocab), EMB_SIZE, HIDDEN_SIZE, NUM_HEADS).to(device)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        print_progress_every=20,
        print_accuracy_every=200,
        batch_size=BATCH_SIZE,
        learning_rate=0.00007,
        epochs=15
    )

    trainer.print_summary()
    trainer()
