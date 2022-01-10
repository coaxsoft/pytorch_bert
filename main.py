import time
from pathlib import Path

from torch.utils.data import DataLoader

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from bert.trainer import BertTrainer

EMB_SIZE = 64
EPOCHS = 4
BATCH_SIZE = 48
NUM_HEADS = 4

CHECKPOINT_DIR = Path('/Users/mikhail/PycharmProjects/pytorch_bert/data/bert_checkpoints')


if __name__ == '__main__':
    ds = IMDBBertDataset('/Users/mikhail/PycharmProjects/rnn_pytorch/data/imdb.csv', max_length=64)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    bert = BERT(len(ds.vocab), EMB_SIZE, 24, NUM_HEADS)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        checkpoint_dir=CHECKPOINT_DIR,
        save_checkpoint_every=50,
        batch_size=BATCH_SIZE,
        learning_rate=0.005
    )

    trainer.print_summary()

    prev = time.time()
    for epoch in range(1, EPOCHS + 1):
        trainer.train(epoch)
        print('-' * 59)
