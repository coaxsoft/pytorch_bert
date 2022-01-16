import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert.dataset import IMDBBertDataset
from bert.model import BERT

BASE_DIR = Path(__file__).resolve().parent

EMB_SIZE = 64
HIDDEN_SIZE = 36
EPOCHS = 4
BATCH_SIZE = 12
NUM_HEADS = 4
LOG_DIR = BASE_DIR.joinpath(f'data/logs/graph')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Prepare dataset")
    ds = IMDBBertDataset(BASE_DIR.joinpath('data/imdb.csv'), ds_from=0, ds_to=5)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    bert = BERT(len(ds.vocab), EMB_SIZE, HIDDEN_SIZE, NUM_HEADS).to(device)
    writer = SummaryWriter(str(LOG_DIR))

    inp, mask, inverse_token_mask, token_target, nsp_target = next(iter(loader))

    writer.add_graph(bert, input_to_model=[inp, mask])
