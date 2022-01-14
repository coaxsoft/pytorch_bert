import random
import typing
from collections import Counter

import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDBBertDataset(Dataset):
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.15

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'

    def __init__(self, path, max_sentence_length=64, max_ds_length=None, should_include_text=False):
        self.ds: pd.Series = pd.read_csv(path)['review']

        if max_ds_length:
            self.ds = self.ds[:max_ds_length]

        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None

        self.max_sentence_length = max_sentence_length
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.NSP_TARGET_COLUMN]

        self.df = self.prepare_nsp()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()
        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()
        nsp_target = torch.Tensor([item[self.NSP_TARGET_COLUMN]]).long()

        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)

        return inp.to(device), attention_mask.to(device), mask_target.to(device), nsp_target.to(device)

    def prepare_nsp(self) -> pd.DataFrame:
        sentences = []
        nsp = []
        max_sentence_len = 0
        for review in self.ds:
            review_sentences = review.split('. ')
            sentences += review_sentences
            max_sentence_len = self._update_max_size(max_sentence_len, review_sentences)

        max_sentence_len = max_sentence_len if max_sentence_len < self.max_sentence_length else self.max_sentence_length
        print(f"Biggest sentence len: {max_sentence_len}")

        print("Create vocabulary")
        for sentence in tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)
        self.vocab = vocab(self.counter, min_freq=2)  # specials= is only in 0.12.0 version

        # Will not work on M1 such as it uses 0.12.0 version
        # 0.11.0 uses this approach to insert specials
        self.vocab.insert_token(self.CLS, 0)
        self.vocab.insert_token(self.PAD, 1)
        self.vocab.insert_token(self.MASK, 2)
        self.vocab.insert_token(self.SEP, 3)
        self.vocab.insert_token(self.UNK, 4)
        self.vocab.set_default_index(4)

        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]

        print("Preprocessing dataset")
        for review in tqdm(self.ds):
            review_sentences = review.split('. ')
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    first, second = self.tokenizer(review_sentences[i]), self.tokenizer(review_sentences[i + 1])
                    nsp.append(self._create_nsp_item(first, second, 1))

                    # False NSP item
                    first, second = self._gen_false_nsp(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_nsp_item(first, second, 0))
        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def _create_nsp_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        updated_first = self._preprocess_sentence(first.copy())
        updated_second = self._preprocess_sentence(second.copy())
        true_nsp_sentence = updated_first + [self.SEP] + updated_second
        true_nsp_indices = self.vocab.lookup_indices(true_nsp_sentence)

        first = self._preprocess_sentence(first.copy(), should_mask=False)
        second = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return true_nsp_sentence, true_nsp_indices, original_nsp_sentence, original_nsp_indices, target
        else:
            return true_nsp_indices, original_nsp_indices, target

    def _update_max_size(self, size: int, sentences: typing.List[str]):
        for v in sentences:
            l = len(v.split())
            if l > size:
                size = l
        return size

    def _gen_false_nsp(self, sentences: typing.List[str]):
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        # To be sure that it's not real next sentence
        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _mask_sentence(self, sentence: typing.List[str]):
        len_s = len(sentence)

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                # All is below 5 is special token
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
        return sentence

    def _pad_sentence(self, sentence: typing.List[str]):
        len_s = len(sentence)

        if len_s >= self.max_sentence_length:
            s = sentence[:self.max_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.max_sentence_length - len_s)
        return s

    def _preprocess_sentence(self, sentence: typing.List[str], should_mask: bool = True):
        if should_mask:
            sentence = self._mask_sentence(sentence)
        sentence = self._pad_sentence([self.CLS] + sentence)

        return sentence


if __name__ == '__main__':
    ds = IMDBBertDataset('/Users/mikhail/PycharmProjects/rnn_pytorch/data/imdb.csv')
    dl = DataLoader(ds, batch_size=64, shuffle=True)
