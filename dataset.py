import glob
import string
import unicodedata
import os

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import numpy as np


class Vocab():
    def __init__(self, max_len, lower=True):

        self.pad_token = "<pad>"
        self.char2id = {self.pad_token:0}
        self.id2char = {0:self.pad_token}
        self.max_len = max_len
        self.lower = lower

    def __len__(self):
        return len(self.char2id)

    def get_ids(self, name):
        ids = []
        if self.lower:
            name=name.lower()
        for ch in name:
            if ch not in self.char2id:
                new_id = len(self.char2id)
                self.char2id[ch] = new_id
                self.id2char[new_id] = ch
            else:
                new_id = self.char2id[ch]
            ids.append(new_id)
        if len(ids) > self.max_len:
            return None

        pad_id = self.char2id[self.pad_token]
        ids = ids + [pad_id] * (self.max_len - len(ids))

        return ids


class NameDataset(Dataset):
    def __init__(self, root_dir=None, max_len=20, vocab=None):
        self.names = []
        self.labels = []
        self.label_map = {}

        self.vocab = Vocab(max_len=max_len) if vocab is None else vocab
        self.root_dir = root_dir

        self.prepare_data()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        X = self.names[index]
        y = self.labels[index]
        return np.array(X), np.array(y)

    def __str__(self):
        str = f'Total names {len(self.names)}'
        return str


    def unicode_to_Ascii(self, s):
        all_letters = string.ascii_letters + ".,;'"
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )


    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_Ascii(line) for line in lines]

    def prepare_data(self):
        count = 0
        filepaths = glob.glob(self.root_dir + "*.txt")  #finds the directory path

        #Reading the files
        for filename in filepaths:
            print(filename)
            label_name = os.path.splitext(os.path.basename(filename))[0]
            if label_name not in self.label_map:
                self.label_map[label_name] = count   #creating label map
                count += 1

            lines = self.read_lines(filename)
            current_label = self.label_map[label_name]
            for name in lines:
                ids = self.vocab.get_ids(name)
                if ids is not None:
                    self.names.append(ids)
                    self.labels.append(current_label)
        print(len(self.labels))
        assert len(self.names) == len(self.labels)


def fetch_dataset(datapath):
    #to do : set seed while splitting
    dataset = NameDataset(datapath)  # creating the dataset
    #print(dataset)
    trainN = int(len(dataset) * 0.8)
    validN = len(dataset) - trainN
    lengths = [trainN, validN]
    trainset, valset = random_split(dataset, lengths)

    train_dl = DataLoader(trainset, batch_size=32, shuffle=True)
    val_dl = DataLoader(valset, batch_size=32, shuffle=False)

    return train_dl, val_dl, dataset.vocab, dataset.label_map

