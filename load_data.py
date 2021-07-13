STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


class BatchDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return list(map(torch.tensor, self.data[index]))

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_file(cls, path):
        return BatchDataset(read_data(path))


def read_data(path):
    with open(path, 'r') as fwriter:
        data = fwriter.readlines()
    sequences = []
    for row in data:
        parsed = row.split('\n')
        parsed = parsed[0].split('\t')
        sequences.append(([ord(char) for char in parsed[0]], int(parsed[1])))
    return sequences


def collate_function(data_set):
    return [[item[0] for item in data_set], torch.stack([item[1] for item in data_set])]


def make_loader(path, batch_size):
    data_set = BatchDataset.from_file(path)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_function)


if __name__ == "__main__":
    path = "./train"
    # all_sequences, X_lengths = LTI(sequences)
    data_loader_train = make_loader(path, 50)
    print('hello')
