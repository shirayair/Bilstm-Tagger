STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
from random import randint

import torch
from torch.utils.data import DataLoader, Dataset


def read_data(path, isPos=True, word_to_idx=None, labels_to_idx=None, isSub=False):
    vocab_sample_all = set()
    vocab_label_all = set()
    sequences = []
    with open(path, 'r') as freader:
        row = freader.readline().rstrip('\n')
        while row:
            rows = []
            while row != '':
                rows.append(row)
                row = freader.readline().rstrip('\n')
            sentence, vocab_sample, vocab_label = split_to_samples_lables(rows, isPos)
            vocab_sample_all.update(vocab_sample)
            vocab_label_all.update(vocab_label)
            sequences.append(sentence)
            row = freader.readline().rstrip('\n')

    seqseq, word_to_idx, labels_to_idx, vocab_sample_all, vocab_label_all, idx_to_pre_suf = convert_word_to_idx(
        sequences, vocab_sample_all, vocab_label_all, word_to_idx, labels_to_idx, isSub=isSub)
    idx_to_word = {val: key for key, val in word_to_idx.items()}
    return seqseq, word_to_idx, labels_to_idx, vocab_sample_all, vocab_label_all, idx_to_word, idx_to_pre_suf


def read_data_test(path, isPos=True, word_to_idx=None, labels_to_idx=None, isSub=False):
    origin_samples = []
    vocab_sample_all = set()
    sequences = []
    with open(path, 'r') as freader:
        row = freader.readline().rstrip('\n')
        while row:
            rows = []
            while row != '':
                origin_samples.append(row)
                rows.append(row)
                row = freader.readline().rstrip('\n')
            # sentence, vocab_sample, vocab_label = split_to_samples_lables(rows, isPos)
            vocab_sample_all.update(set(rows))
            sequences.append(rows)
            row = freader.readline().rstrip('\n')

    seqseq, word_to_idx, _, vocab_sample_all, vocab_label_all, idx_to_pre_suf = convert_word_to_idx(
        sequences, vocab_sample_all, None, word_to_idx, None, isSub=isSub)
    idx_to_word = {val: key for key, val in word_to_idx.items()}
    return seqseq, word_to_idx, labels_to_idx, vocab_sample_all, vocab_label_all, idx_to_word, idx_to_pre_suf, origin_samples


def split_to_samples_lables(rows, isPos):
    delim = ' ' if isPos else '\t'
    sentence = []
    vocab_sample, vocab_label = set(), set()
    for r in rows:
        split = r.split(delim)
        word = split[0]
        # word = word.lower()
        # word = re.sub('[0-9]', 'DG', word)
        vocab_sample.add(word)
        vocab_label.add(split[1])
        sentence.append((word, split[1]))
    return sentence, vocab_sample, vocab_label


def convert_word_to_idx(sequences, vocab_sample, vocab_label, word_to_idx, labels_to_idx, isSub=False):
    idx_to_pre_suf = None
    if not word_to_idx:
        # vocab_sample.add('UN_KNOWN')
        # prefix,sufix
        if isSub:
            word_to_idx, idx_to_pre_suf, vocab_sample = create_pre_suf_dict(vocab_sample)
            idx_to_pre_suf[0] = (0, 0)
        else:
            word_to_idx = {word: i + 1 for i, word in enumerate(vocab_sample)}
        word_to_idx['PAD'] = 0
        # vocab_label.add('UN_KNOWN')
        vocab_label = list(sorted(vocab_label))
        labels_to_idx = {word: i + 1 for i, word in enumerate(vocab_label)}
        labels_to_idx['PAD'] = 0
        seqseq = [([word_to_idx[word[0]] for word in sentence],
                   [labels_to_idx[tag[1]] for tag in sentence]) for sentence in sequences]
    else:
        if labels_to_idx:
            seqseq = [
                (
                [word_to_idx[word[0]] if word[0] in word_to_idx.keys() else randint(0, len(word_to_idx) - 1) for word in
                 sentence], [labels_to_idx[word[1]] for word in sentence]) for sentence in sequences]
        else:
            seqseq = [
                ([word_to_idx[word] if word in word_to_idx.keys() else randint(0, len(word_to_idx) - 1) for word in
                  sentence], [0 for word in sentence]) for sentence in sequences]
    return seqseq, word_to_idx, labels_to_idx, vocab_sample, vocab_label, idx_to_pre_suf


def create_pre_suf_dict(vocab_sample):
    sub_vocab = set()
    for word in vocab_sample:
        pre_word, suf_word = word[:3], word[-3:]
        sub_vocab.add(suf_word)
        sub_vocab.add(pre_word)
    sub_vocab = vocab_sample.union(sub_vocab)
    sub_vocab = list(sorted(sub_vocab))
    word_to_idx = {word: i + 1 for i, word in enumerate(sub_vocab)}
    idx_to_pre_suf = {}
    for word, idx in word_to_idx.items():
        idx_to_pre_suf[idx] = (word_to_idx[word[:3]], word_to_idx[word[-3:]])
    vocab_sample = set(sub_vocab)
    return word_to_idx, idx_to_pre_suf, vocab_sample


class SeqDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return list(map(torch.tensor, self.data[index]))

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_file(cls, sequences):
        return SeqDataset(sequences)


def collate_function(data_set):
    return [item[0] for item in data_set], [item[1] for item in data_set]


def make_loader(sequences, batch_size):
    data_set = SeqDataset.from_file(sequences)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_function)


def make_loader_test(sequences, batch_size):
    data_set = SeqDataset.from_file(sequences)
    return DataLoader(data_set, batch_size=batch_size, shuffle=False, collate_fn=collate_function)


def process_data(isPos, batch_size, batch_dev_size, isSub=False):
    train_path = './pos/train' if isPos else './ner/train'
    dev_path = './pos/dev' if isPos else './ner/dev'
    seqseq, word_to_idx, labels_to_idx, vocab_sample, vocab_label, idx_to_word, idx_to_pre_suf = read_data(
        train_path, isPos=isPos, isSub=isSub)
    train_loader = make_loader(seqseq, batch_size)
    seqseq_valid, _, _, _, _, _, _ = read_data(dev_path, isPos=isPos,
                                               word_to_idx=word_to_idx,
                                               labels_to_idx=labels_to_idx, isSub=isSub)
    valid_loader = make_loader(seqseq_valid, batch_dev_size)
    return train_loader, valid_loader, vocab_sample, vocab_label, word_to_idx, labels_to_idx, idx_to_word, idx_to_pre_suf


if __name__ == "__main__":
    isPos = True
    isSub = False
    # process_data(isPos)
    print('hello')
