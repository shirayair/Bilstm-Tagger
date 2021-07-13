STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import os
import pickle
import sys

import torch
from torch import nn

from bilstm import BiLSTM
from biLSTM_chars import BiLSTM_chars
from biLSTM_sufpref import BiLSTM_sufpref
from biLSTM_concat import BiLSTM_concat

from process_data_3 import read_data, make_loader
from train_biLSTM import apply, calc_accuracy

BATCH = 500
LR = 0.01


def train(model, data_loader, loss_func, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(1):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for (x, y) in data_loader:
            optimizer.zero_grad()
            predictions, loss, y = apply(model, loss_func, x, y)
            epoch_acc += calc_accuracy(predictions, y, labels_to_idx)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {i + 1:02} | Finished Training')
        avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(data_loader), float(epoch_acc) / len(data_loader)
        print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
    return model


if __name__ == "__main__":
    args = sys.argv[1:]
    choice = args[0]
    train_path = args[1]
    model_file = args[2]
    isPos = int(args[3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isSub = True if choice == 'c' else False
    seqseq, word_to_idx, labels_to_idx, vocab_sample, vocab_label, idx_to_word, idx_to_pre_suf = read_data(
        train_path, isPos=isPos, isSub=isSub)
    train_loader = make_loader(seqseq, BATCH)

    if choice == 'a':
        model = BiLSTM(len(labels_to_idx), len(word_to_idx)).to(device)
    elif choice == 'b':
        model = BiLSTM_chars(len(labels_to_idx), idx_to_word).to(device)
    elif choice == 'c':
        model = BiLSTM_sufpref(len(labels_to_idx), len(word_to_idx), idx_to_pre_suf).to(device)
    elif choice == 'd':
        model = BiLSTM_concat(len(labels_to_idx), len(word_to_idx), idx_to_word).to(device)
    else:
        raise ValueError("illegal repr choice")

    if not isPos:
        weights = [1.0, 1.0, 0.1, 1.0, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=labels_to_idx['PAD'], reduction='mean')
    else:
        loss_func = nn.CrossEntropyLoss(ignore_index=labels_to_idx['PAD'], reduction='mean')    model = train(model, train_loader, loss_func, LR)

    # save dics
    dir_path = os.path.dirname(model_file)

    with open(os.path.join(dir_path, 'word_to_idx.p'), 'wb') as fp1:
        pickle.dump(word_to_idx, fp1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir_path, 'labels_to_idx.p'), 'wb') as fp2:
        pickle.dump(labels_to_idx, fp2, protocol=pickle.HIGHEST_PROTOCOL)
    if choice == 'c':
        with open(os.path.join(dir_path, 'idx_to_pre_suf.p'), 'wb') as fp3:
            pickle.dump(idx_to_pre_suf, fp3, protocol=pickle.HIGHEST_PROTOCOL)
    if choice == 'd' or 'b':
        with open(os.path.join(dir_path, 'idx_to_word.p'), 'wb') as fp4:
            pickle.dump(idx_to_word, fp4, protocol=pickle.HIGHEST_PROTOCOL)

    # save model
    model.save(model_file)
