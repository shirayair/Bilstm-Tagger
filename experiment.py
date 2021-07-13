STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import os
import sys

import torch
import torch.nn as nn
import load_data
from torch import nn as nn, autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

EMBEDDING_VOCAB = 126
EMBEDDING_LENGTH = 30
SAMPLE_SIZE = 500
HIDDEN_LAYER_LSTM = 50
HIDDEN_LAYER = 50
EPOCHS = 5
LR = 0.001
BATCH_SIZE = 50


class LSTM(nn.Module):
    def __init__(self, hidden_layer_lstm, hidden_layer_dim, output_dim, embed_dim, vocab_size):
        super(LSTM, self).__init__()
        self.hidden_layer_lstm = hidden_layer_lstm
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = vocab_size
        self.lstm, self.embed, self.fc1, self.fc2 = self.build_model()
        #nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def build_model(self):
        # whenever the embedding sees the padding index it'll make the whole vector zeros

        word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx
        )
        # design LSTM
        lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_layer_lstm,
            batch_first=True
        )

        # output layer which projects back to tag space
        hidden_to_tag = nn.Linear(self.hidden_layer_lstm, self.hidden_layer_dim)
        hidden_to_tag2 = nn.Linear(self.hidden_layer_dim, self.output_dim)
        return lstm, word_embedding, hidden_to_tag, hidden_to_tag2

    # def init_hidden(self, batch):
    #     hidden_1 = torch.zeros(1, batch, self.hidden_layer_lstm)
    #     hidden_2 = torch.zeros(1, batch, self.hidden_layer_lstm)
    #     # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
    #     return (hidden_1, hidden_2)

    def forward(self, x):
        lens = list(map(len, x))
        padded = pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
        # x_length = [i[99] for i in x]
        x = self.embed(padded)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        # now run through LSTM
        output, (c0, ho) = self.lstm(x)
        # undo the packing operation
        x, lens = pad_packed_sequence(output, batch_first=True)
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        last_seq = x[torch.arange(x.shape[0]), lens - 1]
        x = self.fc1(last_seq)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.softmax(x)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        Y_hat = x
        return Y_hat


def calc_accuracy(prediction, labels):
    good = bed = 0
    for p, l in zip(prediction, labels):
        if int(torch.argmax(p)) == int(l):
            good += 1
        else:
            bed += 1
    return good / (good + bed)


def validation_check(i, model, valid_loader, loss_func):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    print(f'Epoch: {i + 1:02} | Starting Evaluation...')
    for x, y_label in valid_loader:
        prediction, loss = apply(model, loss_func, x, y_label)
        epoch_acc += calc_accuracy(prediction, y_label)
        epoch_loss += loss.item()
    print(f'Epoch: {i + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(valid_loader), float(epoch_acc) / len(valid_loader)


def apply(model, criterion, batch, targets):
    pred = model(batch)
    loss_ = criterion(pred, targets)
    return pred, loss_


def train_and_eval_model(model, data_loader, valid_loader, loss_func):
    torch.manual_seed(3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for i in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for a, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            prediction, loss = apply(model, loss_func, x, y)
            epoch_acc += calc_accuracy(prediction, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i + 1:02} | Finished Training')
        avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(data_loader), float(epoch_acc) / len(data_loader)
        avg_epoch_loss_val, avg_epoch_acc_val = validation_check(i, model, valid_loader, loss_func)
        print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
        print(f'\t Val. Loss: {avg_epoch_loss_val:.3f} |  Val. Acc: {avg_epoch_acc_val * 100:.2f}%')
    return model


if __name__ == "__main__":
    path = "./train"
    path_dev = "./dev"

    data_loader = load_data.make_loader(path, BATCH_SIZE)
    BATCH_SIZE = 500
    data_loader_dev = load_data.make_loader(path_dev, BATCH_SIZE)

    model = LSTM(HIDDEN_LAYER_LSTM, HIDDEN_LAYER, 2, EMBEDDING_LENGTH, EMBEDDING_VOCAB)
    model = train_and_eval_model(model, data_loader, data_loader_dev, nn.CrossEntropyLoss())
