import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from train_biLSTM import train_and_eval_model
from process_data_3 import process_data

EMBEDDING_DIM = 50
HIDDEN_LAYER_LSTM = 100
EPOCHS = 5
LR = 0.01
BATCH_SIZE = 500
DEV_BATCH_SIZE = 50


class BiLSTM(nn.Module):

    def __init__(self, output_dim, vocab_size):
        super(BiLSTM, self).__init__()
        torch.manual_seed(3)
        self.hidden_layer_lstm = HIDDEN_LAYER_LSTM
        self.output_dim = output_dim
        self.embed_dim = EMBEDDING_DIM
        self.seq_pad_idx, self.label_pad_idx = 0, 0
        self.vocab_size = vocab_size
        self.lstm, self.embed, self.mlp = self.build_model()
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)

    def build_model(self):
        # whenever the embedding sees the padding index it'll make the whole vector zeros

        word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.embed_dim,
            padding_idx=self.seq_pad_idx

        )
        # design LSTM
        lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_layer_lstm,
            bidirectional=True,
            num_layers=2
        )

        # output layer which projects back to tag space
        mlp = nn.Sequential(nn.Linear(2 * self.hidden_layer_lstm, self.output_dim),
                            nn.Tanh(), nn.Softmax(dim=1))
        return lstm, word_embedding, mlp

    def forward(self, x, y):
        lens = list(map(len, x))
        x = pad_sequence(x, batch_first=True, padding_value=self.seq_pad_idx)
        y = pad_sequence(y, batch_first=True, padding_value=self.label_pad_idx)
        x = self.embed(x)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        # now run through LSTM
        output, (c0, ho) = self.lstm(x)
        # undo the packing operation
        x, lens = pad_packed_sequence(output, batch_first=True)
        Y_hats = self.mlp(x)
        return Y_hats, y

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    isPos = True
    train_loader, valid_loader, vocab_sample, vocab_label, word_to_idx, labels_to_idx = process_data(isPos, BATCH_SIZE,
                                                                                                     DEV_BATCH_SIZE)

    if not isPos:
        weights = [1.0, 1.0, 0.1, 1.0, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=labels_to_idx['PAD'], reduction='mean')
    else:
        loss_func = nn.CrossEntropyLoss(ignore_index=labels_to_idx['PAD'], reduction='mean')

    model = BiLSTM(len(labels_to_idx), len(word_to_idx))
    model = train_and_eval_model(model, train_loader, valid_loader, loss_func, labels_to_idx, EPOCHS, LR)
