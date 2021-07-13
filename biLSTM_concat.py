import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from process_data_3 import process_data
from train_biLSTM import train_and_eval_model

EMBEDDING_DIM = 100
HIDDEN_LAYER_LSTM = 50
EPOCHS = 5
LR = 0.001
BATCH_SIZE = 100
DEV_BATCH_SIZE = 50
CHAR_DIM_EMBED = 30
VOCAB_CHARS = 126


class BiLSTM_concat(nn.Module):

    def __init__(self, output_dim, vocab_size, idx_to_word):
        super(BiLSTM_concat, self).__init__()
        torch.manual_seed(3)
        self.idx_to_word = idx_to_word
        self.hidden_layer_lstm = HIDDEN_LAYER_LSTM
        self.output_dim = output_dim
        self.embed_dim = EMBEDDING_DIM
        self.char_dim = CHAR_DIM_EMBED
        self.seq_pad_idx, self.label_pad_idx = 0, 0
        self.char_pad = VOCAB_CHARS
        self.vocab_size = vocab_size
        self.lstm, self.word_embed, self.char_embed, self.mlp = self.build_model()
        self.lstm_chars = nn.LSTM(input_size=self.char_dim, hidden_size=self.char_dim)
        nn.init.uniform_(self.word_embed.weight, -1.0, 1.0)

    def build_model(self):
        # whenever the embedding sees the padding index it'll make the whole vector zeros

        char_embedding = nn.Embedding(
            num_embeddings=VOCAB_CHARS + 1,
            embedding_dim=self.char_dim,
            padding_idx=self.char_pad
        )
        word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.embed_dim,
            padding_idx=self.seq_pad_idx
        )
        # design LSTM
        lstm = nn.LSTM(
            input_size=self.embed_dim + self.char_dim,
            hidden_size=self.hidden_layer_lstm,
            bidirectional=True,
            num_layers=2
        )

        # output layer which projects back to tag space
        mlp = nn.Sequential(nn.Linear(2 * self.hidden_layer_lstm, self.output_dim),
                            nn.Softmax(dim=1))
        return lstm, word_embedding, char_embedding, mlp

    def forward(self, x, y):
        lens = list(map(len, x))
        embed_word = self.embed_word(x)
        embed_char = self.make_char_lstm(x).split(lens)
        # concat embed word & embed char
        concat = self.cat_embeds(embed_word, embed_char, lens)
        x = pack_padded_sequence(concat, lens, batch_first=True, enforce_sorted=False)
        # now run through LSTM
        output, (c0, ho) = self.lstm(x)
        # undo the packing operation
        x, lens = pad_packed_sequence(output, batch_first=True)
        Y_hats = self.mlp(x)
        y = pad_sequence(y, batch_first=True, padding_value=self.label_pad_idx)
        return Y_hats, y

    def word2chars(self, word_idx):
        word = self.idx_to_word[int(word_idx)]
        chars = list(map(ord, word))
        chars = list(map(torch.tensor, chars))
        return torch.tensor(chars)

    def make_char_lstm(self, batch):
        chars = [[self.word2chars(word) for word in x] for x in batch]
        chars = sum(chars, [])
        lens = list(map(len, chars))
        chars = pad_sequence(chars, batch_first=True, padding_value=self.char_pad)
        chars = self.char_embed(chars)
        chars = pack_padded_sequence(chars, lens, batch_first=True, enforce_sorted=False)
        outputs, (c0, h0) = self.lstm_chars(chars)
        x, lens = pad_packed_sequence(outputs, batch_first=True)
        last_seq = x[torch.arange(x.shape[0]), lens - 1]
        return last_seq

    def embed_word(self, x):
        x = pad_sequence(x, batch_first=True, padding_value=self.seq_pad_idx)
        x = self.word_embed(x)
        return x

    def cat_embeds(self, w_emb, c_emb, lens):
        shape = w_emb.shape[0], w_emb.shape[1], self.embed_dim + self.char_dim
        res = torch.zeros(shape, dtype=torch.float)
        for i, c in enumerate(c_emb):
            res[i, :lens[i], :] = torch.cat((w_emb[i, :lens[i], :], c), dim=1)
        return res

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    isPos = False
    train_loader, valid_loader, vocab_sample, vocab_label, word_to_idx, labels_to_idx, idx_to_word, _ = process_data(
        isPos, BATCH_SIZE, DEV_BATCH_SIZE)

    if not isPos:
        weights = [1.0, 1.0, 0.1, 1.0, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=labels_to_idx['PAD'], reduction='mean')
    else:
        loss_func = nn.CrossEntropyLoss(ignore_index=labels_to_idx['PAD'], reduction='mean')

    model = BiLSTM_concat(len(labels_to_idx), len(word_to_idx), idx_to_word)
    model = train_and_eval_model(model, train_loader, valid_loader, loss_func, labels_to_idx, EPOCHS, LR)
