STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import json
import os
import pickle
import sys

from process_data_3 import read_data_test, make_loader_test
from bilstm import BiLSTM
from biLSTM_chars import BiLSTM_chars
from biLSTM_sufpref import BiLSTM_sufpref
from biLSTM_concat import BiLSTM_concat


def get_key(val, labels_to_idx):
    for key, value in labels_to_idx.items():
        if val == value:
            return key
    return None


def create_test(path, model, test_loader, labels_to_idx, idx_to_word, original_samples):
    model.eval()
    counter = 0
    with open(path, 'w') as fwriter:
        for (x, y) in test_loader:
            predict, y = model(x, y)
            predict = predict[0]
            for j, word in enumerate(predict):
                p = word.argmax()
                label = get_key(p, labels_to_idx)
                # sample = idx_to_word[int(x[0][j])]
                row = original_samples[counter] + ' ' + label + '\n'
                counter += 1
                fwriter.write(row)
            fwriter.write('\n')
    fwriter.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    choice = args[0]
    model_file = args[1]
    input_file = args[2]
    isPos = int(args[3])

    dir_path = os.path.dirname(model_file)

    with open(os.path.join(dir_path, 'word_to_idx.p'), 'rb') as fp1:
        word_to_idx = pickle.load(fp1)
    with open(os.path.join(dir_path, 'labels_to_idx.p'), 'rb') as fp2:
        labels_to_idx = pickle.load(fp2)
    if choice == 'c':
        with open(os.path.join(dir_path, 'idx_to_pre_suf.p'), 'rb') as fp3:
            idx_to_pre_suf = pickle.load(fp3)
    if choice == 'd' or 'b':
        with open(os.path.join(dir_path, 'idx_to_word.p'), 'rb') as fp4:
            idx_to_word = pickle.load(fp4)

    isSub = True if choice == 'c' else False

    if choice == 'a':
        model = BiLSTM(len(labels_to_idx), len(word_to_idx))
    elif choice == 'b':
        model = BiLSTM_chars(len(labels_to_idx), idx_to_word)
    elif choice == 'c':
        model = BiLSTM_sufpref(len(labels_to_idx), len(word_to_idx), idx_to_pre_suf)
    elif choice == 'd':
        model = BiLSTM_concat(len(labels_to_idx), len(word_to_idx), idx_to_word)
    else:
        raise ValueError("illegal repr choice")

    model.load(model_file)

    seqseq, word_to_idx, labels_to_idx, vocab_sample_all, vocab_label_all, idx_to_word, idx_to_pre_suf, original_samples = read_data_test(
        input_file, isPos == isPos, word_to_idx=word_to_idx, labels_to_idx=labels_to_idx, isSub=isSub)
    test_loader = make_loader_test(seqseq, 1)
    data_type = 'pos' if isPos else 'ner'
    create_test('test4.{0}'.format(data_type), model, test_loader, labels_to_idx, idx_to_word, original_samples)
