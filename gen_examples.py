STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import random
from xeger import Xeger


def generate_word(regex):
    """generate word according to regex received, limit the lrngth of the word up to 100 letters"""
    limit = 17
    example = Xeger(limit=limit)
    example = example.xeger(regex)
    while len(example) > 100:
        limit = limit - 2
        example = Xeger(limit=limit)
        example = example.xeger(regex)
    return example


def generate_sequences(path, sign_regex):
    """generate 500 examples of regex given and write to file"""
    examples = set()
    with open(path, mode='w') as fwriter:
        while len(examples) < 500:
            example = generate_word(sign_regex)
            examples.add(example)
            fwriter.write(example + "\n")
    fwriter.close()
    return examples


def create_data(path, good_reg, bad_reg, data_len):
    """create train / dev datasets to lstm acceptor"""
    data = set()
    label_dicd = {0: bad_reg, 1: good_reg}
    while len(data) < data_len:
        label = random.randint(0, 1)
        sample = generate_word(label_dicd[label])
        data.add((sample, label))
    with open(path, 'w') as fwriter:
        for sample, label in data:
            fwriter.write(sample + '\t' + str(label) + '\n')
    fwriter.close()


def create_data_anbn(path, data_len):
    data = set()
    while len(data) < data_len:
        label = random.randint(0, 1)
        if label == 1:
            sample = generate_good_anbn()
        else:
            sample = generate_bed_anbn()
        data.add((sample, label))
    with open(path, 'w') as fwriter:
        for sample, label in data:
            fwriter.write(sample + '\t' + str(label) + '\n')
    fwriter.close()


if __name__ == "__main__":
    good_regex = r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
    bad_regex = r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'
    # pos_sequences = generate_sequences("./pos_examples", good_regex)
    # neg_sequences = generate_sequences("./neg_examples", bad_regex)
    # print(pos_sequences, neg_sequences)
    create_data('./train', good_regex, bad_regex, 10000)
    create_data('./dev', good_regex, bad_regex, 2000)
