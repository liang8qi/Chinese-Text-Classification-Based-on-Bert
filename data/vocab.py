import pickle
import collections
from collections import OrderedDict
import os
import logging
import jieba
import random

class Vocab(object):

    def __init__(self, vocab_src, vocab_size, file_src=None):
        # participle
        self.id2word = {}
        self.word2id = {}

        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        if not os.path.exists(vocab_src):
            self.get_vocab(file_src, vocab_src)

        self.load(vocab_src, vocab_size)
        print("The total 30 words are: ")
        for index in range(30):
            print("{}: {}".format(index, self.id2word[index]))

    def load(self, vocab_src, vocab_size):
        #
        if not os.path.exists(vocab_src):
            print("There is no file named {}".format(vocab_src))
            assert False

        # load special char
        # PAD=0, UNK=1, START=2, END=3
        for id, word in enumerate([self.pad, self.unk, self.bos, self.eos]):
            self.word2id[word] = id
            self.id2word[id] = word

        word_count = []
        with open(vocab_src, 'rb') as f:
            word_count = pickle.load(f)

        if vocab_size is None:
            remained_words = word_count
        else:
            remained_words = word_count.most_common(vocab_size)

        for id, (token, _) in enumerate(remained_words, 4):
            self.word2id[token] = id
            self.id2word[id] = token

        print("The num of total words is {}, the final vocab size is {}".format(len(word_count), self.size()))

    def size(self):
        return len(self.word2id)

    def lookup(self, word, default=None):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return default

    def convert_to_idx(self, seq, bos_word=None, eos_word=None):
        unk = self.lookup(self.unk)
        vec = [self.lookup(word, unk) for word in seq]

        if bos_word is not None:
            vec = [self.lookup(bos_word)] + vec

        if eos_word is not None:
            vec += [self.lookup(eos_word)]

        return vec

    def get_vocab(self, file_src, vocab_src):
        vocab = []
        # print(file_src)
        with open(file_src, 'r', encoding='utf=8') as f:
            for line in f.readlines():
                words = line.strip().split()[1:]
                for word in words:
                    vocab.append(word)
        results = collections.Counter(vocab).most_common()
        # print("results is {}".format(len(results)))
        with open(vocab_src, 'wb') as fs:  # dict转josn
            pickle.dump(results, fs)
        print("Finished vocab built, vocab has been saved at {}".format(vocab_src))
        # with open(vocab_src, 'a', encoding='utf-8') as f:
        #     for pair in results:
        #         f.write("{} {}\n".format(pair[0], pair[1]))


def file_proprecess(data_src, processed_src, is_training, is_participle=False, splied_dataset=True,
                    valid__src=None, ratio=0.5):

    if splied_dataset:
        output_src = "Finished preprocessed, total_examples: {}, positive examples: {}, " \
                     "negative examples: {}, train_data_size: {}"
    else:
        output_src = "Finished preprocessed, total_examples: {}, positive examples: {}, " \
                     "negative examples: {}, results have been saved: {}"
    results = []

    positive_label = 0
    negative_label = 0
    with open(data_src, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if is_training:
                label, sentence = line.strip().split()
                if label == '1':
                    positive_label += 1
                else:
                    negative_label += 1
            else:
                sentence = line.strip()

            if is_participle:
                splited_re = jieba.cut(sentence, cut_all=True)
                # print(type(splited_re))
                new_sentence = " ".join(splited_re)
            else:
                new_sentence = [ch for ch in sentence]
                new_sentence = " ".join(new_sentence)
            if is_training:
                results.append("{} {}\n".format(label, new_sentence))
                # out_file.write("{} {}\n".format(label, new_sentence))
            else:
                results.append("{}\n".format(new_sentence))
                # out_file.write("{}\n".format(new_sentence))

    cnt = len(results)

    if splied_dataset:
        random.shuffle(results)
        index = int(len(results)*ratio)
        valid = results[index:]
        results = results[:index]

        with open(valid__src, 'w', encoding='utf-8') as f:
            for result in valid:
                f.write(result)

    with open(processed_src, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result)

    if splied_dataset:
        print(output_src.format(cnt, positive_label, negative_label, len(results)))
    else:
        print(output_src.format(cnt, positive_label, negative_label, processed_src))


if __name__ == '__main__':
    train_data_src = '../newdataset-7/filtered_train.txt'
    train_preprocessed_src = '../newdataset-7/char_level/train_processed.txt'
    valid_preprocessed_src = '../newdataset-7/char_level/valid_processed.txt'
    test_data_src = '../dataset/test_handout'
    test_preprocessed_src = '../newdataset-7/char_level/test_processed.txt'

    ## processing train dataset
    file_proprecess(train_data_src, train_preprocessed_src, True, is_participle=False, splied_dataset=True,
                    valid__src=valid_preprocessed_src, ratio=0.95)

    ## processing test dataset
    file_proprecess(test_data_src, test_preprocessed_src, False, is_participle=False, splied_dataset=False,
                    valid__src=None)

# Finished preprocessed, total_examples: 13565, positive examples: 4872, negative examples: 8693, train_data_size: 12886
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-2/char_level/test_processed.txt
# avg
# Finished preprocessed, total_examples: 13631, positive examples: 4892, negative examples: 8739, train_data_size: 12949
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-3/char_level/test_processed.txt

# Finished preprocessed, total_examples: 13631, positive examples: 4892, negative examples: 8739, train_data_size: 13358
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-3/char_level/test_processed.txt

# Finished preprocessed, total_examples: 13631, positive examples: 4892, negative examples: 8739, train_data_size: 13085
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-3/char_level/test_processed.txt

# dataset4
# Finished preprocessed, total_examples: 13631, positive examples: 4892, negative examples: 8739, train_data_size: 12949
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-4/char_level/test_processed.txt

# dataset5
# Finished preprocessed, total_examples: 13631, positive examples: 4880, negative examples: 8751, train_data_size: 12949
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-5/char_level/test_processed.txt

# dataset-6
# Finished preprocessed, total_examples: 13565, positive examples: 4872, negative examples: 8693, train_data_size: 12886
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-6/char_level/test_processed.txt

# dataset-7
# Finished preprocessed, total_examples: 13631, positive examples: 4892, negative examples: 8739, train_data_size: 12949
# Finished preprocessed, total_examples: 4189, positive examples: 0, negative examples: 0, results have been saved: ../newdataset-7/char_level/test_processed.txt
