import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

import numpy as np


class BertRecommendDataset(Dataset):
    def __init__(self, file_src, is_training):
        sequence_list = []
        label_list = []
        with open(file_src, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                if is_training:
                    lable, sequence = line[0], line[1:]
                    sequence_list.append(sequence)
                    label_list.append(int(lable))
                else:
                    sequence_list.append(line)
        print("The size of dataset {} is: {}".format(file_src, len(sequence_list)))
        self.sequence_list = sequence_list
        self.label_list = label_list
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.is_training = is_training

    def __getitem__(self, item):

        sequence = ['[CLS]'] + self.sequence_list[item] + ['[SEP]']
        sequence = " ".join(sequence)
        # print(type(sequence))
        sequence = self.tokenizer.tokenize(sequence)
        seq_ids = self.tokenizer.convert_tokens_to_ids(sequence)

        if self.is_training:
            # print(type(seq_ids[0]))
            return self.label_list[item], \
                   torch.tensor(seq_ids).long(), \
                   len(seq_ids), self.sequence_list[item]
        else:
            return None, torch.tensor(seq_ids).long(), len(seq_ids), self.sequence_list[item]

    def __len__(self):
        return len(self.sequence_list)


def bert_batch_preprocessing(batch):
    labels, seq_ids, lens, seqs = zip(*batch)
    seq_ids = pad_sequence(seq_ids, batch_first=True, padding_value=0)

    bsz, max_len = seq_ids.size()
    masks = np.zeros([bsz, max_len], dtype=np.float)

    for index, seq_len in enumerate(lens):
        masks[index][:seq_len] = 1

    masks = torch.from_numpy(masks)

    if labels[0] is not None:
        labels = torch.tensor(labels)
    lens = torch.tensor(lens)

    return labels, seq_ids, lens, masks, seqs
