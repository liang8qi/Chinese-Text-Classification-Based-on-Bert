import time
import random

import jieba

from data import config


def repetition_filter(file_src, outfile_path):

    positive_file = []
    negative_file = []
    file_dict = {}
    cnt = 0
    start_time = time.time()
    with open(file_src, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cnt += 1
            line = line.strip()
            label, sentence = line.split()
            if sentence not in file_dict:
                file_dict[sentence] = []
            file_dict[sentence].append(int(label))

    for sentence in file_dict:
        labels = file_dict[sentence]
        flag = sum(labels)/len(labels)
        if flag > 0.5:
            # 1
            positive_file.append(sentence)
        else:
            negative_file.append(sentence)

    with open(outfile_path, 'w', encoding='utf-8') as f:
        for sentence in positive_file:
            f.write("1 {}\n".format(sentence))

        for sentence in negative_file:
            f.write("0 {}\n".format(sentence))

    print("Finished, total: {}, positive: {}, negative: "
          "{}, spending: {}".format(cnt, len(positive_file), len(negative_file), time.time()-start_time))


def file_preprocess(data_src, preprocess_src, is_training, is_participle=False, divided_dataset=True,
                    valid_src=None, ratio=0.5):

    if divided_dataset:
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
                new_sentence = " ".join(splited_re)
            else:
                new_sentence = [ch for ch in sentence]
                new_sentence = " ".join(new_sentence)
            if is_training:
                results.append("{} {}\n".format(label, new_sentence))
            else:
                results.append("{}\n".format(new_sentence))

    cnt = len(results)

    if divided_dataset:
        random.shuffle(results)
        index = int(len(results)*ratio)
        valid = results[index:]
        results = results[:index]

        with open(valid_src, 'w', encoding='utf-8') as f:
            for result in valid:
                f.write(result)

    with open(preprocess_src, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result)

    if divided_dataset:
        print(output_src.format(cnt, positive_label, negative_label, len(results)))
    else:
        print(output_src.format(cnt, positive_label, negative_label, preprocess_src))


if __name__ == '__main__':
    # 数据预处理，处理多标签和重复数据
    unprocessed_train_file = config.unprocessed_train_file
    filtered_file = config.filtered_file

    repetition_filter(unprocessed_train_file, filtered_file)

    # 将训练集划分成训练集和验证集
    train_data_file = config.train_file_src
    valid_data_file = config.valid_file_src
    divided_ratio = config.data_divided_ratio

    # 因为中文版预训练的Bert是基于字的，所有这里只处理训练集
    file_preprocess(filtered_file, train_data_file, True, is_participle=False, divided_dataset=True,
                    valid_src=valid_data_file, ratio=divided_ratio)


