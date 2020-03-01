


# preprocess
unprocessed_train_file = "../dataset/train_shuffle"
filtered_file = "../dataset/preprocessed_data/filtered_train.txt"
data_divided_ratio = 0.95

# bert based model
pretrained_model_name = 'bert-base-chinese'
bert_hsz = 768
classifier_hsz = 256
categories = 2


# train
train_file_src = '../dataset/preprocessed_data/train_processed.txt'
valid_file_src = '../dataset/preprocessed_data/valid_processed.txt'
max_epoches = 50
eval_every = 1
model_dir = '../Result'
using_pre_trained_emb = False
batch_size = 128
bert_lr = 2e-5
lr = 0.01
summary_flush_every = 100
report_every = 10

# test
test_file_src = '../dataset/test_handout'

