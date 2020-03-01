import torch
from models.bert_based_model import BertBasedModel
from data.dataloader import BertRecommendDataset, bert_batch_preprocessing
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from data import config

import csv


class Test(object):

    def __init__(self, hidden_size, classifier_hsz, categories, model_file_src):
        self.model = BertBasedModel(hidden_size=hidden_size, classifier_hsz=classifier_hsz,
                                    categories=categories, model_file_src=model_file_src).cuda()

    def test(self, test_data_src, result_saved=None, labels_saved_src=None, is_training=False):
        test_dataset = BertRecommendDataset(file_src=test_data_src, is_training=is_training)

        test_batches = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False,
                                                   collate_fn=bert_batch_preprocessing)

        y_test_preds = []
        y_test_s = []
        y_pred_lables = []
        for test_batch in test_batches:
            labels, seq_ids, lens, masks, seqs = test_batch

            if labels[0]:
                labels = labels.cuda()
            seq_ids = seq_ids.cuda()
            masks = masks.cuda()

            probs = self.model(seq_ids, masks)
            pred_label = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
            y_pred_lables += pred_label
            probs = probs[:, 1]
            if labels[0] is not None:
                labels = labels.cpu().numpy().tolist()
                y_test_s += labels
            y_test_preds += probs.tolist()

        csvlists = [[i, y_test_preds[i]] for i in range(len(y_test_preds))]
        with open(result_saved, 'w', newline='') as f:
            f_cvs = csv.writer(f)
            headers = ['ID', 'Prediction']
            f_cvs.writerow(headers)
            f_cvs.writerows(csvlists)
        with open(labels_saved_src, 'w', encoding='utf=8') as f:
            for val in csvlists:
                f.write("{} {}\n".format(val[0], val[1]))

        print("Finished, total {} examples, and results have been saved at: {}".format(len(y_test_preds), result_saved))


if __name__ == '__main__':
    model_file_src = '../Result/models/model_714'
    result_saved = '../Result/submission_random.csv'
    labels_saved = '../Result/labels.txt'
    test_file_src = config.test_file_src
    test = Test(hidden_size=config.bert_hsz, classifier_hsz=config.classifier_hsz,
                categories=2, model_file_src=model_file_src)

    test.test(test_data_src=test_file_src, result_saved=result_saved, labels_saved_src=labels_saved, is_training=False)