import os
import time
import torch
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter

from models.bert_based_model import BertBasedModel
from models import loss_functions

from data.dataloader import BertRecommendDataset
from data.dataloader import bert_batch_preprocessing
from data import utils
from data import config

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score



torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class Train(object):

    def __init__(self):

        self.model_dir = os.path.join(config.model_dir, 'models')
        self.summary_dir = os.path.join(config.model_dir, 'summary')
        for path in [config.model_dir, self.model_dir, self.summary_dir]:
            if not os.path.exists(path):
                os.mkdir(path)

        # training
        self.max_epoches = config.max_epoches
        self.eval_every = config.eval_every
        self.batch_size = 128
        self.summary_flush_every = config.summary_flush_every
        self.report_every = config.report_every

        # model
        self.bert_hsz= config.bert_hsz
        self.classifier_hsz = config.classifier_hsz
        self.categories = config.categories

    def set_train(self, model_file_path=None):

        self.model = BertBasedModel(hidden_size=self.bert_hsz, classifier_hsz=self.classifier_hsz,
                                    categories=self.categories).cuda()

        self.summary_writer = SummaryWriter(log_dir=self.summary_dir)
        bert_parameters = list(self.model.bert.parameters())
        bert_named_parameters = list(self.model.bert.named_parameters())
        classifier_parameters = list(self.model.classifier.parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
             'lr': config.bert_lr},
            {'params': [p for n, p in bert_named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': config.bert_lr},
            {'params': classifier_parameters}]

        self.optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.lr)

    def save_model(self, loss, itr):
        state = {
            "itrs": itr,
            'bert_state_dict': self.model.bert.state_dict(),
            'classifier_state_dict': self.model.classifier.state_dict(),
            'loss': loss
        }

        model_saved_path = os.path.join(self.model_dir, 'model_{}'.format(itr))
        torch.save(state, model_saved_path)

    def train_one_batch(self, labels, seq_ids, masks):

        probs = self.model(seq_ids, masks)
        probs, y_pred, loss = loss_functions.cross_entropy_loss(probs, labels)

        return probs.cpu().detach().numpy(), loss, y_pred.cpu().numpy()

    def train_itrs(self, train_file_src, valid_file_src):
        report_print = "At: epoch:{}: {}, loss: {}, total_time: {}"
        eval_report_print = 'At: epoch{}: {}, the eval result is {}, best result is {}, at {}: {}'

        train_dataset = BertRecommendDataset(file_src=train_file_src, is_training=True)
        validation = BertRecommendDataset(file_src=valid_file_src, is_training=True)

        valid_data_size = validation.__len__() // 10

        self.set_train()

        best_eval_result = 0.0
        best_result_itr = 0
        best_epoch = 0

        total_times = time.time()
        itrs = 0

        train_batches = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                                    collate_fn=bert_batch_preprocessing)

        validation_batches = torch.utils.data.DataLoader(dataset=validation,
                                                         batch_size=valid_data_size, shuffle=False,
                                                         collate_fn=bert_batch_preprocessing)

        loss = 0
        for epoch in range(self.max_epoches):

            for batch in train_batches:
                labels, seq_ids, lens, masks, seqs = batch
                labels = labels.cuda()
                seq_ids = seq_ids.cuda()
                masks = masks.cuda()

                self.optimizer.zero_grad()

                probs, loss, y_pred = self.train_one_batch(labels, seq_ids, masks)

                loss.backward()

                self.optimizer.step()

                itrs += 1
                if itrs % self.summary_flush_every == 0:
                    self.summary_writer.add_scalar(tag='train/loss',
                                                   scalar_value=loss.item(), global_step=itrs)
                    self.summary_writer.flush()

                if itrs % self.report_every == 0:
                    print(report_print.format(epoch, itrs, loss, utils.get_time(total_times)))

            if epoch % self.eval_every == 0:
                self.model.eval()

                y_preds = []
                y_probs = []
                ys = []
                for valid_batch in validation_batches:
                    labels, seq_ids, lens, masks, seqs = valid_batch

                    labels = labels.cuda()
                    seq_ids = seq_ids.cuda()
                    masks = masks.cuda()

                    probs, _, y_pred = self.train_one_batch(labels, seq_ids, masks)

                    labels = labels.cpu().numpy().tolist()
                    y_preds += y_pred.tolist()
                    y_probs += probs.tolist()
                    ys += labels

                accuracy = accuracy_score(ys, y_preds)
                precision = precision_score(ys, y_preds)
                f1 = f1_score(ys, y_preds)
                auc_score = roc_auc_score(ys, y_probs)

                if best_eval_result < auc_score:
                    best_eval_result = auc_score
                    best_result_itr = itrs
                    best_epoch = epoch
                    self.save_model(loss, itrs)
                self.summary_writer.add_scalar(tag='train/auc_score',
                                               scalar_value=auc_score, global_step=itrs)

                print(eval_report_print.format(epoch, itrs, [accuracy, precision, f1, auc_score],
                                               best_eval_result,
                                               best_epoch, best_result_itr))

                self.model.train()


if __name__ == '__main__':
    train = Train()

    train.train_itrs(train_file_src=config.train_file_src, valid_file_src=config.valid_file_src)
