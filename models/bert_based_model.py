import torch
import torch.nn as nn

from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, hidden_size, classifier_hsz, categories):
        super(BertClassifier, self).__init__()

        self.linear = nn.Linear(in_features=hidden_size, out_features=classifier_hsz, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        self.batch_normalization = nn.BatchNorm1d(num_features=classifier_hsz)

        self.classifier = nn.Linear(in_features=classifier_hsz, out_features=categories, bias=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, features):

        features = self.dropout(features)

        dense_features = self.linear(features)
        dense_features = torch.relu(dense_features)

        normalized = self.batch_normalization(dense_features)

        probs = self.classifier(normalized)
        probs = torch.softmax(probs, dim=-1)

        return probs


class BertBasedModel(nn.Module):

    def __init__(self, hidden_size, classifier_hsz, categories, model_file_src=None):
        super(BertBasedModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.classifier = BertClassifier(hidden_size=hidden_size, classifier_hsz=classifier_hsz, categories=categories)

        if model_file_src is not None:
            state = torch.load(model_file_src, map_location=lambda storage, location: storage)
            self.bert.load_state_dict(state_dict=state['bert_state_dict'])
            self.classifier.load_state_dict(state_dict=state['classifier_state_dict'])
            print("Loading model {} successfully".format(model_file_src))

    def forward(self, seq_ids, masks):

        last_hidden_state, pooler_output= self.bert(seq_ids, attention_mask=masks)

        probs = self.classifier(pooler_output)

        return probs