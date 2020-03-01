import torch


def cross_entropy_loss(probs, labels):
    # print(labels)
    y_pred = torch.argmax(probs, dim=-1)
    cross_entropy = torch.nn.CrossEntropyLoss()
    # minus = probs[:, 0] - probs[:, 1]
    # minus = minus.unsqueeze(1)
    # print(minus.size())
    # l2 = torch.norm(minus).mean()
    # print(l2)

    loss = cross_entropy(probs, labels)

    return probs[:, 1], y_pred, loss